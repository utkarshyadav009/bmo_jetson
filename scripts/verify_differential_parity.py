#!/usr/bin/env python3
"""scripts/verify_differential_parity.py

Differential activation and parity benchmark comparing:
1. Golden PyTorch Moshi (unquantized BF16)
2. BMOEngine libbmo.so (clean Q4_0 GGUF)

Evaluates:
- Temporal residual activations (z) cosine similarity, relative L2 error, max absolute error.
- Text logits cosine similarity, argmax parity, top-5 token intersection.
- Depth cascade (8 codebooks) audio logits cosine similarity and codebook argmax match rate.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure local imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

# Ensure moshi path
MOSHI_PATH = "/home/jovyan/work/BMO-Project/personaplex_repo/moshi"
if MOSHI_PATH not in sys.path:
    sys.path.insert(0, MOSHI_PATH)

import sentencepiece as spm
from bmo_engine import BMOEngine
from moshi.models.loaders import _lm_kwargs
from moshi.models.lm import LMModel
import safetensors.torch

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64).ravel()
    b_f = b.astype(np.float64).ravel()
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_f, b_f) / (norm_a * norm_b))

def rel_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64).ravel()
    b_f = b.astype(np.float64).ravel()
    denom = max(float(np.linalg.norm(a_f)), 1e-12)
    return float(np.linalg.norm(a_f - b_f) / denom)

def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))

def main():
    torch.backends.cuda.enable_cudnn_sdp(False)

    gguf_path = os.environ.get("BMO_GGUF", "/home/jovyan/work/BMO Voice Engine/standard_moshi_8cb/bmo_moshi_8cb_clean_q4.gguf")
    safetensors_path = os.environ.get("BMO_SAFETENSORS", "/home/jovyan/work/BMO-Project/models/moshiko/model.safetensors")
    spm_path = os.environ.get("BMO_SPM_PATH", "/home/jovyan/work/BMO-Project/models/moshiko/tokenizer_spm_32k_3.model")

    print(f"[PARITY] Loading SPM tokenizer from: {spm_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)

    print(f"[PARITY] Loading Golden PyTorch Moshi BF16 from: {safetensors_path}")
    kwargs = dict(_lm_kwargs)
    kwargs["dep_q"] = 8
    pt_model = LMModel(device="cuda", dtype=torch.bfloat16, **kwargs)
    state = safetensors.torch.load_file(safetensors_path, device="cuda")
    pt_model.load_state_dict(state, strict=False)
    pt_model.to(device="cuda", dtype=torch.bfloat16)
    pt_model.eval()

    print(f"[PARITY] Loading BMOEngine libbmo.so from: {gguf_path}")
    engine = BMOEngine(gguf_path, n_ctx=512)

    n_steps = 15
    print("\n" + "="*96)
    print(f"  RUNNING DIFFERENTIAL ACTIVATION PARITY HARNESS ({n_steps} STEPS)")
    print("="*96)

    # Standard silence codebooks for Mimi
    user_silence = np.array([1049, 243, 783, 1562, 340, 2010, 183, 1665], dtype=np.int32)
    prev_user_audio = np.full(8, 2048, dtype=np.int32)
    prev_agent_text = 32000
    prev_agent_audio = np.full(8, 2048, dtype=np.int32)

    tokens_17_np = np.zeros(17, dtype=np.int32)

    results = []

    with torch.no_grad():
        with pt_model.streaming(batch_size=1):
            for step in range(n_steps):
                # 17-Token Interleave schedule
                if step == 0:
                    tokens_17_np[0] = 32000
                    tokens_17_np[1:] = 2048
                elif step == 1:
                    tokens_17_np[0] = prev_agent_text
                    tokens_17_np[1] = prev_agent_audio[0]
                    tokens_17_np[2:9] = 2048
                    tokens_17_np[9] = user_silence[0]
                    tokens_17_np[10:17] = 2048
                else:
                    tokens_17_np[0] = prev_agent_text
                    tokens_17_np[1:9] = prev_agent_audio
                    tokens_17_np[9] = user_silence[0]
                    tokens_17_np[10:17] = prev_user_audio[1:8]

                # PyTorch forward
                tokens_pt = torch.from_numpy(tokens_17_np).view(1, 17, 1).long().to("cuda")
                z_pt_tensor, text_logits_pt_tensor = pt_model.forward_codes(tokens_pt)
                z_pt = z_pt_tensor[0, 0].float().cpu().numpy()
                text_logits_pt = text_logits_pt_tensor[0, 0, 0].float().cpu().numpy()

                # BMOEngine forward
                z_bmo, text_logits_bmo = engine.forward_temporal(tokens_17_np, copy=True)

                # Temporal comparisons
                cos_z = cosine_similarity(z_pt, z_bmo)
                l2_z = rel_l2_error(z_pt, z_bmo)
                max_diff_z = max_abs_diff(z_pt, z_bmo)

                # Text logits comparisons
                valid_logits_pt = text_logits_pt[:32000]
                valid_logits_bmo = text_logits_bmo[:32000]
                cos_text = cosine_similarity(valid_logits_pt, valid_logits_bmo)

                top_pt = int(np.argmax(valid_logits_pt))
                top_bmo = int(np.argmax(valid_logits_bmo))

                top5_pt = set(np.argsort(valid_logits_pt)[-5:])
                top5_bmo = set(np.argsort(valid_logits_bmo)[-5:])
                top5_overlap = len(top5_pt.intersection(top5_bmo))

                # Depth cascade comparisons
                pt_depth_tokens = []
                pt_depth_logits_list = []
                prev_tok_pt = top_pt
                with pt_model.depformer.streaming(batch_size=1):
                    for cb in range(8):
                        seq_in = torch.tensor([[[prev_tok_pt]]], dtype=torch.long, device="cuda")
                        d_lgt = pt_model.forward_depformer(cb, seq_in, z_pt_tensor)
                        lgt_np = d_lgt[0, 0, 0, :2048].float().cpu().numpy()
                        pt_depth_logits_list.append(lgt_np)
                        top_audio_pt = int(np.argmax(lgt_np))
                        pt_depth_tokens.append(top_audio_pt)
                        prev_tok_pt = top_audio_pt

                # BMO depth cascade forward (greedy)
                bmo_depth_tokens = engine.forward_depth_cascade(top_bmo, z_bmo, temp=0.0, top_k=0)

                # Compare depth tokens
                depth_matches = sum(1 for p, b in zip(pt_depth_tokens, bmo_depth_tokens) if p == b)

                step_info = {
                    "step": step,
                    "cos_z": cos_z,
                    "rel_l2_z": l2_z,
                    "max_diff_z": max_diff_z,
                    "cos_text": cos_text,
                    "top_pt": top_pt,
                    "top_bmo": top_bmo,
                    "top5_overlap": top5_overlap,
                    "pt_depth": pt_depth_tokens,
                    "bmo_depth": bmo_depth_tokens.tolist(),
                    "depth_matches": depth_matches,
                }
                results.append(step_info)

                pt_piece = sp.id_to_piece(top_pt).replace(" ", " ")
                bmo_piece = sp.id_to_piece(top_bmo).replace(" ", " ")

                print(f"Step {step:2d} | cos(z)={cos_z:7.4f} | rel_L2(z)={l2_z:6.4f} | "
                      f"cos(text)={cos_text:7.4f} | PT:'{pt_piece}'({top_pt}) BMO:'{bmo_piece}'({top_bmo}) | "
                      f"top5_ovlp={top5_overlap}/5 | depth_match={depth_matches}/8 | "
                      f"audio[0..3]: PT={pt_depth_tokens[:4]} BMO={bmo_depth_tokens[:4].tolist()}")

                # Teacher-force next step using BMO tokens to track cumulative error under actual runtime loop
                prev_agent_text = top_bmo
                prev_agent_audio = bmo_depth_tokens
                prev_user_audio = user_silence.copy()

    print("\n" + "="*96)
    print("  DIFFERENTIAL ACTIVATION PARITY SUMMARY TABLE")
    print("="*96)
    print(f"{'Step':>4} | {'cos(z)':>8} | {'rel_L2(z)':>10} | {'cos(text)':>10} | {'Top5 Ovlp':>9} | {'Text Match':>10} | {'Depth Match':>11}")
    print("-" * 96)
    for r in results:
        text_match_str = "MATCH" if r["top_pt"] == r["top_bmo"] else f"MISMATCH ({r['top_pt']} vs {r['top_bmo']})"
        print(f"{r['step']:4d} | {r['cos_z']:8.4f} | {r['rel_l2_z']:10.4f} | {r['cos_text']:10.4f} | {r['top5_overlap']:5d} / 5 | {text_match_str:>10} | {r['depth_matches']:6d} / 8")

    mean_cos_z = float(np.mean([r["cos_z"] for r in results]))
    mean_rel_l2_z = float(np.mean([r["rel_l2_z"] for r in results]))
    mean_cos_text = float(np.mean([r["cos_text"] for r in results]))
    text_matches = sum(1 for r in results if r["top_pt"] == r["top_bmo"])
    mean_depth_matches = float(np.mean([r["depth_matches"] for r in results]))

    print("-" * 96)
    print(f"MEAN | {mean_cos_z:8.4f} | {mean_rel_l2_z:10.4f} | {mean_cos_text:10.4f} |           | {text_matches:5d} / {len(results)} | {mean_depth_matches:6.2f} / 8")
    print("="*96)

if __name__ == "__main__":
    main()

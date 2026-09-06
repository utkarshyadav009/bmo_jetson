#!/usr/bin/env python3
"""Diagnostic script for Moshi 8-CB: Step-by-Step Activation and Entropy Tracing.

Tests whether the model maintains coherent text/audio predictions across
timesteps 0, 1, 2... or whether KV-cache/RoPE/RMSNorm introduces divergence.
"""

import os
import sys
import numpy as np

# Performance & memory safeguards
os.environ.setdefault("NO_TORCH_COMPILE", "1")
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sentencepiece as spm
from bmo_engine import BMOEngine

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / (np.sum(e_x) + 1e-12)

def main():
    gguf_path = os.environ.get("BMO_GGUF", "bmo_moshi_8cb_clean_q4.gguf")
    spm_path = os.environ.get("BMO_SPM_PATH", "/home/bmo/bmo_models/moshi-common/tokenizer_spm_32k_3.model")

    sp = None
    if os.path.exists(spm_path):
        sp = spm.SentencePieceProcessor()
        sp.load(spm_path)
        print(f"Loaded tokenizer: {spm_path} (vocab={sp.get_piece_size()})")

    print(f"Initializing BMOEngine with model: {gguf_path}...")
    engine = BMOEngine(gguf_path, n_ctx=512)

    print("\n" + "="*80)
    print("  STEP-BY-STEP DIAGNOSTIC TRACE (15 STEPS)")
    print("="*80)

    # Moshi standard initial tokens
    # Text initial: 32000, Audio initial: 2048
    # Mimi silence tokens: [1049, 243, 783, 1562, 340, 2010, 183, 1665]
    user_silence = np.array([1049, 243, 783, 1562, 340, 2010, 183, 1665], dtype=np.int32)
    prev_user_audio = np.full(8, 2048, dtype=np.int32)
    prev_agent_text = 32000
    prev_agent_audio = np.full(8, 2048, dtype=np.int32)
    prev_agent_cb0 = 1049

    tokens_17 = np.zeros(17, dtype=np.int32)
    generated_text_tokens = []
    generated_audio_frames = []

    for step in range(15):
        # 17-Token Interleave invariant
        if step == 0:
            tokens_17[0] = 32000
            tokens_17[1:] = 2048
        elif step == 1:
            tokens_17[0] = prev_agent_text
            tokens_17[1] = prev_agent_audio[0]
            tokens_17[2:9] = 2048
            tokens_17[9] = user_silence[0]
            tokens_17[10:17] = 2048
        else:
            tokens_17[0] = prev_agent_text
            tokens_17[1:9] = prev_agent_audio
            tokens_17[9] = user_silence[0]
            tokens_17[10:17] = prev_user_audio[1:8]

        # Temporal pass
        z, text_logits = engine.forward_temporal(tokens_17)
        z_norm = float(np.linalg.norm(z))
        z_mean = float(np.mean(z))
        z_std = float(np.std(z))

        # Filter special range
        valid_logits = text_logits[:32000].copy()
        probs = softmax(valid_logits)
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        top_indices = np.argsort(valid_logits)[-5:][::-1]
        top_tok = int(top_indices[0])
        top_prob = float(probs[top_tok])

        # Depth cascade (greedy)
        curr_agent_audio = engine.forward_depth_cascade(top_tok, z, temp=0.0, top_k=0)

        # Decode frame mapping
        decode_audio = curr_agent_audio.copy()
        if step > 0:
            decode_audio[0] = prev_agent_cb0

        top_pieces = []
        if sp:
            for idx in top_indices[:3]:
                piece = sp.id_to_piece(int(idx)).replace(" ", " ")
                top_pieces.append(f"{idx}('{piece}': {probs[idx]*100:.1f}%)")

        print(f"Step {step:2d} | pos={engine.pos-1:2d} | ||z||={z_norm:6.2f} (μ={z_mean:+.3f}, σ={z_std:.3f}) | "
              f"H={entropy:5.2f} | top={top_tok} ({top_prob*100:5.1f}%) | "
              f"audio[0..3]={curr_agent_audio[:4].tolist()} | top3: {', '.join(top_pieces)}")

        # Update delay registers
        prev_agent_cb0 = curr_agent_audio[0]
        prev_agent_audio = curr_agent_audio
        prev_agent_text = top_tok
        prev_user_audio = user_silence.copy()

        if top_tok not in (0, 3, 32000):
            generated_text_tokens.append(top_tok)
        generated_audio_frames.append(decode_audio)

    if sp:
        full_text = sp.decode(generated_text_tokens)
        print("\n" + "="*80)
        print(f"Generated Text Transcripts (Non-pad/eos tokens): '{full_text}'")
        print("="*80)

if __name__ == "__main__":
    main()

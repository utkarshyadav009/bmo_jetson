#!/usr/bin/env python3
"""Stage 3: Offline End-to-End Audio Pipeline Benchmark for Jetson Orin Nano.

Validates:
1. Mimi audio encode (24 kHz PCM -> 8 codebooks) on CUDA.
2. Full-duplex 17-codebook input interleaving:
   - 1 text token
   - 8 user audio codebooks
   - 8 agent audio codebooks
3. BMOEngine CUDA Graph accelerated execution:
   - Temporal forward (32 layers)
   - Depth cascade (8 steps, 6 layers each)
4. Mimi audio decode (8 agent codebooks -> 24 kHz PCM) on CUDA.
5. Timing breakdown, Real-Time Factor (RTF), and P99 latency.
6. Audio artifact generation (output_bmo_response.wav).
"""

import os
import sys
import time
import psutil
import numpy as np

# Ensure safeguards before importing torch/moshi
os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
# Pin CPU threads for UMA bus protection
torch.set_num_threads(2)

import sphn
from moshi.models import get_mimi
from bmo_engine import BMOEngine


def get_vram_mb() -> float:
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)


def main():
    print("=" * 70)
    print("  BMO Stage 3: Offline End-to-End Audio Pipeline (Mimi + libbmo)")
    print("=" * 70)

    # 1. Paths
    gguf_path = os.environ.get(
        "BMO_GGUF",
        os.path.join(os.path.dirname(__file__), "bmo_moshi_8cb_q4.gguf")
    )
    mimi_path = os.environ.get(
        "BMO_MIMI_WEIGHT",
        "/home/bmo/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors"
    )
    input_wav = os.environ.get(
        "BMO_INPUT_WAV",
        "/home/bmo/bmo_ref_clip.wav"
    )
    output_wav = os.environ.get(
        "BMO_OUTPUT_WAV",
        os.path.join(os.path.dirname(__file__), "output_bmo_response.wav")
    )

    if not os.path.isfile(gguf_path):
        raise FileNotFoundError(f"GGUF model not found: {gguf_path}")
    if not os.path.isfile(mimi_path):
        raise FileNotFoundError(f"Mimi checkpoint not found: {mimi_path}")
    if not os.path.isfile(input_wav):
        raise FileNotFoundError(f"Input WAV not found: {input_wav}")

    print(f"[*] Input WAV:       {input_wav}")
    print(f"[*] Mimi Checkpoint: {mimi_path}")
    print(f"[*] GGUF Model:      {gguf_path}")
    print(f"[*] Output WAV:      {output_wav}")
    print(f"[*] Initial VmRSS:   {get_vram_mb():.1f} MB")

    # 2. Load Mimi on CUDA
    print("\n[+] Loading Mimi 24kHz audio codec on CUDA...")
    t_load_mimi_start = time.perf_counter()
    mimi = get_mimi(mimi_path, device="cuda")
    mimi.eval()
    mimi.streaming_forever(1)
    t_load_mimi = time.perf_counter() - t_load_mimi_start
    print(f"    Loaded Mimi in {t_load_mimi:.2f} s | VmRSS: {get_vram_mb():.1f} MB")

    # 3. Load BMOEngine & Capture CUDA Graphs
    print("\n[+] Initializing BMOEngine (n_ctx=512)...")
    t_load_bmo_start = time.perf_counter()
    engine = BMOEngine(gguf_path, n_ctx=512)
    t_load_bmo = time.perf_counter() - t_load_bmo_start
    print(f"    Loaded BMOEngine in {t_load_bmo:.2f} s | VmRSS: {get_vram_mb():.1f} MB")

    print("[+] Capturing CUDA Graphs for Temporal and Depth transformers...")
    rc_graph = engine.capture_graphs()
    if rc_graph != 0 or not engine.has_cuda_graphs():
        print(f"[!] WARNING: Graph capture returned {rc_graph}, proceeding in eager mode.")
    else:
        print(f"    CUDA Graphs captured successfully! (VmRSS: {get_vram_mb():.1f} MB)")

    # 4. Load & Preprocess Audio Clip
    print(f"\n[+] Loading input audio clip: {input_wav}...")
    raw_pcm, raw_sr = sphn.read(input_wav)
    print(f"    Source: {raw_pcm.shape} channels={raw_pcm.shape[0]}, sr={raw_sr} Hz")

    # Convert to mono if multi-channel
    if raw_pcm.shape[0] > 1:
        pcm_mono = raw_pcm.mean(axis=0, keepdims=True)
    else:
        pcm_mono = raw_pcm

    # Resample to 24000 Hz if necessary
    target_sr = 24000
    if raw_sr != target_sr:
        pcm_24k = sphn.resample(pcm_mono, src_sample_rate=raw_sr, dst_sample_rate=target_sr)
    else:
        pcm_24k = pcm_mono

    total_samples = pcm_24k.shape[-1]
    duration_s = total_samples / target_sr
    print(f"    Resampled: {pcm_24k.shape} mono @ {target_sr} Hz ({duration_s:.2f} s)")

    frame_size = int(target_sr * 0.080)  # 1920 samples = 80 ms per frame
    n_frames = int(np.ceil(total_samples / frame_size))
    print(f"    Frames: {n_frames} frames ({frame_size} samples/frame @ 12.5 Hz)")

    # Pad audio to exact multiple of frame_size
    pad_len = (n_frames * frame_size) - total_samples
    if pad_len > 0:
        pcm_padded = np.pad(pcm_24k, ((0, 0), (0, pad_len)), mode="constant")
    else:
        pcm_padded = pcm_24k

    # 5. Preallocate static GPU buffers and warmup pipeline
    print("\n[+] Warming up end-to-end pipeline (5 dummy frames)...")
    t_chunk_gpu = torch.empty((1, 1, frame_size), dtype=torch.float32, device="cuda")
    agent_tensor_gpu = torch.empty((1, 8, 1), dtype=torch.long, device="cuda")

    dummy_frame = torch.zeros((1, 1, frame_size), device="cuda", dtype=torch.float32)
    with torch.no_grad():
        for _ in range(5):
            c = mimi.encode(dummy_frame)
            toks = c[0, :, 0].detach().cpu().numpy().astype(np.int32)
            inp_toks = np.zeros(17, dtype=np.int32)
            inp_toks[1:9] = toks
            z, t_logits = engine.forward_temporal(inp_toks)
            prev_t = int(np.argmax(t_logits))
            for d in range(8):
                d_logits = engine.forward_depth(d, prev_t, z)
                prev_t = int(np.argmax(d_logits))
            dec_in = torch.zeros((1, 8, 1), device="cuda", dtype=torch.long)
            _ = mimi.decode(dec_in)

    mimi.reset_streaming()
    engine.reset()
    torch.cuda.synchronize()
    print("    Warmup complete. Streaming state and KV cache reset.")

    # 6. Stream frames through offline pipeline
    print(f"\n[+] Streaming {n_frames} frames through end-to-end pipeline...")
    latencies_enc = []
    latencies_temporal = []
    latencies_depth = []
    latencies_bmo = []
    latencies_dec = []
    latencies_total = []

    generated_pcm_frames = []
    agent_prev_audio = np.zeros(8, dtype=np.int32)
    current_text_token = 3  # PAD token
    tokens_17 = np.zeros(17, dtype=np.int32)

    start_rss = get_vram_mb()
    t_pipeline_start = time.perf_counter()

    with torch.no_grad():
        for frame_idx in range(n_frames):
            frame_start_sample = frame_idx * frame_size
            frame_chunk = pcm_padded[:, frame_start_sample : frame_start_sample + frame_size]

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Step A: Mimi Encode
            t_chunk_gpu.copy_(torch.from_numpy(frame_chunk))
            t_enc_start = time.perf_counter()
            user_codes = mimi.encode(t_chunk_gpu)  # [1, 8, 1]
            t_enc = (time.perf_counter() - t_enc_start) * 1000.0

            user_tokens = user_codes[0, :, 0].detach().cpu().numpy().astype(np.int32)

            # Step B: Interleave 17 tokens
            tokens_17[0] = current_text_token
            tokens_17[1:9] = user_tokens
            tokens_17[9:17] = agent_prev_audio

            # Step C: BMO Temporal Forward
            t_temp_start = time.perf_counter()
            z, text_logits = engine.forward_temporal(tokens_17)
            t_temporal = (time.perf_counter() - t_temp_start) * 1000.0

            next_text_token = int(np.argmax(text_logits))

            # Step D: BMO Depth Cascade (8 codebooks)
            t_depth_start = time.perf_counter()
            agent_tokens = np.zeros(8, dtype=np.int32)
            depth_prev = next_text_token
            for cb_idx in range(8):
                depth_logits = engine.forward_depth(cb_idx, depth_prev, z)
                depth_prev = int(np.argmax(depth_logits))
                agent_tokens[cb_idx] = depth_prev
            t_depth = (time.perf_counter() - t_depth_start) * 1000.0

            t_bmo = t_temporal + t_depth

            # Step E: Mimi Decode
            agent_tensor_gpu.copy_(torch.from_numpy(agent_tokens).view(1, 8, 1))
            t_dec_start = time.perf_counter()
            decoded_frame = mimi.decode(agent_tensor_gpu)  # [1, 1, 1920]
            torch.cuda.synchronize()
            t_dec = (time.perf_counter() - t_dec_start) * 1000.0

            t_total = (time.perf_counter() - t0) * 1000.0

            # Record frame output and update state
            pcm_out = decoded_frame[0, 0].detach().cpu().numpy()
            generated_pcm_frames.append(pcm_out)
            agent_prev_audio = agent_tokens
            current_text_token = next_text_token

            # Log metrics
            latencies_enc.append(t_enc)
            latencies_temporal.append(t_temporal)
            latencies_depth.append(t_depth)
            latencies_bmo.append(t_bmo)
            latencies_dec.append(t_dec)
            latencies_total.append(t_total)

            if (frame_idx + 1) % 20 == 0 or (frame_idx + 1) == n_frames:
                print(
                    f"    Frame {frame_idx + 1:3d}/{n_frames} | "
                    f"Enc: {t_enc:4.1f}ms | Temp: {t_temporal:4.1f}ms | "
                    f"Depth: {t_depth:4.1f}ms | Dec: {t_dec:4.1f}ms | "
                    f"Total: {t_total:4.1f}ms (Budget: 80.0ms)"
                )

    t_pipeline_total = time.perf_counter() - t_pipeline_start
    final_rss = get_vram_mb()

    # 7. Write Output Audio
    print(f"\n[+] Concatenating generated audio frames...")
    output_audio = np.concatenate(generated_pcm_frames, axis=0)
    # Trim to exact duration of input
    output_audio = output_audio[:total_samples]
    # Reshape to (1, N) for sphn
    output_audio_2d = output_audio[np.newaxis, :]
    sphn.write_wav(output_wav, output_audio_2d, target_sr)
    print(f"    Wrote output WAV to: {output_wav}")
    out_stat = os.stat(output_wav)
    print(f"    File size: {out_stat.st_size / 1024:.1f} KB | Duration: {output_audio.size / target_sr:.2f} s")

    # 8. Benchmark Metrics & Verification
    arr_enc = np.array(latencies_enc)
    arr_temp = np.array(latencies_temporal)
    arr_depth = np.array(latencies_depth)
    arr_bmo = np.array(latencies_bmo)
    arr_dec = np.array(latencies_dec)
    arr_total = np.array(latencies_total)

    budget_ms = 80.0
    rtf = np.median(arr_total) / budget_ms

    print("\n" + "=" * 70)
    print("  OFFLINE PIPELINE BENCHMARK REPORT")
    print("=" * 70)
    print(f"  Processed Frames:        {n_frames}")
    print(f"  Audio Duration:          {duration_s:.2f} s")
    print(f"  Wallclock Runtime:       {t_pipeline_total:.2f} s")
    print(f"  Median Frame Latency:    {np.median(arr_total):.1f} ms (Budget: {budget_ms:.1f} ms)")
    print(f"  P95 Frame Latency:       {np.percentile(arr_total, 95):.1f} ms")
    print(f"  P99 Frame Latency:       {np.percentile(arr_total, 99):.1f} ms")
    print(f"  Real-Time Factor (RTF):  {rtf:.3f}x (RTF < 1.0 is faster than real-time)")
    print("-" * 70)
    print(f"  Stage Latency Breakdown (Mean / Median / P95 / P99):")
    print(f"    - Mimi Encode:         {np.mean(arr_enc):5.1f} / {np.median(arr_enc):5.1f} / {np.percentile(arr_enc, 95):5.1f} / {np.percentile(arr_enc, 99):5.1f} ms")
    print(f"    - Temporal Transformer:{np.mean(arr_temp):5.1f} / {np.median(arr_temp):5.1f} / {np.percentile(arr_temp, 95):5.1f} / {np.percentile(arr_temp, 99):5.1f} ms")
    print(f"    - Depth Cascade (8x):  {np.mean(arr_depth):5.1f} / {np.median(arr_depth):5.1f} / {np.percentile(arr_depth, 95):5.1f} / {np.percentile(arr_depth, 99):5.1f} ms")
    print(f"    - libbmo Engine Total: {np.mean(arr_bmo):5.1f} / {np.median(arr_bmo):5.1f} / {np.percentile(arr_bmo, 95):5.1f} / {np.percentile(arr_bmo, 99):5.1f} ms")
    print(f"    - Mimi Decode:         {np.mean(arr_dec):5.1f} / {np.median(arr_dec):5.1f} / {np.percentile(arr_dec, 95):5.1f} / {np.percentile(arr_dec, 99):5.1f} ms")
    print(f"    - TOTAL FRAME LATENCY: {np.mean(arr_total):5.1f} / {np.median(arr_total):5.1f} / {np.percentile(arr_total, 95):5.1f} / {np.percentile(arr_total, 99):5.1f} ms")
    print("-" * 70)
    print(f"  Memory Footprint:")
    print(f"    - Pipeline Start VmRSS: {start_rss:.1f} MB")
    print(f"    - Pipeline End VmRSS:   {final_rss:.1f} MB")
    print(f"    - Delta VmRSS:          +{final_rss - start_rss:.2f} MB")
    print("=" * 70)

    # Verification assertions
    assert os.path.exists(output_wav), f"Output file does not exist: {output_wav}"
    assert out_stat.st_size > 1000, f"Output file is too small: {out_stat.st_size} bytes"
    assert not np.isnan(output_audio).any(), "Output audio contains NaN values!"
    assert not np.isinf(output_audio).any(), "Output audio contains Inf values!"

    print("\n[VERIFICATION RESULTS]")
    if np.median(arr_total) <= budget_ms * 1.15:
        print(f"  [PASS] Frame Latency is within real-time streaming margin ({np.median(arr_total):.1f} ms)")
    else:
        print(f"  [WARN] Frame Latency exceeds budget ({np.median(arr_total):.1f} ms)")

    if abs(final_rss - start_rss) < 15.0:
        print(f"  [PASS] Resident Memory Stable (Delta: +{final_rss - start_rss:.2f} MB)")
    else:
        print(f"  [WARN] Resident Memory Delta (+{final_rss - start_rss:.2f} MB)")

    print(f"\nStage 3 Offline Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()

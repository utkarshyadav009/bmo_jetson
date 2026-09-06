#!/usr/bin/env python3
"""Stage 1 & 2: Python C-ABI Roundtrip & CUDA Graph Benchmark Harness for libbmo.so."""

import os
import sys
import time
import numpy as np

def get_vmrss_mb():
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return 0.0

def main():
    print("=================================================================")
    print("  BMO Stage 1 & 2: Python C-ABI & CUDA Graph Verification")
    print("=================================================================")

    from bmo_engine import BMOEngine

    gguf_path = "/home/bmo/jetson_moshi_work/bmo_jetson/bmo_moshi_8cb_q4.gguf"
    if not os.path.exists(gguf_path):
        print(f"ERROR: Model file not found: {gguf_path}")
        sys.exit(1)

    rss_init = get_vmrss_mb()
    print(f"Initial VmRSS: {rss_init:.1f} MB")
    print(f"Loading engine with n_ctx=1024 from {gguf_path} ...")
    t0_load = time.perf_counter()
    engine = BMOEngine(gguf_path, n_ctx=1024)
    t_load = time.perf_counter() - t0_load
    print(f"Engine loaded in {t_load:.2f} s")

    print(f"  n_layers:    {engine.n_layers}")
    print(f"  n_embd:      {engine.n_embd}")
    print(f"  n_codebooks: {engine.n_codebooks} (Invariant: 17)")
    print(f"  dep_q:       {engine.dep_q} (Invariant: 8)")
    print(f"  text_vocab:  {engine.text_vocab}")
    print(f"  audio_vocab: {engine.audio_vocab}")
    print(f"  heads:       {engine.n_attn_heads} x {engine.head_dim}")

    assert engine.n_codebooks == 17, f"Invariant failed: n_codebooks={engine.n_codebooks} != 17"
    assert engine.dep_q == 8, f"Invariant failed: dep_q={engine.dep_q} != 8"
    assert engine.n_embd == 4096, f"n_embd={engine.n_embd} != 4096"
    assert engine.text_vocab == 32000, f"text_vocab={engine.text_vocab} != 32000"
    assert engine.audio_vocab == 2048, f"audio_vocab={engine.audio_vocab} != 2048"

    rss_loaded = get_vmrss_mb()
    print(f"Post-load VmRSS: {rss_loaded:.1f} MB (Delta: +{rss_loaded - rss_init:.1f} MB)")

    # Warmup in eager mode
    print("\nRunning warm-up frame in eager mode...")
    np.random.seed(42)
    warm_tokens = np.zeros(17, dtype=np.int32)
    warm_tokens[0] = 3  # PAD
    z, text_logits = engine.forward_temporal(warm_tokens)
    assert z.shape == (4096,)
    assert text_logits.shape == (32000,)
    assert not np.isnan(z).any() and not np.isnan(text_logits).any()

    prev_token = int(np.argmax(text_logits))
    for s in range(engine.dep_q):
        a_logits = engine.forward_depth(s, prev_token, z)
        assert a_logits.shape == (2048,)
        assert not np.isnan(a_logits).any()
        prev_token = int(np.argmax(a_logits))

    engine.reset()
    rss_warm = get_vmrss_mb()
    print(f"Post-warmup VmRSS: {rss_warm:.1f} MB")

    # CUDA Graph Capture
    print("\n--- Capturing CUDA Graphs for Temporal & Depth Passes ---")
    t0_cap = time.perf_counter()
    rc_cap = engine.capture_graphs()
    t_cap = (time.perf_counter() - t0_cap) * 1000.0
    has_graphs = engine.has_cuda_graphs()
    print(f"CUDA Graph capture finished in {t_cap:.1f} ms | rc={rc_cap}, has_cuda_graphs={has_graphs}")
    assert has_graphs, "CUDA Graph capture failed!"

    engine.reset()

    # 100-iteration benchmark
    N_ITERS = 100
    print(f"\nRunning {N_ITERS} iterations with CUDA Graph acceleration...")
    frame_times_ms = []
    temporal_times_ms = []
    depth_times_ms = []
    rss_checkpoints = {}

    for i in range(N_ITERS):
        t0_frame = time.perf_counter()

        # 1 text token (in 0..31999) + 8 user audio (in 0..2047) + 8 agent audio (in 0..2047)
        tok_text = np.random.randint(0, engine.text_vocab)
        tok_audio = np.random.randint(0, engine.audio_vocab, size=16)
        tokens = np.concatenate([[tok_text], tok_audio]).astype(np.int32)

        # 1. Temporal step (CUDA Graph)
        t0_temp = time.perf_counter()
        z, text_logits = engine.forward_temporal(tokens)
        t_temp_ms = (time.perf_counter() - t0_temp) * 1000.0
        temporal_times_ms.append(t_temp_ms)

        # Invariant checks
        if np.isnan(z).any() or np.isnan(text_logits).any():
            raise RuntimeError(f"NaN detected in temporal step at iteration {i}")

        # 2. Depth cascade (8 steps with CUDA Graphs)
        t0_dep = time.perf_counter()
        prev_tok = int(np.argmax(text_logits))
        sampled_audio = []
        for s in range(engine.dep_q):
            a_logits = engine.forward_depth(s, prev_tok, z)
            if np.isnan(a_logits).any():
                raise RuntimeError(f"NaN detected in depth step {s} at iteration {i}")
            prev_tok = int(np.argmax(a_logits))
            sampled_audio.append(prev_tok)
        t_dep_ms = (time.perf_counter() - t0_dep) * 1000.0
        depth_times_ms.append(t_dep_ms)

        t_frame_ms = (time.perf_counter() - t0_frame) * 1000.0
        frame_times_ms.append(t_frame_ms)

        if (i + 1) in (1, 10, 25, 50, 75, 100):
            cur_rss = get_vmrss_mb()
            rss_checkpoints[i + 1] = cur_rss
            print(f"  [Iter {i+1:3d}/{N_ITERS}] Frame: {t_frame_ms:5.1f} ms "
                  f"(Temp: {t_temp_ms:5.1f} ms, Depth: {t_dep_ms:4.1f} ms) | VmRSS: {cur_rss:.1f} MB")

    print("\n--- Latency Performance (CUDA Graph Mode) ---")
    f_arr = np.array(frame_times_ms)
    t_arr = np.array(temporal_times_ms)
    d_arr = np.array(depth_times_ms)

    med_frame = float(np.median(f_arr))
    p90_frame = float(np.percentile(f_arr, 90))
    p99_frame = float(np.percentile(f_arr, 99))
    min_frame = float(np.min(f_arr))
    max_frame = float(np.max(f_arr))

    print(f"  Frame Latency:    Median = {med_frame:.1f} ms (Target <= 68.0 ms), P90 = {p90_frame:.1f} ms, P99 = {p99_frame:.1f} ms (Target < 78.0 ms)")
    print(f"                    Min = {min_frame:.1f} ms, Max = {max_frame:.1f} ms")
    print(f"  Temporal Latency: Median = {np.median(t_arr):.1f} ms, P90 = {np.percentile(t_arr, 90):.1f} ms, P99 = {np.percentile(t_arr, 99):.1f} ms")
    print(f"  Depth Latency:    Median = {np.median(d_arr):.1f} ms, P90 = {np.percentile(d_arr, 90):.1f} ms, P99 = {np.percentile(d_arr, 99):.1f} ms")

    print("\n--- Resident Memory (VmRSS) Stability Check ---")
    rss_start = rss_checkpoints[1]
    rss_end = rss_checkpoints[100]
    leak_mb = rss_end - rss_start
    print(f"  VmRSS @ Iter 1:   {rss_start:.1f} MB")
    print(f"  VmRSS @ Iter 10:  {rss_checkpoints[10]:.1f} MB")
    print(f"  VmRSS @ Iter 50:  {rss_checkpoints[50]:.1f} MB")
    print(f"  VmRSS @ Iter 100: {rss_end:.1f} MB")
    print(f"  Total RSS Delta:  {leak_mb:+.1f} MB")

    pass_median = med_frame <= 68.0
    pass_p99 = p99_frame < 78.0
    pass_leak = abs(leak_mb) <= 5.0

    print(f"\n>>> TARGET AUDIT:")
    print(f"  [ {'PASS' if pass_median else 'FAIL'} ] Median Latency: {med_frame:.1f} ms <= 68.0 ms")
    print(f"  [ {'PASS' if pass_p99 else 'FAIL'} ] P99 Latency:    {p99_frame:.1f} ms < 78.0 ms")
    print(f"  [ {'PASS' if pass_leak else 'FAIL'} ] Memory Stability: Delta {leak_mb:+.1f} MB <= 5.0 MB")

    if pass_median and pass_p99 and pass_leak:
        print("\n>>> ALL STAGE 2 ACCEPTANCE CRITERIA MET! <<<")
    else:
        print("\n>>> AUDIT SUMMARY: Completed with items to review.")

if __name__ == "__main__":
    main()

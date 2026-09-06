#!/usr/bin/env python3
"""Stage 4: Real-Time Full-Duplex Audio Streaming Benchmark on Jetson Orin Nano.

Features:
1. Low-latency ALSA / sounddevice full-duplex streaming:
   - 24000 Hz, Mono, 1920 samples per block (80 ms frame duration = 12.5 Hz).
2. Mimi audio codec (CUDA):
   - Streaming encode: PCM -> 8 codebooks
   - Streaming decode: 8 codebooks -> PCM
3. BMO Temporal & Depth Engine (CUDA Graphs via libbmo.so):
   - Interleaved 17-codebook temporal transformer
   - 8-step depth autoregressive cascade
4. Industrial Duplex Threading:
   - Dedicated audio callback thread
   - Pre-roll jitter buffer (3 frames = 240 ms)
   - Continuous audio streaming & buffer health monitoring over 60 seconds.
"""

import os
import sys
import time
import queue
import argparse
import psutil
import threading
import numpy as np

# Performance & memory safeguards
os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
# Pin CPU threads for UMA bus protection
torch.set_num_threads(2)

import sounddevice as sd
import sphn
from moshi.models import get_mimi
from bmo_engine import BMOEngine


def get_vram_mb() -> float:
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Real-Time Duplex Audio Streaming Benchmark")
    parser.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds (default: 60.0s)")
    parser.add_argument("--device", type=str, default="default", help="Audio device name or index (default: 'default')")
    parser.add_argument("--input-wav", type=str, default="/home/bmo/bmo_ref_clip.wav", help="Audio clip to loop for input testing")
    parser.add_argument("--use-mic", action="store_true", help="Capture from physical microphone instead of simulated stream")
    parser.add_argument("--warmup-frames", type=int, default=5, help="Number of warmup frames before timing (default: 5)")
    args = parser.parse_args()

    print("=" * 70)
    print("  BMO Stage 4: Real-Time Full-Duplex Audio Streaming Benchmark")
    print("=" * 70)
    print(f"[*] Target Duration:  {args.duration:.1f} s (~{int(args.duration / 0.080)} frames)")
    print(f"[*] Audio Device:     {args.device}")
    print(f"[*] Mode:             {'Physical Microphone' if args.use_mic else 'Simulated Real-Time Audio (' + args.input_wav + ')'}")
    print(f"[*] Sample Rate:      24,000 Hz Mono")
    print(f"[*] Frame Size:       1920 samples (80.0 ms @ 12.5 Hz)")
    print(f"[*] Initial VmRSS:    {get_vram_mb():.1f} MB")

    # 1. Models & Paths
    gguf_path = os.environ.get(
        "BMO_GGUF",
        os.path.join(os.path.dirname(__file__), "bmo_moshi_8cb_q4.gguf")
    )
    mimi_path = os.environ.get(
        "BMO_MIMI_WEIGHT",
        "/home/bmo/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors"
    )

    if not os.path.isfile(gguf_path):
        raise FileNotFoundError(f"GGUF model not found: {gguf_path}")
    if not os.path.isfile(mimi_path):
        raise FileNotFoundError(f"Mimi checkpoint not found: {mimi_path}")

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

    # 4. Prepare Input Audio Frames
    frame_size = 1920
    sim_frames = []
    if not args.use_mic:
        if os.path.isfile(args.input_wav):
            pcm, sr = sphn.read(args.input_wav)
            if pcm.shape[0] > 1:
                pcm = pcm.mean(axis=0, keepdims=True)
            if sr != 24000:
                pcm = sphn.resample(pcm, src_sample_rate=sr, dst_sample_rate=24000)
            pcm_flat = pcm[0]
            n_chunks = len(pcm_flat) // frame_size
            for k in range(n_chunks):
                sim_frames.append(pcm_flat[k * frame_size : (k + 1) * frame_size].astype(np.float32))
            print(f"    Loaded {len(sim_frames)} audio frames ({len(sim_frames)*0.08:.2f}s) to stream continuously.")
        else:
            t = np.linspace(0, 0.08, frame_size, endpoint=False, dtype=np.float32)
            tone = 0.2 * np.sin(2 * np.pi * 440 * t)
            sim_frames = [tone]

    # 5. Preallocate static GPU buffers and Warmup
    print("\n[+] Preallocating tensors and warming up pipeline...")
    t_chunk_gpu = torch.empty((1, 1, frame_size), dtype=torch.float32, device="cuda")
    agent_tensor_gpu = torch.empty((1, 8, 1), dtype=torch.long, device="cuda")

    dummy_frame = torch.zeros((1, 1, frame_size), device="cuda", dtype=torch.float32)
    with torch.no_grad():
        for _ in range(args.warmup_frames):
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
    print("    Warmup complete. Streaming state and KV cache ready.")

    # 6. Setup Audio I/O Queues and Duplex Stream
    input_queue = queue.Queue(maxsize=32)
    output_queue = queue.Queue(maxsize=32)

    underflow_count = 0
    overflow_count = 0
    callback_calls = 0
    stop_event = threading.Event()

    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal underflow_count, overflow_count, callback_calls
        callback_calls += 1
        if status.input_overflow:
            overflow_count += 1
        if status.output_underflow:
            underflow_count += 1

        if args.use_mic:
            chunk = indata[:, 0].copy()
            try:
                input_queue.put_nowait(chunk)
            except queue.Full:
                overflow_count += 1

        try:
            pcm_out = output_queue.get_nowait()
            outdata[:, 0] = pcm_out
        except queue.Empty:
            outdata.fill(0)
            underflow_count += 1

    # Pre-roll 4 frames (320 ms) of comfort silence into output queue
    silence_frame = np.zeros(frame_size, dtype=np.float32)
    for _ in range(6):
        output_queue.put(silence_frame.copy())

    # If simulated stream, start a dedicated feeder thread that paces at 80ms
    def feeder_thread():
        idx = 0
        t_feed = time.perf_counter()
        while not stop_event.is_set():
            t_feed += 0.080
            now = time.perf_counter()
            if t_feed > now:
                time.sleep(t_feed - now)
            chunk = sim_frames[idx % len(sim_frames)]
            idx += 1
            try:
                input_queue.put(chunk, timeout=0.100)
            except queue.Full:
                pass

    feeder = None
    if not args.use_mic:
        feeder = threading.Thread(target=feeder_thread, daemon=True)
        feeder.start()

    print("\n[+] Starting Real-Time Audio Duplex Stream...")
    stream = sd.Stream(
        samplerate=24000,
        blocksize=1920,
        device=args.device,
        channels=1,
        dtype='float32',
        callback=audio_callback,
    )

    latencies_enc = []
    latencies_temporal = []
    latencies_depth = []
    latencies_bmo = []
    latencies_dec = []
    latencies_total = []

    tokens_17 = np.zeros(17, dtype=np.int32)
    agent_prev_audio = np.zeros(8, dtype=np.int32)
    current_text_token = 3  # PAD

    start_rss = get_vram_mb()
    stream.start()

    print(f"[+] Audio stream active. Running {args.duration:.1f}s benchmark...")

    t_start = time.perf_counter()
    frame_count = 0

    try:
        with torch.no_grad():
            while (time.perf_counter() - t_start) < args.duration and not stop_event.is_set():
                try:
                    frame_chunk = input_queue.get(timeout=0.150)
                except queue.Empty:
                    continue

                t_frame_start = time.perf_counter()

                # Step A: Mimi Encode
                t_chunk_gpu.copy_(torch.from_numpy(frame_chunk))
                t_enc_0 = time.perf_counter()
                user_codes = mimi.encode(t_chunk_gpu)
                t_enc = (time.perf_counter() - t_enc_0) * 1000.0

                user_tokens = user_codes[0, :, 0].detach().cpu().numpy().astype(np.int32)

                # Step B: 17-Token Interleave
                tokens_17[0] = current_text_token
                tokens_17[1:9] = user_tokens
                tokens_17[9:17] = agent_prev_audio

                # Step C: Temporal Forward
                t_temp_0 = time.perf_counter()
                z, text_logits = engine.forward_temporal(tokens_17)
                t_temporal = (time.perf_counter() - t_temp_0) * 1000.0

                next_text_token = int(np.argmax(text_logits))

                # Step D: Depth Cascade (8 steps)
                t_depth_0 = time.perf_counter()
                agent_tokens = np.zeros(8, dtype=np.int32)
                depth_prev = next_text_token
                for cb_idx in range(8):
                    depth_logits = engine.forward_depth(cb_idx, depth_prev, z)
                    depth_prev = int(np.argmax(depth_logits))
                    agent_tokens[cb_idx] = depth_prev
                t_depth = (time.perf_counter() - t_depth_0) * 1000.0

                t_bmo = t_temporal + t_depth

                # Step E: Mimi Decode
                agent_tensor_gpu.copy_(torch.from_numpy(agent_tokens).view(1, 8, 1))
                t_dec_0 = time.perf_counter()
                decoded_frame = mimi.decode(agent_tensor_gpu)
                torch.cuda.synchronize()
                t_dec = (time.perf_counter() - t_dec_0) * 1000.0

                t_frame_total = (time.perf_counter() - t_frame_start) * 1000.0

                # Queue output for playback
                pcm_out = decoded_frame[0, 0].detach().cpu().numpy()
                try:
                    output_queue.put_nowait(pcm_out)
                except queue.Full:
                    pass

                agent_prev_audio = agent_tokens
                current_text_token = next_text_token
                frame_count += 1

                latencies_enc.append(t_enc)
                latencies_temporal.append(t_temporal)
                latencies_depth.append(t_depth)
                latencies_bmo.append(t_bmo)
                latencies_dec.append(t_dec)
                latencies_total.append(t_frame_total)

                # Reset dialogue turn context every 150 frames (~12 seconds of dialogue)
                if engine.pos >= 400:
                    engine.reset()

                if frame_count % 50 == 0:
                    elapsed = time.perf_counter() - t_start
                    print(
                        f"    [{elapsed:4.1f}s/{args.duration:.0f}s] Frame {frame_count:4d} | "
                        f"Enc: {t_enc:4.1f}ms | Temp: {t_temporal:4.1f}ms | "
                        f"Depth: {t_depth:4.1f}ms | Dec: {t_dec:4.1f}ms | "
                        f"Total: {t_frame_total:4.1f}ms | Underruns: {underflow_count}"
                    )

    finally:
        stop_event.set()
        stream.stop()
        stream.close()

    total_wallclock = time.perf_counter() - t_start
    final_rss = get_vram_mb()

    arr_enc = np.array(latencies_enc)
    arr_temp = np.array(latencies_temporal)
    arr_depth = np.array(latencies_depth)
    arr_bmo = np.array(latencies_bmo)
    arr_dec = np.array(latencies_dec)
    arr_total = np.array(latencies_total)

    budget_ms = 80.0
    rtf = np.median(arr_total) / budget_ms
    underrun_rate = (underflow_count / max(1, callback_calls)) * 100.0

    print("\n" + "=" * 70)
    print("  REAL-TIME STREAMING BENCHMARK REPORT (60-SECOND DUPLEX)")
    print("=" * 70)
    print(f"  Wallclock Runtime:       {total_wallclock:.2f} s")
    print(f"  Frames Streamed:         {frame_count}")
    print(f"  Audio Callbacks:         {callback_calls}")
    print(f"  Buffer Underruns:        {underflow_count} ({underrun_rate:.2f}%)")
    print(f"  Buffer Overflows:        {overflow_count}")
    print("-" * 70)
    print(f"  Frame Processing Latencies (Budget: {budget_ms:.1f} ms):")
    print(f"    - Median Frame Latency: {np.median(arr_total):.1f} ms")
    print(f"    - P95 Frame Latency:    {np.percentile(arr_total, 95):.1f} ms")
    print(f"    - P99 Frame Latency:    {np.percentile(arr_total, 99):.1f} ms")
    print(f"    - Min / Max Latency:    {np.min(arr_total):.1f} / {np.max(arr_total):.1f} ms")
    print(f"    - Real-Time Factor:     {rtf:.3f}x")
    print("-" * 70)
    print(f"  Stage Breakdown (Median / P95 / P99):")
    print(f"    - Mimi Encode:          {np.median(arr_enc):4.1f} / {np.percentile(arr_enc, 95):4.1f} / {np.percentile(arr_enc, 99):4.1f} ms")
    print(f"    - Temporal Transformer: {np.median(arr_temp):4.1f} / {np.percentile(arr_temp, 95):4.1f} / {np.percentile(arr_temp, 99):4.1f} ms")
    print(f"    - Depth Cascade (8x):   {np.median(arr_depth):4.1f} / {np.percentile(arr_depth, 95):4.1f} / {np.percentile(arr_depth, 99):4.1f} ms")
    print(f"    - libbmo Engine Total:  {np.median(arr_bmo):4.1f} / {np.percentile(arr_bmo, 95):4.1f} / {np.percentile(arr_bmo, 99):4.1f} ms")
    print(f"    - Mimi Decode:          {np.median(arr_dec):4.1f} / {np.percentile(arr_dec, 95):4.1f} / {np.percentile(arr_dec, 99):4.1f} ms")
    print("-" * 70)
    print(f"  Memory Footprint:")
    print(f"    - Starting VmRSS:       {start_rss:.1f} MB")
    print(f"    - Final VmRSS:          {final_rss:.1f} MB")
    print(f"    - Memory Drift:         +{final_rss - start_rss:.2f} MB")
    print("=" * 70)

    # Verification assertions
    assert frame_count > 0, "No frames were processed!"
    assert underrun_rate < 15.0, f"Underrun rate too high: {underrun_rate:.2f}%"

    print("\n[VERIFICATION RESULTS]")
    if underrun_rate < 15.0:
        print(f"  [PASS] Buffer Underrun Rate ({underrun_rate:.2f}%) within real-time streaming tolerance")
    else:
        print(f"  [WARN] Buffer Underrun Rate ({underrun_rate:.2f}%) exceeds tolerance")

    if abs(final_rss - start_rss) < 20.0:
        print(f"  [PASS] Memory Stable over 60s stream (Drift: +{final_rss - start_rss:.2f} MB)")
    else:
        print(f"  [WARN] Memory Drift (+{final_rss - start_rss:.2f} MB)")

    print("\nStage 4 Real-Time Audio Streaming execution completed successfully!")


if __name__ == "__main__":
    main()

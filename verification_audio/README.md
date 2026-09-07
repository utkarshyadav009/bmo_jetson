# BMO Jetson Speech Synthesis Verification Audio

This directory contains reference audio samples synthesized by BMO on the NVIDIA Jetson Orin Nano (24,000 Hz mono PCM) demonstrating resolution of the speech corruption and real-time streaming performance.

---

### Audio Samples

| Filename | Source / Mode | Duration | Whisper Transcription | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`realtime_stream_session.wav`** | Live Real-Time Benchmark (`test_realtime_stream.py --duration 6 --audio-device pulse`) | 5.52 s | `“ I, how is your day?”` | Full-duplex real-time streaming with hardware playback echo gating, sub-80 ms frame turnaround, and verified eager mode. |
| **`eager_clean_synthesis.wav`** | Offline Synthesis (`BMOEngine` eager mode) | 2.40 s | `“ Hey there, how is it going?”` | 30 autoregressive token steps with greedy text argmax + stochastic audio sampling ($T=0.8, K=250$) decoded through stateful CUDA-graphed Mimi SEANet. |
| **`graph_mode_baseline_gibberish.wav`** | Broken CUDA Graph Baseline (Pre-fix) | 2.40 s | `“”` (unparseable / gibberish) | Shows the previous issue where static CUDA graph capture fixed RoPE at position 0, destroying self-attention coherence across steps. |

---

### Inspection & Playback

To inspect or transcribe these files locally:

```bash
# Transcribe with Whisper
python3 -c '
import whisper
model = whisper.load_model("tiny.en")
for f in ["verification_audio/realtime_stream_session.wav", "verification_audio/eager_clean_synthesis.wav"]:
    print(f, "->", model.transcribe(f)["text"])
'
```

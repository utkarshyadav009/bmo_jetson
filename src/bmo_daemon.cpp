// bmo_daemon.cpp - Native C++ real-time audio daemon for BMO Companion Robot.
//
// Eliminates all Python GIL, ctypes marshal, and sounddevice queue overhead
// by running the entire duplex audio stream (ALSA/PortAudio + TRT Mimi + libbmo)
// in a single native C++ process on a unified CUDA stream.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <memory>
#include <thread>
#include <atomic>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <alsa/asoundlib.h>

#include "bmo_api.h"

// ---------------------------------------------------------------------------
// TensorRT Logger & Helper
// ---------------------------------------------------------------------------
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
} gTRTLogger;

class TRTRunner {
public:
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::string input_name;
    std::string output_name;

    bool load(const std::string & engine_path, const std::string & in_name, const std::string & out_name) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[!] Failed to open TRT engine: " << engine_path << std::endl;
            return false;
        }
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buf(size);
        file.read(buf.data(), size);

        runtime.reset(nvinfer1::createInferRuntime(gTRTLogger));
        if (!runtime) return false;
        engine.reset(runtime->deserializeCudaEngine(buf.data(), size));
        if (!engine) return false;
        context.reset(engine->createExecutionContext());
        if (!context) return false;

        input_name = in_name;
        output_name = out_name;
        return true;
    }

    void execute(void * d_in, void * d_out, cudaStream_t stream) {
        context->setTensorAddress(input_name.c_str(), d_in);
        context->setTensorAddress(output_name.c_str(), d_out);
        context->enqueueV3(stream);
    }
};

// ---------------------------------------------------------------------------
// WAV File Reader / Writer Helper
// ---------------------------------------------------------------------------
struct WAVHeader {
    char riff[4];
    uint32_t riff_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
};

static std::vector<float> load_wav_mono_24k(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return {};

    if (path.length() >= 4 && path.substr(path.length() - 4) == ".raw") {
        file.seekg(0, std::ios::end);
        size_t bytes = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<float> pcm(bytes / sizeof(float));
        file.read(reinterpret_cast<char *>(pcm.data()), bytes);
        return pcm;
    }

    WAVHeader hdr;
    file.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
    if (std::memcmp(hdr.riff, "RIFF", 4) != 0) {
        // Fallback: treat as raw float32
        file.seekg(0, std::ios::end);
        size_t bytes = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<float> pcm(bytes / sizeof(float));
        file.read(reinterpret_cast<char *>(pcm.data()), bytes);
        return pcm;
    }

    std::vector<int16_t> raw(hdr.data_size / sizeof(int16_t));
    file.read(reinterpret_cast<char *>(raw.data()), hdr.data_size);

    std::vector<float> pcm(raw.size() / std::max((int)hdr.num_channels, 1));
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm[i] = (float) raw[i * hdr.num_channels] / 32768.0f;
    }
    return pcm;
}

static void save_wav_mono_24k(const std::string & path, const std::vector<float> & pcm) {
    WAVHeader hdr;
    std::memcpy(hdr.riff, "RIFF", 4);
    hdr.riff_size = sizeof(WAVHeader) - 8 + pcm.size() * sizeof(int16_t);
    std::memcpy(hdr.wave, "WAVE", 4);
    std::memcpy(hdr.fmt, "fmt ", 4);
    hdr.fmt_size = 16;
    hdr.audio_format = 1; // PCM
    hdr.num_channels = 1;
    hdr.sample_rate = 24000;
    hdr.byte_rate = 24000 * sizeof(int16_t);
    hdr.block_align = sizeof(int16_t);
    hdr.bits_per_sample = 16;
    std::memcpy(hdr.data, "data", 4);
    hdr.data_size = pcm.size() * sizeof(int16_t);

    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    for (float s : pcm) {
        float clamped = std::max(-1.0f, std::min(1.0f, s));
        int16_t val = (int16_t) (clamped * 32767.0f);
        file.write(reinterpret_cast<const char *>(&val), sizeof(val));
    }
}

// ---------------------------------------------------------------------------
// ALSA PCM Device Management
// ---------------------------------------------------------------------------
class ALSADuplexStream {
public:
    snd_pcm_t * playback_handle = nullptr;
    snd_pcm_t * capture_handle  = nullptr;
    bool is_live = false;

    bool init(const std::string & device_name, bool live_capture) {
        is_live = live_capture;
        unsigned int rate = 24000;
        int err;

        // Open playback
        err = snd_pcm_open(&playback_handle, device_name.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
        if (err < 0) {
            std::cerr << "[!] ALSA: cannot open playback device (" << snd_strerror(err) << ")" << std::endl;
            return false;
        }
        snd_pcm_set_params(playback_handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED,
                           1, rate, 1, 250000); // 250ms buffer latency

        if (is_live) {
            err = snd_pcm_open(&capture_handle, device_name.c_str(), SND_PCM_STREAM_CAPTURE, 0);
            if (err < 0) {
                std::cerr << "[!] ALSA: cannot open capture device (" << snd_strerror(err) << ")" << std::endl;
                snd_pcm_close(playback_handle);
                playback_handle = nullptr;
                return false;
            }
            snd_pcm_set_params(capture_handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED,
                               1, rate, 1, 250000);
        }
        return true;
    }

    void write_pcm(const float * pcm_mono, int n_samples) {
        if (!playback_handle) return;
        std::vector<int16_t> s16(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            float s = std::max(-1.0f, std::min(1.0f, pcm_mono[i]));
            s16[i] = (int16_t) (s * 32767.0f);
        }
        snd_pcm_sframes_t frames = snd_pcm_writei(playback_handle, s16.data(), n_samples);
        if (frames < 0) {
            snd_pcm_recover(playback_handle, frames, 0);
        }
    }

    void read_pcm(float * pcm_mono, int n_samples) {
        if (!capture_handle) {
            std::memset(pcm_mono, 0, n_samples * sizeof(float));
            return;
        }
        std::vector<int16_t> s16(n_samples);
        snd_pcm_sframes_t frames = snd_pcm_readi(capture_handle, s16.data(), n_samples);
        if (frames < 0) {
            snd_pcm_recover(capture_handle, frames, 0);
            std::memset(pcm_mono, 0, n_samples * sizeof(float));
            return;
        }
        for (int i = 0; i < n_samples; ++i) {
            pcm_mono[i] = (float) s16[i] / 32768.0f;
        }
    }

    void close() {
        if (playback_handle) { snd_pcm_drain(playback_handle); snd_pcm_close(playback_handle); playback_handle = nullptr; }
        if (capture_handle)  { snd_pcm_drain(capture_handle);  snd_pcm_close(capture_handle);  capture_handle = nullptr; }
    }

    ~ALSADuplexStream() { close(); }
};

// ---------------------------------------------------------------------------
// Main Daemon Engine Loop
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::string model_path      = "bmo_moshi_8cb_q4.gguf";
    std::string enc_engine_path = "seanet_encoder.engine";
    std::string dec_engine_path = "seanet_decoder.engine";
    std::string rvq_bin_path    = "bmo_rvq_weights.bin";
    std::string input_wav_path  = "/home/bmo/bmo_ref_clip.wav";
    std::string alsa_device     = "default";
    float duration_sec          = 15.0f;
    bool use_mic                = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--enc-engine" && i + 1 < argc) enc_engine_path = argv[++i];
        else if (arg == "--dec-engine" && i + 1 < argc) dec_engine_path = argv[++i];
        else if (arg == "--rvq-weights" && i + 1 < argc) rvq_bin_path = argv[++i];
        else if (arg == "--wav" && i + 1 < argc) input_wav_path = argv[++i];
        else if (arg == "--duration" && i + 1 < argc) duration_sec = std::stof(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) alsa_device = argv[++i];
        else if (arg == "--mic" || arg == "--live") use_mic = true;
    }

    std::cout << "======================================================================\n";
    std::cout << "  🤖 BMO Native C++ Audio Daemon (Zero-Python Real-Time Hot Loop)\n";
    std::cout << "======================================================================\n";
    std::cout << "[*] Model:       " << model_path << "\n";
    std::cout << "[*] TRT Encoder: " << enc_engine_path << "\n";
    std::cout << "[*] TRT Decoder: " << dec_engine_path << "\n";
    std::cout << "[*] RVQ Weights: " << rvq_bin_path << "\n";
    std::cout << "[*] Mode:        " << (use_mic ? "Live Microphone" : ("WAV Feeder: " + input_wav_path)) << "\n";
    std::cout << "[*] ALSA Device: " << alsa_device << "\n";

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // 1. Load TensorRT Engines
    std::cout << "\n[+] Loading TensorRT engines on CUDA..." << std::endl;
    TRTRunner trt_enc, trt_dec;
    if (!trt_enc.load(enc_engine_path, "audio", "latent")) {
        std::cerr << "[!] Failed to load SEANet encoder engine" << std::endl;
        return 1;
    }
    if (!trt_dec.load(dec_engine_path, "latent", "audio")) {
        std::cerr << "[!] Failed to load SEANet decoder engine" << std::endl;
        return 1;
    }
    std::cout << "    Loaded SEANet Encoder & Decoder engines successfully!" << std::endl;

    // 2. Load RVQ Weights onto GPU
    std::cout << "[+] Loading precomputed Mimi RVQ weights into GPU VRAM..." << std::endl;
    std::ifstream rvq_file(rvq_bin_path, std::ios::binary);
    if (!rvq_file.is_open()) {
        std::cerr << "[!] Cannot open RVQ weights binary: " << rvq_bin_path << std::endl;
        return 1;
    }

    float *d_proj_tables, *d_w_in0, *d_e0, *d_norm0, *d_w_in_rest, *d_e_rest, *d_norm_rest;
    size_t sz_proj   = 8 * 2048 * 512 * sizeof(float);
    size_t sz_win0   = 256 * 512 * sizeof(float);
    size_t sz_e0     = 2048 * 256 * sizeof(float);
    size_t sz_norm0  = 2048 * sizeof(float);
    size_t sz_winr   = 256 * 512 * sizeof(float);
    size_t sz_er     = 7 * 2048 * 256 * sizeof(float);
    size_t sz_normr  = 7 * 2048 * sizeof(float);

    cudaMalloc(&d_proj_tables, sz_proj);
    cudaMalloc(&d_w_in0, sz_win0);
    cudaMalloc(&d_e0, sz_e0);
    cudaMalloc(&d_norm0, sz_norm0);
    cudaMalloc(&d_w_in_rest, sz_winr);
    cudaMalloc(&d_e_rest, sz_er);
    cudaMalloc(&d_norm_rest, sz_normr);

    std::vector<char> h_buf(std::max({sz_proj, sz_er, sz_e0}));
    rvq_file.read(h_buf.data(), sz_proj);  cudaMemcpy(d_proj_tables, h_buf.data(), sz_proj, cudaMemcpyHostToDevice);
    rvq_file.read(h_buf.data(), sz_win0);  cudaMemcpy(d_w_in0, h_buf.data(), sz_win0, cudaMemcpyHostToDevice);
    rvq_file.read(h_buf.data(), sz_e0);    cudaMemcpy(d_e0, h_buf.data(), sz_e0, cudaMemcpyHostToDevice);
    rvq_file.read(h_buf.data(), sz_norm0); cudaMemcpy(d_norm0, h_buf.data(), sz_norm0, cudaMemcpyHostToDevice);
    rvq_file.read(h_buf.data(), sz_winr);  cudaMemcpy(d_w_in_rest, h_buf.data(), sz_winr, cudaMemcpyHostToDevice);
    rvq_file.read(h_buf.data(), sz_er);    cudaMemcpy(d_e_rest, h_buf.data(), sz_er, cudaMemcpyHostToDevice);
    rvq_file.read(h_buf.data(), sz_normr); cudaMemcpy(d_norm_rest, h_buf.data(), sz_normr, cudaMemcpyHostToDevice);
    std::cout << "    Mimi RVQ weights mapped to GPU memory (49.1 MB)" << std::endl;

    // 3. Initialize BMO C-ABI Engine & Capture CUDA Graphs
    std::cout << "[+] Initializing libbmo C-ABI engine..." << std::endl;
    bmo_handle_t * engine = bmo_init(model_path.c_str(), 512);
    if (!engine) {
        std::cerr << "[!] Failed to initialize libbmo" << std::endl;
        return 1;
    }
    std::cout << "[+] Capturing CUDA graphs for temporal and depth cascades..." << std::endl;
    bmo_capture_graphs(engine);
    std::cout << "    CUDA Graphs captured successfully!" << std::endl;

    // Preallocated GPU I/O buffers (Zero-allocation hot loop)
    float *d_pcm_in, *d_enc_latent, *d_dec_latent, *d_pcm_out, *d_rvq_scratch;
    int32_t *d_rvq_codes, *d_dec_codes;
    cudaMalloc(&d_pcm_in, 1920 * sizeof(float));
    cudaMalloc(&d_enc_latent, 512 * 2 * sizeof(float));
    cudaMalloc(&d_dec_latent, 512 * sizeof(float));
    cudaMalloc(&d_pcm_out, 1920 * sizeof(float));
    cudaMalloc(&d_rvq_scratch, 512 * sizeof(float));
    cudaMalloc(&d_rvq_codes, 8 * sizeof(int32_t));
    cudaMalloc(&d_dec_codes, 8 * sizeof(int32_t));

    // Host pinned buffers for C-ABI exchange
    float *h_z = nullptr;
    float *h_text_logits = nullptr;
    float *h_audio_logits = nullptr;
    cudaMallocHost(&h_z, 4096 * sizeof(float));
    cudaMallocHost(&h_text_logits, 32000 * sizeof(float));
    cudaMallocHost(&h_audio_logits, 2048 * sizeof(float));

    // 4. Initialize ALSA Audio Stream
    std::cout << "[+] Initializing ALSA audio subsystem (" << alsa_device << ")..." << std::endl;
    ALSADuplexStream alsa;
    if (!alsa.init(alsa_device, use_mic)) {
        std::cerr << "[!] Could not open ALSA audio device. Exiting." << std::endl;
        return 1;
    }

    // Load feeder audio if not using live mic
    std::vector<float> feeder_pcm;
    if (!use_mic) {
        feeder_pcm = load_wav_mono_24k(input_wav_path);
        std::cout << "    Loaded " << feeder_pcm.size() / 1920 << " frames ("
                  << (float) feeder_pcm.size() / 24000.0f << " s) from " << input_wav_path << std::endl;
    }

    // Delay line states (Kyutai Moshi invariant)
    int32_t prev_agent_text = 32000;
    int32_t prev_agent_audio[8] = {2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048};
    int32_t prev_user_audio[8]  = {2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048};
    int32_t prev_agent_cb0 = 1049;
    int32_t tokens_17[17] = {0};

    std::vector<double> latencies_enc, latencies_temp, latencies_depth, latencies_dec, latencies_total;
    std::vector<float> out_recording;

    int total_frames = (int) (duration_sec * 12.5f);
    if (!use_mic && !feeder_pcm.empty()) {
        total_frames = std::min(total_frames, (int) (feeder_pcm.size() / 1920));
    }

    std::cout << "\n[+] Starting Real-Time Audio Core (" << total_frames << " frames, " << duration_sec << " s)...\n";
    if (use_mic) {
        std::cout << "  >>> SPEAK INTO MICROPHONE NOW (Direct C++ ALSA Stream) <<<\n\n";
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    for (int f = 0; f < total_frames; ++f) {
        auto t_f0 = std::chrono::high_resolution_clock::now();

        // Step A: Audio Ingestion
        std::vector<float> h_chunk(1920, 0.0f);
        if (use_mic) {
            alsa.read_pcm(h_chunk.data(), 1920);
        } else {
            std::memcpy(h_chunk.data(), feeder_pcm.data() + f * 1920, 1920 * sizeof(float));
        }
        cudaMemcpyAsync(d_pcm_in, h_chunk.data(), 1920 * sizeof(float), cudaMemcpyHostToDevice, stream);

        // Step B: Mimi Encode (TensorRT FP16 SEANet + Fused RVQ)
        auto t_enc0 = std::chrono::high_resolution_clock::now();
        trt_enc.execute(d_pcm_in, d_enc_latent, stream);
        bmo_rvq_encode(d_enc_latent, d_w_in0, d_e0, d_norm0, d_w_in_rest, d_e_rest, d_norm_rest,
                       d_rvq_codes, d_rvq_scratch, stream);
        cudaStreamSynchronize(stream);
        auto t_enc1 = std::chrono::high_resolution_clock::now();
        double ms_enc = std::chrono::duration<double, std::milli>(t_enc1 - t_enc0).count();

        int32_t user_tokens[8];
        cudaMemcpy(user_tokens, d_rvq_codes, 8 * sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Step C: Interleave 17 Codebook Channels (Moshi invariant)
        tokens_17[0] = prev_agent_text;
        for (int i = 0; i < 8; ++i) tokens_17[1 + i] = prev_agent_audio[i];
        tokens_17[9] = user_tokens[0];
        for (int i = 1; i < 8; ++i) tokens_17[9 + i] = prev_user_audio[i];

        // Step D: libbmo Temporal Forward (CUDA Graph)
        auto t_temp0 = std::chrono::high_resolution_clock::now();
        bmo_forward_temporal(engine, tokens_17, 17, f, h_z, h_text_logits);
        auto t_temp1 = std::chrono::high_resolution_clock::now();
        double ms_temp = std::chrono::duration<double, std::milli>(t_temp1 - t_temp0).count();

        // Sample text token
        h_text_logits[32000] = -1e9f;
        int next_text_token = std::max_element(h_text_logits, h_text_logits + 32000) - h_text_logits;

        // Step E: libbmo Depth Cascade (8 steps, CUDA Graph)
        auto t_dep0 = std::chrono::high_resolution_clock::now();
        int32_t curr_agent_audio[8] = {0};
        int depth_prev = next_text_token;
        for (int cb = 0; cb < 8; ++cb) {
            bmo_forward_depth(engine, cb, depth_prev, h_z, h_audio_logits);
            h_audio_logits[2048] = -1e9f;
            depth_prev = std::max_element(h_audio_logits, h_audio_logits + 2048) - h_audio_logits;
            curr_agent_audio[cb] = depth_prev;
        }
        auto t_dep1 = std::chrono::high_resolution_clock::now();
        double ms_depth = std::chrono::duration<double, std::milli>(t_dep1 - t_dep0).count();

        // Step F: Mimi Decode (Fused RVQ + TensorRT FP16 SEANet)
        auto t_dec0 = std::chrono::high_resolution_clock::now();
        int32_t decode_tokens[8];
        for (int i = 0; i < 8; ++i) decode_tokens[i] = curr_agent_audio[i];
        decode_tokens[0] = prev_agent_cb0; // Un-delayed cb 0

        cudaMemcpyAsync(d_dec_codes, decode_tokens, 8 * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        bmo_rvq_decode(d_dec_codes, d_proj_tables, d_dec_latent, stream);
        trt_dec.execute(d_dec_latent, d_pcm_out, stream);
        cudaStreamSynchronize(stream);
        auto t_dec1 = std::chrono::high_resolution_clock::now();
        double ms_dec = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();

        // Step G: Playback & State Update
        std::vector<float> pcm_out_h(1920);
        cudaMemcpy(pcm_out_h.data(), d_pcm_out, 1920 * sizeof(float), cudaMemcpyDeviceToHost);
        alsa.write_pcm(pcm_out_h.data(), 1920);

        out_recording.insert(out_recording.end(), pcm_out_h.begin(), pcm_out_h.end());

        prev_agent_cb0   = curr_agent_audio[0];
        for (int i = 0; i < 8; ++i) {
            prev_agent_audio[i] = curr_agent_audio[i];
            prev_user_audio[i]  = user_tokens[i];
        }
        prev_agent_text  = next_text_token;

        auto t_f1 = std::chrono::high_resolution_clock::now();
        double ms_total = std::chrono::duration<double, std::milli>(t_f1 - t_f0).count();

        // If simulated feeder mode, pace audio playback to real-time 80ms
        if (!use_mic && ms_total < 80.0) {
            double sleep_ms = 80.0 - ms_total;
            std::this_thread::sleep_for(std::chrono::microseconds((int)(sleep_ms * 1000.0)));
        }

        latencies_enc.push_back(ms_enc);
        latencies_temp.push_back(ms_temp);
        latencies_depth.push_back(ms_depth);
        latencies_dec.push_back(ms_dec);
        latencies_total.push_back(ms_total);
        frame_count++;

        if (frame_count % 25 == 0) {
            std::cout << "  Frame " << frame_count << "/" << total_frames
                      << " | Enc: " << ms_enc << "ms"
                      << " | Temp: " << ms_temp << "ms"
                      << " | Depth: " << ms_depth << "ms"
                      << " | Dec: " << ms_dec << "ms"
                      << " | TOTAL: " << ms_total << " ms (Budget: 80.0ms)\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    // Compute percentiles
    std::sort(latencies_total.begin(), latencies_total.end());
    double p50 = latencies_total[latencies_total.size() * 50 / 100];
    double p95 = latencies_total[latencies_total.size() * 95 / 100];
    double p99 = latencies_total[latencies_total.size() * 99 / 100];
    double mean = std::accumulate(latencies_total.begin(), latencies_total.end(), 0.0) / latencies_total.size();

    std::cout << "\n======================================================================\n";
    std::cout << "  BMO NATIVE C++ AUDIO DAEMON BENCHMARK REPORT\n";
    std::cout << "======================================================================\n";
    std::cout << "  Processed Frames:        " << frame_count << "\n";
    std::cout << "  Audio Duration:          " << frame_count * 0.08f << " s\n";
    std::cout << "  Wallclock Runtime:       " << total_sec << " s\n";
    std::cout << "  Real-Time Factor (RTF):  " << total_sec / (frame_count * 0.08f) << "x (RTF < 1.0 is faster than real-time)\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "  Frame Latencies (Budget: 80.0 ms):\n";
    std::cout << "    - Mean Latency:        " << mean << " ms\n";
    std::cout << "    - Median (P50):        " << p50 << " ms\n";
    std::cout << "    - P95 Latency:         " << p95 << " ms\n";
    std::cout << "    - P99 Latency:         " << p99 << " ms\n";
    std::cout << "    - Headroom Margin:     +" << (80.0 - p50) << " ms\n";
    std::cout << "======================================================================\n";

    save_wav_mono_24k("output_bmo_daemon.wav", out_recording);
    std::cout << "[+] Saved generated audio to output_bmo_daemon.wav\n";

    // Cleanup
    cudaFree(d_proj_tables); cudaFree(d_w_in0); cudaFree(d_e0); cudaFree(d_norm0);
    cudaFree(d_w_in_rest); cudaFree(d_e_rest); cudaFree(d_norm_rest);
    cudaFree(d_pcm_in); cudaFree(d_enc_latent); cudaFree(d_dec_latent); cudaFree(d_pcm_out);
    cudaFree(d_rvq_scratch); cudaFree(d_rvq_codes); cudaFree(d_dec_codes);
    cudaFreeHost(h_z); cudaFreeHost(h_text_logits); cudaFreeHost(h_audio_logits);
    bmo_free(engine);
    cudaStreamDestroy(stream);
    return 0;
}

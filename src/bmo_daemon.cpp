// bmo_daemon.cpp - Native C++ real-time audio daemon for BMO Companion Robot.
//
// Eliminates all Python GIL, ctypes marshal, and sounddevice queue overhead
// by running the entire duplex audio stream (ALSA + TRT Mimi + libbmo)
// in a single native C++ process on a unified CUDA stream.
//
// Features:
// 1. Decoupled multithreaded audio I/O (Capture Thread + Playback Thread + Inference Core)
// 2. Jitter buffer pre-roll to eliminate all ALSA underruns (xruns)
// 3. Sub-millisecond top-k & temperature stochastic sampling (Text + 8-step Audio Depth)
// 4. Live streaming SentencePiece text display
// 5. Automatic Bluetooth (Sony WH-1000XM4) & USB (ReSpeaker) PulseAudio endpoint routing.

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
#include <mutex>
#include <condition_variable>
#include <queue>
#include <random>
#include <csignal>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <alsa/asoundlib.h>

#include "bmo_api.h"

// ---------------------------------------------------------------------------
// Global Termination Signal Handler
// ---------------------------------------------------------------------------
static std::atomic<bool> g_running{true};

static void sigint_handler(int sig) {
    (void)sig;
    g_running.store(false, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Thread-Safe Bounded Audio Queue (Zero-Lag Drop-Oldest on Overflow)
// ---------------------------------------------------------------------------
template <typename T>
class SafeAudioQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_;

public:
    explicit SafeAudioQueue(size_t max_size = 32) : max_size_(max_size) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_) {
            queue_.pop(); // Drop oldest to prevent latency drift
        }
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    bool pop(T & item, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                          [this]() { return !queue_.empty() || !g_running.load(std::memory_order_relaxed); })) {
            return false;
        }
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
    }
};

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
// Fast C++ Top-K & Temperature Stochastic Token Sampler
// ---------------------------------------------------------------------------
static int sample_token_cpp(
    const float * logits,
    int n_vocab,
    float temp,
    int top_k,
    std::vector<std::pair<float, int>> & scratch,
    std::mt19937 & rng
) {
    if (top_k <= 0 || temp <= 1e-4f) {
        return (int)(std::max_element(logits, logits + n_vocab) - logits);
    }
    scratch.resize(n_vocab);
    for (int i = 0; i < n_vocab; ++i) scratch[i] = {logits[i], i};
    int k = std::min(top_k, n_vocab);
    std::partial_sort(scratch.begin(), scratch.begin() + k, scratch.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });

    float max_l = scratch[0].first / temp;
    std::vector<float> probs(k);
    float sum_p = 0.0f;
    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp(scratch[i].first / temp - max_l);
        sum_p += probs[i];
    }
    if (sum_p <= 0.0f || std::isnan(sum_p)) return scratch[0].second;

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return scratch[dist(rng)].second;
}

// ---------------------------------------------------------------------------
// SentencePiece Vocabulary Loader & Formatter
// ---------------------------------------------------------------------------
static std::vector<std::string> load_vocab(const std::string & path) {
    std::ifstream file(path);
    if (!file.is_open()) return {};
    std::vector<std::string> vocab;
    std::string line;
    while (std::getline(file, line)) {
        vocab.push_back(line);
    }
    return vocab;
}

static void print_token(const std::string & token) {
    if (token.empty() || token == "<unk>" || token == "<pad>" || token == "<s>" || token == "</s>") {
        return;
    }
    std::string piece = token;
    // Replace SentencePiece space prefix (\xe2\x96\x81) with standard space
    size_t pos = 0;
    while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
        piece.replace(pos, 3, " ");
        pos += 1;
    }
    std::cout << piece;
    std::cout.flush();
}

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

    char riff[4];
    uint32_t riff_size;
    char wave[4];
    file.read(riff, 4);
    file.read(reinterpret_cast<char *>(&riff_size), 4);
    file.read(wave, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        file.seekg(0, std::ios::end);
        size_t bytes = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<float> pcm(bytes / sizeof(float));
        file.read(reinterpret_cast<char *>(pcm.data()), bytes);
        return pcm;
    }

    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint16_t audio_format = 1;

    char chunk_id[4];
    uint32_t chunk_size = 0;

    while (file.read(chunk_id, 4) && file.read(reinterpret_cast<char *>(&chunk_size), 4)) {
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            file.read(reinterpret_cast<char *>(&audio_format), 2);
            file.read(reinterpret_cast<char *>(&num_channels), 2);
            uint32_t sample_rate, byte_rate;
            uint16_t block_align;
            file.read(reinterpret_cast<char *>(&sample_rate), 4);
            file.read(reinterpret_cast<char *>(&byte_rate), 4);
            file.read(reinterpret_cast<char *>(&block_align), 2);
            file.read(reinterpret_cast<char *>(&bits_per_sample), 2);
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            if (bits_per_sample == 16) {
                std::vector<int16_t> raw(chunk_size / sizeof(int16_t));
                file.read(reinterpret_cast<char *>(raw.data()), chunk_size);
                int ch = std::max((int)num_channels, 1);
                std::vector<float> pcm(raw.size() / ch);
                for (size_t i = 0; i < pcm.size(); ++i) {
                    pcm[i] = (float) raw[i * ch] / 32768.0f;
                }
                return pcm;
            } else if (bits_per_sample == 32) {
                std::vector<float> raw(chunk_size / sizeof(float));
                file.read(reinterpret_cast<char *>(raw.data()), chunk_size);
                int ch = std::max((int)num_channels, 1);
                std::vector<float> pcm(raw.size() / ch);
                for (size_t i = 0; i < pcm.size(); ++i) {
                    pcm[i] = raw[i * ch];
                }
                return pcm;
            } else {
                file.seekg(chunk_size, std::ios::cur);
            }
        } else {
            file.seekg(chunk_size, std::ios::cur);
        }
    }
    return {};
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
// Threaded ALSA Duplex Stream Manager
// ---------------------------------------------------------------------------
class ALSADuplexEngine {
public:
    snd_pcm_t * playback_handle = nullptr;
    snd_pcm_t * capture_handle  = nullptr;
    SafeAudioQueue<std::vector<float>> in_queue{16};
    SafeAudioQueue<std::vector<float>> out_queue{16};
    std::unique_ptr<std::thread> capture_thread;
    std::unique_ptr<std::thread> playback_thread;
    bool is_live = false;
    std::atomic<float> last_mic_rms{0.0f};

    bool init(const std::string & device_name, bool live_capture) {
        is_live = live_capture;
        unsigned int rate = 24000;
        int err;

        // Open playback PCM (250ms buffer latency for anti-xrun safety)
        err = snd_pcm_open(&playback_handle, device_name.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
        if (err < 0) {
            std::cerr << "[!] ALSA: cannot open playback device '" << device_name << "' (" << snd_strerror(err) << ")" << std::endl;
            return false;
        }
        err = snd_pcm_set_params(playback_handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED,
                                 1, rate, 1, 250000);
        if (err < 0) {
            std::cerr << "[!] ALSA: cannot set playback params (" << snd_strerror(err) << ")" << std::endl;
            snd_pcm_close(playback_handle);
            playback_handle = nullptr;
            return false;
        }

        // Open capture PCM if in live mic mode
        if (is_live) {
            err = snd_pcm_open(&capture_handle, device_name.c_str(), SND_PCM_STREAM_CAPTURE, 0);
            if (err < 0) {
                std::cerr << "[!] ALSA: cannot open capture device '" << device_name << "' (" << snd_strerror(err) << ")" << std::endl;
                snd_pcm_close(playback_handle);
                playback_handle = nullptr;
                return false;
            }
            err = snd_pcm_set_params(capture_handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED,
                                     1, rate, 1, 250000);
            if (err < 0) {
                std::cerr << "[!] ALSA: cannot set capture params (" << snd_strerror(err) << ")" << std::endl;
                snd_pcm_close(playback_handle);
                snd_pcm_close(capture_handle);
                playback_handle = nullptr;
                capture_handle = nullptr;
                return false;
            }
        }

        return true;
    }

    void start_threads() {
        // Playback Worker Thread
        playback_thread = std::make_unique<std::thread>([this]() {
            // Pre-roll comfort silence (4 frames = 320ms) into ALSA ring buffer to prevent startup starvation
            std::vector<int16_t> silence(1920, 0);
            for (int k = 0; k < 4; ++k) {
                snd_pcm_writei(playback_handle, silence.data(), 1920);
            }

            std::vector<float> pcm;
            std::vector<int16_t> s16(1920);

            while (g_running.load(std::memory_order_relaxed)) {
                if (out_queue.pop(pcm, 100)) {
                    size_t n = std::min((size_t)1920, pcm.size());
                    for (size_t i = 0; i < n; ++i) {
                        float clamped = std::max(-1.0f, std::min(1.0f, pcm[i]));
                        s16[i] = (int16_t)(clamped * 32767.0f);
                    }
                    snd_pcm_sframes_t frames = snd_pcm_writei(playback_handle, s16.data(), n);
                    if (frames < 0) {
                        snd_pcm_recover(playback_handle, frames, 0);
                    }
                } else if (g_running.load(std::memory_order_relaxed)) {
                    // Feed comfort silence during pauses to prevent underflow
                    snd_pcm_writei(playback_handle, silence.data(), 1920);
                }
            }
        });

        // Capture Worker Thread
        if (is_live && capture_handle) {
            capture_thread = std::make_unique<std::thread>([this]() {
                std::vector<int16_t> s16(1920);
                std::vector<float> pcm(1920);

                while (g_running.load(std::memory_order_relaxed)) {
                    snd_pcm_sframes_t frames = snd_pcm_readi(capture_handle, s16.data(), 1920);
                    if (frames < 0) {
                        snd_pcm_recover(capture_handle, frames, 0);
                        continue;
                    }
                    if (frames > 0) {
                        float sum_sq = 0.0f;
                        for (int i = 0; i < frames; ++i) {
                            float val = (float)s16[i] / 32768.0f;
                            pcm[i] = val;
                            sum_sq += val * val;
                        }
                        last_mic_rms.store(std::sqrt(sum_sq / (float)frames), std::memory_order_relaxed);
                        in_queue.push(pcm);
                    }
                }
            });
        }
    }

    void stop() {
        g_running.store(false, std::memory_order_relaxed);
        if (capture_thread && capture_thread->joinable()) {
            capture_thread->join();
        }
        if (playback_thread && playback_thread->joinable()) {
            playback_thread->join();
        }
        if (playback_handle) {
            snd_pcm_drop(playback_handle);
            snd_pcm_close(playback_handle);
            playback_handle = nullptr;
        }
        if (capture_handle) {
            snd_pcm_drop(capture_handle);
            snd_pcm_close(capture_handle);
            capture_handle = nullptr;
        }
    }

    ~ALSADuplexEngine() {
        stop();
    }
};

// ---------------------------------------------------------------------------
// Main Daemon Engine Loop
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::signal(SIGINT, sigint_handler);

    std::string model_path      = "bmo_moshi_8cb_q4.gguf";
    std::string enc_engine_path = "seanet_encoder.engine";
    std::string dec_engine_path = "seanet_decoder.engine";
    std::string rvq_bin_path    = "bmo_rvq_weights.bin";
    std::string vocab_path      = "bmo_vocab.txt";
    std::string input_wav_path  = "/home/bmo/bmo_ref_clip.wav";
    std::string alsa_device     = "pulse";
    float duration_sec          = 0.0f; // 0 = run until Ctrl+C
    bool use_mic                = true; // default to live mic
    bool use_sampling           = true;
    float temp_text             = 0.7f;
    float temp_audio            = 0.8f;
    int top_k_text              = 25;
    int top_k_audio             = 250;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--enc-engine" && i + 1 < argc) enc_engine_path = argv[++i];
        else if (arg == "--dec-engine" && i + 1 < argc) dec_engine_path = argv[++i];
        else if (arg == "--rvq-weights" && i + 1 < argc) rvq_bin_path = argv[++i];
        else if (arg == "--vocab" && i + 1 < argc) vocab_path = argv[++i];
        else if (arg == "--wav" && i + 1 < argc) { input_wav_path = argv[++i]; use_mic = false; }
        else if (arg == "--duration" && i + 1 < argc) duration_sec = std::stof(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) alsa_device = argv[++i];
        else if (arg == "--mic" || arg == "--live") use_mic = true;
        else if (arg == "--greedy") use_sampling = false;
        else if (arg == "--temp-text" && i + 1 < argc) temp_text = std::stof(argv[++i]);
        else if (arg == "--temp-audio" && i + 1 < argc) temp_audio = std::stof(argv[++i]);
        else if (arg == "--top-k-text" && i + 1 < argc) top_k_text = std::stoi(argv[++i]);
        else if (arg == "--top-k-audio" && i + 1 < argc) top_k_audio = std::stoi(argv[++i]);
    }

    // Auto-configure audio routing to ensure Bluetooth headset or USB speaker/mic is active
    std::cout << "[*] Synchronizing PulseAudio endpoints via bmo_audio_routing..." << std::endl;
    int ret_route = std::system("python3 /home/bmo/bmo_audio_routing.py");
    (void)ret_route;

    std::cout << "======================================================================\n";
    std::cout << "  🤖 BMO Native C++ Full-Duplex Conversational Core (bmo_daemon)\n";
    std::cout << "======================================================================\n";
    std::cout << "[*] Model:         " << model_path << "\n";
    std::cout << "[*] TRT Encoder:   " << enc_engine_path << "\n";
    std::cout << "[*] TRT Decoder:   " << dec_engine_path << "\n";
    std::cout << "[*] RVQ Weights:   " << rvq_bin_path << "\n";
    std::cout << "[*] Vocabulary:    " << vocab_path << "\n";
    std::cout << "[*] Mode:          " << (use_mic ? "Live Full-Duplex Microphone" : ("WAV Feeder: " + input_wav_path)) << "\n";
    std::cout << "[*] Audio Device:  " << alsa_device << " (24kHz Mono, 80ms Frames)\n";
    std::cout << "[*] Sampling:      " << (use_sampling ? "Top-K Stochastic (Text: T=0.7 K=25, Audio: T=0.8 K=250)" : "Greedy Argmax") << "\n";
    std::cout << "[*] Target Run:    " << (duration_sec > 0.0f ? std::to_string((int)duration_sec) + " seconds" : "Continuous until Ctrl+C") << "\n";
    std::cout << "======================================================================\n";

    // Load Vocabulary
    auto vocab = load_vocab(vocab_path);
    if (!vocab.empty()) {
        std::cout << "[+] Loaded SentencePiece vocabulary: " << vocab.size() << " tokens." << std::endl;
    } else {
        std::cout << "[!] Warning: Vocab file not found at " << vocab_path << ", transcript will be numeric." << std::endl;
    }

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // 1. Load TensorRT Engines
    std::cout << "\n[1/4] Loading TensorRT FP16 Mimi Codec Engines on CUDA..." << std::endl;
    TRTRunner trt_enc, trt_dec;
    if (!trt_enc.load(enc_engine_path, "audio", "latent")) {
        std::cerr << "[!] Failed to load SEANet encoder engine" << std::endl;
        return 1;
    }
    if (!trt_dec.load(dec_engine_path, "latent", "audio")) {
        std::cerr << "[!] Failed to load SEANet decoder engine" << std::endl;
        return 1;
    }
    std::cout << "      Loaded SEANet Encoder & Decoder engines successfully!" << std::endl;

    // 2. Load RVQ Weights onto GPU
    std::cout << "[2/4] Loading precomputed Mimi RVQ weights into GPU VRAM..." << std::endl;
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
    std::cout << "      Mimi RVQ weights mapped to GPU memory (49.1 MB)" << std::endl;

    // 3. Initialize BMO C-ABI Engine & Capture CUDA Graphs
    std::cout << "[3/4] Initializing libbmo C-ABI engine (n_ctx=512)..." << std::endl;
    bmo_handle_t * engine = bmo_init(model_path.c_str(), 512);
    if (!engine) {
        std::cerr << "[!] Failed to initialize libbmo" << std::endl;
        return 1;
    }
    std::cout << "      Capturing CUDA graphs for temporal and depth cascades..." << std::endl;
    bmo_capture_graphs(engine);
    std::cout << "      CUDA Graphs captured successfully!" << std::endl;

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

    // 4. Initialize ALSA Audio Subsystem
    std::cout << "[4/4] Starting ALSA Audio Duplex Engine (" << alsa_device << ")..." << std::endl;
    ALSADuplexEngine alsa;
    if (!alsa.init(alsa_device, use_mic)) {
        std::cerr << "[!] Could not open ALSA audio device '" << alsa_device << "'. Exiting." << std::endl;
        return 1;
    }
    alsa.start_threads();
    std::cout << "      Audio capture and playback threads started successfully!" << std::endl;

    // Load feeder audio if not using live mic
    std::vector<float> feeder_pcm;
    if (!use_mic) {
        feeder_pcm = load_wav_mono_24k(input_wav_path);
        std::cout << "      Loaded " << feeder_pcm.size() / 1920 << " frames ("
                  << (float) feeder_pcm.size() / 24000.0f << " s) from " << input_wav_path << std::endl;
    }

    // Delay line states (Kyutai Moshi invariant)
    int32_t prev_agent_text = 32000;
    int32_t prev_agent_audio[8] = {2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048};
    int32_t prev_user_audio[8]  = {2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048};
    int32_t prev_agent_cb0 = 1049;
    int32_t tokens_17[17] = {0};

    std::mt19937 rng(1337);
    std::vector<std::pair<float, int>> scratch_text, scratch_audio;

    std::vector<double> latencies_enc, latencies_temp, latencies_depth, latencies_dec, latencies_total;
    std::vector<float> out_recording;

    int total_frames = (duration_sec > 0.0f) ? (int)(duration_sec * 12.5f) : 1000000;
    if (!use_mic && !feeder_pcm.empty()) {
        total_frames = std::min(total_frames, (int)(feeder_pcm.size() / 1920));
    }

    std::cout << "\n======================================================================\n";
    std::cout << "  🎤 LIVE FULL-DUPLEX LISTENING & SPEAKING ACTIVE\n";
    std::cout << "  Speak into your microphone naturally. Pausing allows BMO to reply.\n";
    std::cout << "  Press Ctrl-C anytime to exit.\n";
    std::cout << "======================================================================\n\n";

    auto t_start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    bool user_is_speaking = false;

    while (g_running.load(std::memory_order_relaxed) && frame_count < total_frames) {
        // Step A: Audio Ingestion from Queue
        std::vector<float> h_chunk(1920, 0.0f);
        if (use_mic) {
            if (!alsa.in_queue.pop(h_chunk, 120)) {
                // If waiting for mic chunk, continue
                continue;
            }
        } else {
            std::memcpy(h_chunk.data(), feeder_pcm.data() + (frame_count % (feeder_pcm.size() / 1920)) * 1920, 1920 * sizeof(float));
        }

        auto t_f0 = std::chrono::high_resolution_clock::now();

        // Live Voice Activity Indication
        float mic_rms = 0.0f;
        for (float s : h_chunk) mic_rms += s * s;
        mic_rms = std::sqrt(mic_rms / 1920.0f);

        if (mic_rms > 0.008f) {
            if (!user_is_speaking) {
                user_is_speaking = true;
                std::cout << "\n[🎙️  User Speaking...]\n";
                std::cout.flush();
            }
        } else {
            user_is_speaking = false;
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

        // Step C: Interleave 17 Codebook Channels (Kyutai Moshi invariant)
        tokens_17[0] = prev_agent_text;
        for (int i = 0; i < 8; ++i) tokens_17[1 + i] = prev_agent_audio[i];
        tokens_17[9] = user_tokens[0];
        for (int i = 1; i < 8; ++i) tokens_17[9 + i] = prev_user_audio[i];

        // Step D: libbmo Temporal Forward (CUDA Graph)
        auto t_temp0 = std::chrono::high_resolution_clock::now();
        bmo_forward_temporal(engine, tokens_17, 17, frame_count, h_z, h_text_logits);
        auto t_temp1 = std::chrono::high_resolution_clock::now();
        double ms_temp = std::chrono::duration<double, std::milli>(t_temp1 - t_temp0).count();

        // Sample next text token
        h_text_logits[32000] = -1e9f;
        int next_text_token = sample_token_cpp(h_text_logits, 32000, temp_text, top_k_text, scratch_text, rng);

        // Print streamed text
        if (next_text_token >= 0 && next_text_token < (int)vocab.size()) {
            print_token(vocab[next_text_token]);
        }

        // Step E: libbmo Depth Cascade (8 steps, CUDA Graph)
        auto t_dep0 = std::chrono::high_resolution_clock::now();
        int32_t curr_agent_audio[8] = {0};
        int depth_prev = next_text_token;
        for (int cb = 0; cb < 8; ++cb) {
            bmo_forward_depth(engine, cb, depth_prev, h_z, h_audio_logits);
            h_audio_logits[2048] = -1e9f;
            depth_prev = sample_token_cpp(h_audio_logits, 2048, temp_audio, top_k_audio, scratch_audio, rng);
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

        // Step G: Playback Queue & State Update
        std::vector<float> pcm_out_h(1920);
        cudaMemcpy(pcm_out_h.data(), d_pcm_out, 1920 * sizeof(float), cudaMemcpyDeviceToHost);
        alsa.out_queue.push(pcm_out_h);

        if (out_recording.size() < 24000 * 300) { // Limit recording buffer to 5 mins
            out_recording.insert(out_recording.end(), pcm_out_h.begin(), pcm_out_h.end());
        }

        prev_agent_cb0   = curr_agent_audio[0];
        for (int i = 0; i < 8; ++i) {
            prev_agent_audio[i] = curr_agent_audio[i];
            prev_user_audio[i]  = user_tokens[i];
        }
        prev_agent_text  = next_text_token;

        auto t_f1 = std::chrono::high_resolution_clock::now();
        double ms_total = std::chrono::duration<double, std::milli>(t_f1 - t_f0).count();

        latencies_enc.push_back(ms_enc);
        latencies_temp.push_back(ms_temp);
        latencies_depth.push_back(ms_depth);
        latencies_dec.push_back(ms_dec);
        latencies_total.push_back(ms_total);
        frame_count++;

        // For simulated feeder mode, maintain real-time 80ms pacing
        if (!use_mic && ms_total < 80.0) {
            double sleep_ms = 80.0 - ms_total;
            std::this_thread::sleep_for(std::chrono::microseconds((int)(sleep_ms * 1000.0)));
        }

        if (frame_count % 50 == 0) {
            std::cout << "\n  [Frame " << frame_count
                      << " | Compute: " << (int)ms_total << "ms"
                      << " (Enc: " << (int)ms_enc << "m Temp: " << (int)ms_temp
                      << "m Dep: " << (int)ms_depth << "m Dec: " << (int)ms_dec
                      << "m) | Headroom: +" << (80.0 - ms_total) << "ms]\n";
            std::cout.flush();
        }
    }

    std::cout << "\n\n[*] Shutting down audio engine gracefully..." << std::endl;
    alsa.stop();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    if (!latencies_total.empty()) {
        std::sort(latencies_total.begin(), latencies_total.end());
        double p50 = latencies_total[latencies_total.size() * 50 / 100];
        double p95 = latencies_total[latencies_total.size() * 95 / 100];
        double p99 = latencies_total[latencies_total.size() * 99 / 100];
        double mean = std::accumulate(latencies_total.begin(), latencies_total.end(), 0.0) / latencies_total.size();

        std::cout << "\n======================================================================\n";
        std::cout << "  BMO NATIVE C++ AUDIO DAEMON PERFORMANCE REPORT\n";
        std::cout << "======================================================================\n";
        std::cout << "  Processed Frames:        " << frame_count << "\n";
        std::cout << "  Audio Duration:          " << frame_count * 0.08f << " s\n";
        std::cout << "  Wallclock Runtime:       " << total_sec << " s\n";
        std::cout << "  Real-Time Factor (RTF):  " << total_sec / (frame_count * 0.08f) << "x (RTF < 1.0 is faster than real-time)\n";
        std::cout << "----------------------------------------------------------------------\n";
        std::cout << "  Frame GPU Latencies (Budget: 80.0 ms):\n";
        std::cout << "    - Mean Latency:        " << mean << " ms\n";
        std::cout << "    - Median (P50):        " << p50 << " ms\n";
        std::cout << "    - P95 Latency:         " << p95 << " ms\n";
        std::cout << "    - P99 Latency:         " << p99 << " ms\n";
        std::cout << "    - Headroom Margin:     +" << (80.0 - p50) << " ms\n";
        std::cout << "======================================================================\n";
    }

    if (!out_recording.empty()) {
        save_wav_mono_24k("output_bmo_daemon.wav", out_recording);
        std::cout << "[+] Saved session audio to output_bmo_daemon.wav\n";
    }

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

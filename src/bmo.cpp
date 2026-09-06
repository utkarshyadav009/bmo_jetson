// bmo.cpp - model loader and KV cache allocator
// Jetson prepare uploads per-row tier bases (row_c2/c4/c8) for fused matvec.

#include "bmo.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <unistd.h>
#include <vector>

#ifdef BMO_ENABLE_CUDA
#include <cuda_runtime.h>
#include "ggml-backend.h"
#include "ggml-cuda.h"
#endif

void bmo_print_mem_diag(const std::string & phase) {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    long mem_avail = 0;
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            std::sscanf(line.c_str(), "MemAvailable: %ld kB", &mem_avail);
            break;
        }
    }

    std::ifstream status("/proc/self/status");
    long vm_rss = 0, vm_lck = 0;
    while (std::getline(status, line)) {
        if (line.find("VmRSS:") == 0) {
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &vm_rss);
        } else if (line.find("VmLck:") == 0) {
            std::sscanf(line.c_str(), "VmLck: %ld kB", &vm_lck);
        }
    }
    std::fprintf(stderr,
                 "[mem_diag] %-20s | MemAvail: %4ld MB | VmRSS: %4ld MB | VmLck: %4ld MB\n",
                 phase.c_str(),
                 mem_avail / 1024,
                 vm_rss / 1024,
                 vm_lck / 1024);
}

namespace {

static ggml_tensor * get_tensor(ggml_context * data_ctx, const std::string & name) {
    return ggml_get_tensor(data_ctx, name.c_str());
}

static int32_t read_scalar_i32(ggml_context * data_ctx, const char * name, int32_t fallback = 0) {
    ggml_tensor * t = get_tensor(data_ctx, name);
    if (!t || ggml_nbytes(t) < (int) sizeof(int32_t)) {
        return fallback;
    }
    int32_t out = fallback;
    std::memcpy(&out, t->data, sizeof(int32_t));
    return out;
}

static float read_scalar_f32(ggml_context * data_ctx, const char * name, float fallback = 0.0f) {
    ggml_tensor * t = get_tensor(data_ctx, name);
    if (!t || ggml_nbytes(t) < (int) sizeof(float)) {
        return fallback;
    }
    float out = fallback;
    std::memcpy(&out, t->data, sizeof(float));
    return out;
}

static inline uint8_t unpack_u2_le(uint8_t byte, int lane) {
    return (byte >> (lane * 2)) & 0x3;
}

static void add_tensor_bytes_unique(ggml_tensor * t, std::unordered_set<const void *> & seen, size_t & total_bytes) {
    if (!t || !t->data) {
        return;
    }
    const void * key = t->data;
    if (seen.insert(key).second) {
        total_bytes += (size_t) ggml_nbytes(t);
    }
}

static void add_layer_bytes_unique(const bmo_layer & L, std::unordered_set<const void *> & seen, size_t & total_bytes) {
    add_tensor_bytes_unique(L.packed_weights, seen, total_bytes);
    add_tensor_bytes_unique(L.packed_mask, seen, total_bytes);
    add_tensor_bytes_unique(L.scale_low, seen, total_bytes);
    add_tensor_bytes_unique(L.scale_int4, seen, total_bytes);
    add_tensor_bytes_unique(L.scale_int8, seen, total_bytes);
    add_tensor_bytes_unique(L.fp16_indices, seen, total_bytes);
    add_tensor_bytes_unique(L.fp16_values, seen, total_bytes);
    add_tensor_bytes_unique(L.weight, seen, total_bytes);
    add_tensor_bytes_unique(L.bias, seen, total_bytes);
    add_tensor_bytes_unique(L.wq, seen, total_bytes);
    add_tensor_bytes_unique(L.wk, seen, total_bytes);
    add_tensor_bytes_unique(L.wv, seen, total_bytes);
    add_tensor_bytes_unique(L.wo, seen, total_bytes);
    add_tensor_bytes_unique(L.ffn_in, seen, total_bytes);
    add_tensor_bytes_unique(L.ffn_out, seen, total_bytes);
    add_tensor_bytes_unique(L.norm1_weight, seen, total_bytes);
    add_tensor_bytes_unique(L.norm2_weight, seen, total_bytes);
}

#ifdef BMO_ENABLE_CUDA
static float read_scalar_f32_gguf(ggml_context * ctx, const std::string & name, float fallback = 0.0f) {
    ggml_tensor * t = get_tensor(ctx, name);
    if (!t || ggml_nbytes(t) < (int) sizeof(float)) {
        return fallback;
    }
    float out = fallback;
    std::memcpy(&out, t->data, sizeof(float));
    return out;
}

static void free_device_packed_owned_buffers(device_packed_t & dp) {
#ifdef BMO_JETSON
    if (dp.row_c2) {
        cudaFree(dp.row_c2);
        dp.row_c2 = nullptr;
    }
    if (dp.row_c4) {
        cudaFree(dp.row_c4);
        dp.row_c4 = nullptr;
    }
    if (dp.row_c8) {
        cudaFree(dp.row_c8);
        dp.row_c8 = nullptr;
    }
    if (dp.row_c16) {
        cudaFree(dp.row_c16);
        dp.row_c16 = nullptr;
    }
    // canonical_base_host is freed in bmo_free_cuda_resources (needs access to ctx).
#endif
#ifndef BMO_JETSON
    if (dp.packed_weights) cudaFree(dp.packed_weights);
    if (dp.packed_mask) cudaFree(dp.packed_mask);
    if (dp.fp16_values) cudaFree(dp.fp16_values);
    if (dp.fp16_indices) cudaFree(dp.fp16_indices);
    if (dp.block_offset) cudaFree(dp.block_offset);
#endif
    dp = device_packed_t{};
}
#endif

} // namespace

void bmo_load_model(const char * fname, bmo_model & model, bmo_context & ctx) {
    ctx.streaming_big_pool = nullptr;
    ctx.streaming_big_pool_size = 0;
    ctx.streaming_big_pool_registered = false;
    ctx.streaming_scalar_pool = nullptr;
    ctx.streaming_scalar_pool_size = 0;

    // 1. Init without mmap
    ggml_context * data_ctx = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx =*/ &data_ctx,
    };

    gguf_context * gctx = gguf_init_from_file(fname, params);
    if (!gctx || !data_ctx) {
        throw std::runtime_error("Failed to parse GGUF");
    }
    model.gctx = gctx;
    model.wctx = data_ctx;

    // 2. Open file for reading
    int fd = open(fname, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("Failed to open GGUF");
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);

    // 3. Measure pools
    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    const size_t SCALAR_MAX = 4096;
    const size_t ALIGN = 64;
    auto round_up = [](size_t x, size_t a) -> size_t { return (x + a - 1) & ~(a - 1); };

    size_t scalar_total = 0;
    size_t big_total = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = ggml_get_tensor(data_ctx, gguf_get_tensor_name(gctx, i));
        if (!t) continue;
        size_t nb = (size_t) ggml_nbytes(t);
        if (nb <= SCALAR_MAX) scalar_total += round_up(nb, ALIGN);
        else                  big_total += round_up(nb, ALIGN);
    }

    // 4. Allocate pools
    void * scalar_base = nullptr;
    if (scalar_total > 0) {
        const size_t PAGE = 4096;
        size_t scalar_aligned = round_up(scalar_total, PAGE);
        if (posix_memalign(&scalar_base, PAGE, scalar_aligned) != 0) {
            close(fd);
            throw std::runtime_error("Failed to allocate scalar pool");
        }
#ifdef BMO_JETSON
        if (cudaHostRegister(scalar_base, scalar_aligned, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
            ctx.streaming_scalar_pool_registered = true;
        } else {
            cudaGetLastError(); // non-fatal
        }
#endif
    }

    void * big_base = nullptr;
    if (big_total > 0) {
        const size_t PAGE = 4096;
        size_t big_aligned = round_up(big_total, PAGE);
        if (posix_memalign(&big_base, PAGE, big_aligned) != 0) {
            close(fd);
            if (scalar_base) std::free(scalar_base);
            throw std::runtime_error("Failed to allocate big pool");
        }
#ifdef BMO_JETSON
        if (cudaHostRegister(big_base, big_aligned, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
            ctx.streaming_big_pool_registered = true;
        } else {
            close(fd);
            if (scalar_base) std::free(scalar_base);
            std::free(big_base);
            throw std::runtime_error("cudaHostRegister failed for big pool");
        }
#endif
        ctx.streaming_big_pool = big_base;
        ctx.streaming_big_pool_size = big_aligned;
    }
    ctx.streaming_scalar_pool = scalar_base;
    ctx.streaming_scalar_pool_size = scalar_total;

    // 5. Read data from disk
    const size_t data_offset = gguf_get_data_offset(gctx);
    size_t scalar_used = 0;
    size_t big_used = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = ggml_get_tensor(data_ctx, gguf_get_tensor_name(gctx, i));
        if (!t) continue;

        size_t nb = (size_t) ggml_nbytes(t);
        size_t aligned_nb = round_up(nb, ALIGN);
        off_t file_off = (off_t) (data_offset + gguf_get_tensor_offset(gctx, i));

        void * dst = nullptr;
        if (nb <= SCALAR_MAX) {
            dst = (uint8_t *) scalar_base + scalar_used;
            scalar_used += aligned_nb;
        } else {
            dst = (uint8_t *) big_base + big_used;
            big_used += aligned_nb;
        }

        size_t remaining = nb;
        uint8_t * out = (uint8_t *) dst;
        while (remaining > 0) {
            ssize_t r = pread(fd, out, remaining, file_off);
            if (r <= 0) {
                close(fd);
                throw std::runtime_error("pread failed while loading GGUF tensor payload");
            }
            out += (size_t) r;
            file_off += r;
            remaining -= (size_t) r;
        }
        t->data = dst; // Patch tensor
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);

    ctx.n_layers = read_scalar_i32(data_ctx, "n_layers", 0);
    if (ctx.n_layers <= 0) ctx.n_layers = read_scalar_i32(data_ctx, "n_layer", 32);

    ctx.n_heads = read_scalar_i32(data_ctx, "n_heads", 0);
    if (ctx.n_heads <= 0) ctx.n_heads = read_scalar_i32(data_ctx, "n_head", 0);

    ctx.n_embd = read_scalar_i32(data_ctx, "n_embd", 0);
    if (ctx.n_embd <= 0) ctx.n_embd = read_scalar_i32(data_ctx, "hidden_size", 0);

    ctx.n_ctx = read_scalar_i32(data_ctx, "n_ctx", 0);
    if (ctx.n_ctx <= 0) ctx.n_ctx = read_scalar_i32(data_ctx, "context_length", 2048);

    ctx.head_dim = read_scalar_i32(data_ctx, "head_dim", 0);
    if (ctx.head_dim <= 0) ctx.head_dim = read_scalar_i32(data_ctx, "n_embd_head_k", 0);

    // RoPE base frequency: try the canonical names, fall back to Moshi default.
    ctx.rope_theta = read_scalar_f32(data_ctx, "rope_theta", 0.0f);
    if (ctx.rope_theta <= 0.0f) ctx.rope_theta = read_scalar_f32(data_ctx, "rope_freq_base", 0.0f);
    if (ctx.rope_theta <= 0.0f) ctx.rope_theta = 10000.0f;

    // RMSNorm epsilon: standard Moshi rms_norm_f32 uses 1e-8f
    ctx.norm_eps = read_scalar_f32(data_ctx, "norm_eps", 0.0f);
    if (ctx.norm_eps <= 0.0f) ctx.norm_eps = read_scalar_f32(data_ctx, "rms_norm_eps", 0.0f);
    if (ctx.norm_eps <= 0.0f) ctx.norm_eps = 1e-8f;

    // Infer missing temporal dimensions from packed QKV metadata in layer 0.
    {
        const std::string qkv0 = "transformer_layers_0_self_attn_in_proj_weight";
        const int32_t qkv_rows = read_scalar_i32(data_ctx, (qkv0 + ".rows").c_str(), 0);
        const int32_t qkv_cols = read_scalar_i32(data_ctx, (qkv0 + ".cols").c_str(), 0);

        if (ctx.n_embd <= 0 && qkv_cols > 0) {
            ctx.n_embd = qkv_cols;
        }
        if (ctx.n_embd <= 0 && qkv_rows > 0 && (qkv_rows % 3) == 0) {
            ctx.n_embd = qkv_rows / 3;
        }
        if (ctx.head_dim <= 0 && ctx.n_heads > 0 && ctx.n_embd > 0) {
            ctx.head_dim = ctx.n_embd / ctx.n_heads;
        }
        if (ctx.n_heads <= 0 && ctx.head_dim > 0 && ctx.n_embd > 0 && (ctx.n_embd % ctx.head_dim) == 0) {
            ctx.n_heads = ctx.n_embd / ctx.head_dim;
        }
    }

    if (ctx.n_heads <= 0) ctx.n_heads = 32;
    if (ctx.n_embd <= 0) ctx.n_embd = 4096;
    if (ctx.head_dim <= 0 && ctx.n_heads > 0) {
        ctx.head_dim = ctx.n_embd / ctx.n_heads;
    }

    model.temporal_layers.resize((size_t) ctx.n_layers);
    for (int i = 0; i < ctx.n_layers; ++i) {
        auto & layer = model.temporal_layers[(size_t) i];
        std::string base = "transformer_layers_" + std::to_string(i);
        layer.name = base;
        layer.packed_weights = ggml_get_tensor(data_ctx, (base + "_self_attn_in_proj_weight.packed_weights").c_str());
        layer.packed_mask = ggml_get_tensor(data_ctx, (base + "_self_attn_in_proj_weight.packed_mask").c_str());
        layer.fp16_indices = ggml_get_tensor(data_ctx, (base + "_self_attn_in_proj_weight.fp16_indices").c_str());
        layer.fp16_values = ggml_get_tensor(data_ctx, (base + "_self_attn_in_proj_weight.fp16_values").c_str());
        // The exporter writes per-layer RMSNorm gamma as
        // `transformer_layers_{i}_norm1_weight` / `_norm2_weight` (underscore
        // separator, no dot). Older keys are kept as fallbacks for backward
        // compatibility with experimental GGUFs. If none of these are found,
        // norm{1,2}_weight stays NULL and we silently fall back to the LAZY
        // ggml_rms_norm path -- whose ->data field is uninitialized when the
        // eager QKV linear reads it on Jetson, producing all-zero Q/K/V and
        // killing attention for the entire stack. (This was the actual root
        // cause of the Phase 4.4 "gibberish text" bug.)
        layer.norm1_weight = ggml_get_tensor(data_ctx, (base + "_norm1_weight").c_str());
        layer.norm2_weight = ggml_get_tensor(data_ctx, (base + "_norm2_weight").c_str());
        if (!layer.norm1_weight) layer.norm1_weight = ggml_get_tensor(data_ctx, (base + "_attn_norm.weight").c_str());
        if (!layer.norm2_weight) layer.norm2_weight = ggml_get_tensor(data_ctx, (base + "_ffn_norm.weight").c_str());
        if (!layer.norm1_weight) layer.norm1_weight = ggml_get_tensor(data_ctx, (base + "_norm1.weight").c_str());
        if (!layer.norm2_weight) layer.norm2_weight = ggml_get_tensor(data_ctx, (base + "_norm2.weight").c_str());
        layer.weight = ggml_get_tensor(data_ctx, (base + "_self_attn_in_proj_weight").c_str());
        if (!layer.weight) layer.weight = ggml_get_tensor(data_ctx, (base + "_self_attn_in_proj").c_str());
        layer.wo = ggml_get_tensor(data_ctx, (base + "_self_attn_out_proj_weight").c_str());
        if (!layer.wo) layer.wo = ggml_get_tensor(data_ctx, (base + "_self_attn_out_proj").c_str());
        layer.ffn_in = ggml_get_tensor(data_ctx, (base + "_gating_linear_in_weight").c_str());
        if (!layer.ffn_in) layer.ffn_in = ggml_get_tensor(data_ctx, (base + "_gating_linear_in").c_str());
        layer.ffn_out = ggml_get_tensor(data_ctx, (base + "_gating_linear_out_weight").c_str());
        if (!layer.ffn_out) layer.ffn_out = ggml_get_tensor(data_ctx, (base + "_gating_linear_out").c_str());
        if (!layer.norm1_weight || !layer.norm2_weight) {
            std::cerr << "[bmo_load_model] WARNING: missing per-layer norm weights for " << base
                      << " (norm1=" << (layer.norm1_weight ? "found" : "MISSING")
                      << ", norm2=" << (layer.norm2_weight ? "found" : "MISSING")
                      << "). Attention/FFN inputs will read uninitialised memory.\n";
        }
    }

    constexpr int kDepthLayers = 6;
    model.depth_layers.resize(kDepthLayers);
    for (int i = 0; i < kDepthLayers; ++i) {
        auto & layer = model.depth_layers[(size_t) i];
        std::string base = "depformer_layers_" + std::to_string(i);
        layer.name = base;
        layer.norm1_weight = ggml_get_tensor(data_ctx, ("depformer.layers." + std::to_string(i) + ".norm1.weight").c_str());
        layer.norm2_weight = ggml_get_tensor(data_ctx, ("depformer.layers." + std::to_string(i) + ".norm2.weight").c_str());
        if (!layer.norm1_weight) layer.norm1_weight = ggml_get_tensor(data_ctx, (base + "_norm1_weight").c_str());
        if (!layer.norm2_weight) layer.norm2_weight = ggml_get_tensor(data_ctx, (base + "_norm2_weight").c_str());

        std::string dot_prefix = "depformer.layers." + std::to_string(i);
        model.depth_in_proj[i] = ggml_get_tensor(data_ctx, (dot_prefix + ".self_attn.in_proj_weight").c_str());
        if (!model.depth_in_proj[i]) {
            model.depth_in_proj[i] = ggml_get_tensor(data_ctx, (base + "_self_attn_in_proj_weight").c_str());
        }
        model.depth_out_proj[i] = ggml_get_tensor(data_ctx, (dot_prefix + ".self_attn.out_proj.weight").c_str());
        if (!model.depth_out_proj[i]) {
            model.depth_out_proj[i] = ggml_get_tensor(data_ctx, (base + "_self_attn_out_proj_weight").c_str());
        }

        for (int s = 0; s < DEP_Q; ++s) {
            std::string s_idx = std::to_string(s);
            std::string in_key = base + "_gating_" + s_idx + "_linear_in_weight";
            model.depth_gating_in[i][s] = ggml_get_tensor(data_ctx, in_key.c_str());
            if (!model.depth_gating_in[i][s]) {
                model.depth_gating_in[i][s] = ggml_get_tensor(data_ctx, (base + "_gating_" + s_idx + "_linear_in").c_str());
            }
            std::string out_key = base + "_gating_" + s_idx + "_linear_out_weight";
            model.depth_gating_out[i][s] = ggml_get_tensor(data_ctx, out_key.c_str());
            if (!model.depth_gating_out[i][s]) {
                model.depth_gating_out[i][s] = ggml_get_tensor(data_ctx, (base + "_gating_" + s_idx + "_linear_out").c_str());
            }
        }
    }

    constexpr int kDepformerCodebooks = DEP_Q;
    model.audio_embs.resize(kDepformerCodebooks, nullptr);
    model.depformer_in.resize(kDepformerCodebooks, nullptr);
    for (int i = 0; i < kDepformerCodebooks; ++i) {
        std::string idx = std::to_string(i);
        model.audio_embs[(size_t) i] = ggml_get_tensor(data_ctx, ("depformer_emb." + idx + ".weight").c_str());
        model.depformer_in[(size_t) i] = ggml_get_tensor(data_ctx, ("depformer_in." + idx + ".weight").c_str());
    }

    model.text_emb = ggml_get_tensor(data_ctx, "depformer_text_emb.weight");
    model.text_linear = ggml_get_tensor(data_ctx, "text_linear.weight");
    model.text_linear_bias = ggml_get_tensor(data_ctx, "text_linear.bias");
    model.out_norm_weight = ggml_get_tensor(data_ctx, "out_norm_weight");
    model.token_embedding = ggml_get_tensor(data_ctx, "token_embedding");
    model.output_head = ggml_get_tensor(data_ctx, "output_head");

    // Per-codebook depth output heads: linears.{k}.weight
    model.audio_heads.assign(model.depformer_in.size(), nullptr);
    for (size_t k = 0; k < model.depformer_in.size(); ++k) {
        const std::string key = "linears." + std::to_string(k) + ".weight";
        model.audio_heads[k] = ggml_get_tensor(data_ctx, key.c_str());
    }

#ifdef BMO_JETSON
    // Pre-cache CUDA device pointers for all static weights across temporal & depth tiers
    for (auto & L : model.temporal_layers) {
        for (auto * t : {L.weight, L.wo, L.ffn_in, L.ffn_out, L.norm1_weight, L.norm2_weight}) {
            if (t && !t->extra) {
                void * p_dev = nullptr;
                if (cudaHostGetDevicePointer(&p_dev, t->data, 0) == cudaSuccess) {
                    t->extra = p_dev;
                }
            }
        }
    }
    if (model.out_norm_weight && !model.out_norm_weight->extra) {
        void * p_dev = nullptr;
        if (cudaHostGetDevicePointer(&p_dev, model.out_norm_weight->data, 0) == cudaSuccess) {
            model.out_norm_weight->extra = p_dev;
        }
    }
    for (int i = 0; i < kDepthLayers; ++i) {
        for (auto * t : {model.depth_in_proj[i], model.depth_out_proj[i], model.depth_layers[(size_t) i].norm1_weight, model.depth_layers[(size_t) i].norm2_weight}) {
            if (t && !t->extra) {
                void * p_dev = nullptr;
                if (cudaHostGetDevicePointer(&p_dev, t->data, 0) == cudaSuccess) {
                    t->extra = p_dev;
                }
            }
        }
        for (int s = 0; s < DEP_Q; ++s) {
            for (auto * t : {model.depth_gating_in[i][s], model.depth_gating_out[i][s]}) {
                if (t && !t->extra) {
                    void * p_dev = nullptr;
                    if (cudaHostGetDevicePointer(&p_dev, t->data, 0) == cudaSuccess) {
                        t->extra = p_dev;
                    }
                }
            }
        }
    }
    for (auto * t : model.depformer_in) {
        if (t && !t->extra) {
            void * p_dev = nullptr;
            if (cudaHostGetDevicePointer(&p_dev, t->data, 0) == cudaSuccess) {
                t->extra = p_dev;
            }
        }
    }
    for (auto * t : model.audio_heads) {
        if (t && !t->extra) {
            void * p_dev = nullptr;
            if (cudaHostGetDevicePointer(&p_dev, t->data, 0) == cudaSuccess) {
                t->extra = p_dev;
            }
        }
    }
#endif

    // Temporal-tier (n_embd-wide) embedding tables. The export writes one
    // emb.{k}.weight per audio codebook plus a single text_emb.weight; we
    // size to a generous upper bound and trim trailing nulls so the array
    // length equals the true audio codebook count.
    constexpr int kMaxTemporalAudioCodebooks = 32;
    {
        std::vector<ggml_tensor *> embs(kMaxTemporalAudioCodebooks, nullptr);
        int last_present = -1;
        for (int i = 0; i < kMaxTemporalAudioCodebooks; ++i) {
            std::string idx = std::to_string(i);
            ggml_tensor * t = ggml_get_tensor(data_ctx, ("emb." + idx + ".weight").c_str());
            embs[(size_t) i] = t;
            if (t) last_present = i;
        }
        embs.resize((size_t) (last_present + 1));
        model.temporal_audio_embs = std::move(embs);
    }
    model.temporal_text_emb = ggml_get_tensor(data_ctx, "text_emb.weight");

    // NOTE: token_embedding can belong to a different module width than temporal
    // transformer blocks; do not overwrite temporal n_embd from that tensor.

    // ---- Vocabulary / codebook geometry derived from the loaded tensors ----
    {
        // Count temporal-tier audio codebooks; fall back to the depformer
        // tables only if the temporal embeddings are missing (legacy GGUFs).
        int32_t n_q = 0;
        int32_t input_vocab = 0;  // input embedding row count (often vocab + EPAD)
        for (auto * t : model.temporal_audio_embs) {
            if (!t) continue;
            ++n_q;
            if (input_vocab == 0 && t->ne[1] > 0) input_vocab = (int32_t) t->ne[1];
        }
        if (n_q == 0) {
            for (auto * t : model.audio_embs) {
                if (!t) continue;
                ++n_q;
                if (input_vocab == 0 && t->ne[1] > 0) input_vocab = (int32_t) t->ne[1];
            }
        }

        int32_t dq = 0;
        for (auto * t : model.depformer_in) {
            if (t) ++dq;
        }

        // The depth output vocab is what bmo_forward_depth actually emits, so
        // it's the canonical audio_vocab_size for the C-API. In Moshi-style
        // models the input embedding usually has one extra row for EPAD
        // (input_vocab = output_vocab + 1) -- we don't want callers sizing
        // their buffers for that extra slot since the head never produces it.
        int32_t output_vocab = 0;
        for (auto * t : model.audio_heads) {
            if (!t) continue;
            if (t->ne[1] > 0) { output_vocab = (int32_t) t->ne[1]; break; }
        }

        // Moshi exposes K = n_q + 1 channels (text + audio) as the temporal input.
        ctx.num_codebooks    = (n_q > 0) ? (n_q + 1) : 0;
        ctx.dep_q            = dq;
        ctx.audio_vocab_size = (output_vocab > 0) ? output_vocab : input_vocab;
        if (output_vocab > 0 && input_vocab > 0 && output_vocab != input_vocab) {
            std::cout << "[bmo_load_model] audio vocab: input=" << input_vocab
                      << " output=" << output_vocab
                      << " (using output dim as canonical audio_vocab_size; "
                         "extra input rows are EPAD/special tokens)\n";
        }

        int32_t t_vocab = 0;
        if (model.text_linear && model.text_linear->ne[1] > 0) {
            t_vocab = (int32_t) model.text_linear->ne[1];
        } else if (model.temporal_text_emb && model.temporal_text_emb->ne[1] > 0) {
            t_vocab = (int32_t) model.temporal_text_emb->ne[1];
        } else if (model.text_emb && model.text_emb->ne[1] > 0) {
            t_vocab = (int32_t) model.text_emb->ne[1];
        }
        ctx.text_vocab_size = t_vocab;
    }

    size_t total_bytes = 0;
    std::unordered_set<const void *> seen;
    for (const auto & L : model.temporal_layers) add_layer_bytes_unique(L, seen, total_bytes);
    for (const auto & L : model.depth_layers) add_layer_bytes_unique(L, seen, total_bytes);
    for (auto * t : model.audio_embs) add_tensor_bytes_unique(t, seen, total_bytes);
    for (auto * t : model.depformer_in) add_tensor_bytes_unique(t, seen, total_bytes);
    for (auto * t : model.temporal_audio_embs) add_tensor_bytes_unique(t, seen, total_bytes);
    add_tensor_bytes_unique(model.text_emb, seen, total_bytes);
    add_tensor_bytes_unique(model.temporal_text_emb, seen, total_bytes);
    add_tensor_bytes_unique(model.text_linear, seen, total_bytes);
    add_tensor_bytes_unique(model.text_linear_bias, seen, total_bytes);
    add_tensor_bytes_unique(model.out_norm_weight, seen, total_bytes);
    add_tensor_bytes_unique(model.token_embedding, seen, total_bytes);
    add_tensor_bytes_unique(model.output_head, seen, total_bytes);
    for (auto * t : model.audio_heads) add_tensor_bytes_unique(t, seen, total_bytes);
    ctx.weights_bytes = total_bytes;

    std::cout << "[bmo_load_model] Loaded model '" << fname << "'\n";
    bmo_prepare_device_packed_tensors(model, ctx);
    std::cout << "[bmo_load_model] n_layers=" << ctx.n_layers
              << " n_heads=" << ctx.n_heads
              << " n_embd=" << ctx.n_embd
              << " n_ctx=" << ctx.n_ctx
              << " rope_theta=" << ctx.rope_theta
              << " norm_eps=" << ctx.norm_eps
              << " num_codebooks=" << ctx.num_codebooks
              << " dep_q=" << ctx.dep_q
              << " text_vocab=" << ctx.text_vocab_size
              << " audio_vocab=" << ctx.audio_vocab_size
              << " temporal_emb_tables=" << model.temporal_audio_embs.size()
              << (model.temporal_text_emb ? " temporal_text_emb=present" : " temporal_text_emb=MISSING")
              << (model.out_norm_weight ? " out_norm=present" : " out_norm=MISSING")
              << (model.text_linear_bias ? " text_linear_bias=present" : " text_linear_bias=MISSING")
              << " audio_heads=" << std::count_if(
                       model.audio_heads.begin(), model.audio_heads.end(),
                       [](ggml_tensor * t) { return t != nullptr; })
              << "/" << model.audio_heads.size()
              << "\n";
    std::cout << "[bmo_load_model] Total weight bytes: " << (double) total_bytes / (1024.0 * 1024.0) << " MB\n";

    // Diagnostic: dump the exact ggml type and shape of the head tensors so we
    // can detect a transposed/quantized text_linear or a missing out_norm gamma
    // (both manifest as ~uniform clustered text logits after Phase 4.4).
    auto ttype = [](ggml_tensor * t) -> const char * {
        if (!t) return "NULL";
        return ggml_type_name(t->type);
    };
    auto tshape = [](ggml_tensor * t) -> std::string {
        if (!t) return "NULL";
        return std::to_string(t->ne[0]) + "x" + std::to_string(t->ne[1])
            + "x" + std::to_string(t->ne[2]) + "x" + std::to_string(t->ne[3]);
    };
    int n_norm1 = 0, n_norm2 = 0;
    for (auto & L : model.temporal_layers) {
        if (L.norm1_weight) ++n_norm1;
        if (L.norm2_weight) ++n_norm2;
    }
    std::cout << "[bmo_load_model] per-layer norm gammas: norm1=" << n_norm1
              << "/" << model.temporal_layers.size()
              << " norm2=" << n_norm2 << "/" << model.temporal_layers.size() << "\n";

    // Dump per-layer norm gamma stats. Anomalously large gammas (e.g. mean
    // >> 2 or values > 5x the layer median) flag a wrong-tensor or scale
    // bug, which would explode the residual stream's DC component layer
    // by layer -- exactly the "DC attractor" symptom we're investigating.
    auto dump_norm_stats = [&](const char * tag, ggml_tensor * t, int layer) {
        if (!t || !t->data) return;
        const int64_t n = ggml_nelements(t);
        double s = 0, smax = -1e30, smin = 1e30;
        if (t->type == GGML_TYPE_F32) {
            const float * w = (const float *) t->data;
            for (int64_t i = 0; i < n; ++i) { s += w[i]; if (w[i] > smax) smax = w[i]; if (w[i] < smin) smin = w[i]; }
        } else if (t->type == GGML_TYPE_F16) {
            const ggml_fp16_t * w = (const ggml_fp16_t *) t->data;
            for (int64_t i = 0; i < n; ++i) {
                const float v = ggml_fp16_to_fp32(w[i]);
                s += v; if (v > smax) smax = v; if (v < smin) smin = v;
            }
        } else {
            return;
        }
        std::cout << "[bmo_load_model] " << tag << "[" << layer << "] n=" << n
                  << " mean=" << (s / (double) n)
                  << " min=" << smin << " max=" << smax << "\n";
    };
    if (getenv("BMO_LOG_NORM_STATS")) {
        for (size_t li = 0; li < model.temporal_layers.size(); ++li) {
            dump_norm_stats("norm1", model.temporal_layers[li].norm1_weight, (int) li);
            dump_norm_stats("norm2", model.temporal_layers[li].norm2_weight, (int) li);
        }
    }

    std::cout << "[bmo_load_model] head tensors:"
              << " text_linear=" << ttype(model.text_linear)
              << "[" << tshape(model.text_linear) << "]"
              << " out_norm_weight=" << ttype(model.out_norm_weight)
              << "[" << tshape(model.out_norm_weight) << "]"
              << " temporal_text_emb=" << ttype(model.temporal_text_emb)
              << "[" << tshape(model.temporal_text_emb) << "]"
              << "\n";
    if (model.out_norm_weight && model.out_norm_weight->type == GGML_TYPE_F32 && model.out_norm_weight->data) {
        const float * w = (const float *) model.out_norm_weight->data;
        const int64_t n = ggml_nelements(model.out_norm_weight);
        double s = 0, smax = -1e30, smin = 1e30;
        for (int64_t i = 0; i < n; ++i) { s += w[i]; if (w[i] > smax) smax = w[i]; if (w[i] < smin) smin = w[i]; }
        std::cout << "[bmo_load_model] out_norm_weight stats: n=" << n
                  << " mean=" << (s / (double) n) << " min=" << smin << " max=" << smax << "\n";
    }
}

void bmo_prepare_device_packed_tensors(bmo_model & model, bmo_context & ctx) {
#ifndef BMO_ENABLE_CUDA
    std::cerr << "[bmo_prepare_device_packed_tensors] CUDA not enabled; skipping GPU allocation\n";
    return;
#endif

#ifdef BMO_ENABLE_CUDA
    bmo_print_mem_diag("Start Prepare");
    size_t max_unpack_elems = 0;

    // Rebuild packed registry if called repeatedly; do not free streaming pools here.
    for (auto & kv : ctx.packed_registry) {
        free_device_packed_owned_buffers(kv.second);
    }
    ctx.packed_registry.clear();

    if (!ctx.cuda_backend) {
        ggml_backend_t backend = ggml_backend_cuda_init(0);
        if (!backend) {
            std::cerr << "[bmo_prepare_device_packed_tensors] failed to initialize CUDA backend; skipping\n";
            return;
        }
        ctx.cuda_backend = backend;
    }

#ifdef BMO_JETSON
    bmo_print_mem_diag("After Stream Alloc");
#endif

    // Process all packed temporal matrices for each temporal layer.
    for (int i = 0; i < ctx.n_layers; ++i) {
        std::string prefix = "transformer_layers_" + std::to_string(i);
        std::vector<std::string> matrices = {
            prefix + "_self_attn_in_proj_weight",
            prefix + "_self_attn_out_proj_weight",
            prefix + "_gating_linear_in_weight",
            prefix + "_gating_linear_out_weight"
        };

        for (const std::string & base : matrices) {
            if (getenv("BMO_LOG_INIT")) {
                std::cout << "[bmo_prepare_device_packed_tensors] packing candidate base=" << base << "\n";
            }
            ggml_tensor * pw = ggml_get_tensor(model.wctx, (base + ".packed_weights").c_str());
            if (!pw) {
                if (getenv("BMO_LOG_INIT")) {
                    std::cout << "[bmo_prepare_device_packed_tensors] skip " << base
                              << ": no .packed_weights tensor\n";
                }
                continue; // Not packed for this matrix
            }

            ggml_tensor * pm = ggml_get_tensor(model.wctx, (base + ".packed_mask").c_str());
            ggml_tensor * fv = ggml_get_tensor(model.wctx, (base + ".fp16_values").c_str());

            // 1. Read exact dimensions from GGUF scalars
            int32_t rows = read_scalar_i32(model.wctx, (base + ".rows").c_str(), 0);
            if (rows <= 0) rows = read_scalar_i32(model.wctx, (base + ".out_features").c_str(), 0);
            int32_t cols = read_scalar_i32(model.wctx, (base + ".cols").c_str(), 0);

            if (rows <= 0 || cols <= 0) {
                std::cerr << "[bmo_prepare_device_packed_tensors] Invalid dims for " << base << "\n";
                continue;
            }

            int32_t block_size = read_scalar_i32(model.wctx, (base + ".block_size").c_str(), 0);
            int32_t n_blocks = read_scalar_i32(model.wctx, (base + ".n_blocks").c_str(), 0);
            if (block_size <= 0) block_size = 32;
            if (n_blocks <= 0) n_blocks = (rows * cols + block_size - 1) / block_size;

            if (!pm || !fv) {
                std::cerr << "[bmo_prepare_device_packed_tensors] Missing block-wise packed tensors for " << base << "; skipping\n";
                continue;
            }

            int64_t n_fp16 = 0;
            if (fv->type == GGML_TYPE_F16) {
                n_fp16 = ggml_nbytes(fv) / sizeof(ggml_fp16_t);
            } else if (fv->type == GGML_TYPE_F32) {
                n_fp16 = ggml_nbytes(fv) / sizeof(float);
            } else {
                std::cerr << "[bmo_prepare_device_packed_tensors] Unsupported fp16_values type for " << base << "; skipping\n";
                continue;
            }

            // 3. Allocate and Copy to CUDA
            device_packed_t dp;
            dp.rows = rows;
            dp.cols = cols;
            dp.block_size = block_size;
            dp.n_blocks = n_blocks;
            dp.n_fp16 = n_fp16;
            dp.is_blockwise = true;

            size_t pw_bytes = (size_t) ggml_nbytes(pw);
            size_t pm_bytes = (size_t) ggml_nbytes(pm);

            const int32_t packing_version =
                read_scalar_i32(model.wctx, (base + ".packing_version").c_str(), 3);
            dp.packing_version = packing_version;

            const int64_t elem_total = (int64_t) rows * (int64_t) cols;
            const size_t pm_expect_v5 =
                (elem_total <= 0) ? 0 : (size_t) ((elem_total + (int64_t) 3) / (int64_t) 4);
            const bool v5_layout = (packing_version >= 5);
            if (v5_layout) {
                if (pm_expect_v5 == 0 || pm_bytes != pm_expect_v5) {
                    std::cerr << "[bmo_prepare_device_packed_tensors] " << base
                              << ": v5 packed_mask size mismatch: got " << pm_bytes << " bytes, expected "
                              << pm_expect_v5 << " for rows=" << rows << " cols=" << cols << "\n";
                    continue;
                }
            }

#ifdef BMO_JETSON
            if (!v5_layout) {
                const int32_t n_blocks_from_mask = static_cast<int32_t>(pm_bytes * 4);
                n_blocks = n_blocks_from_mask;
                dp.n_blocks = n_blocks_from_mask;
            }
#else
            const uint8_t * pm_host = reinterpret_cast<const uint8_t *>(pm->data);
            std::vector<int32_t> block_offset;
            if (!v5_layout) {
                block_offset.assign((size_t) n_blocks, 0);
                int32_t c2 = 0;
                int32_t c4 = 0;
                int32_t c8 = 0;
                int32_t c16 = 0;
                for (int32_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
                    const uint8_t mbyte = pm_host[(size_t) block_idx / 4];
                    const uint8_t tier = unpack_u2_le(mbyte, block_idx % 4);
                    if (tier == 0) {
                        block_offset[(size_t) block_idx] = c16;
                        c16 += block_size;
                    } else if (tier == 1) {
                        block_offset[(size_t) block_idx] = c8;
                        c8 += block_size;
                    } else if (tier == 2) {
                        block_offset[(size_t) block_idx] = c4;
                        c4 += block_size;
                    } else {
                        block_offset[(size_t) block_idx] = c2;
                        c2 += block_size;
                    }
                }
            }
#endif

            cudaError_t err = cudaSuccess;

#ifdef BMO_JETSON
            if (fv->type == GGML_TYPE_F32) {
                std::cerr << "[bmo_prepare_device_packed_tensors] Jetson fused path requires fp16_values to be F16 for " << base << "\n";
                continue;
            }
            dp.host_packed_weights = pw->data;
            dp.pw_size = pw_bytes;
            dp.host_packed_mask = pm->data;
            dp.pm_size = pm_bytes;
            dp.host_fp16_values = fv->data;
            dp.fv_size = (size_t) ggml_nbytes(fv);

            dp.n_2bit_bytes = read_scalar_i32(model.wctx, (base + ".n_2bit_bytes").c_str(), 0);
            dp.n_4bit_bytes = read_scalar_i32(model.wctx, (base + ".n_4bit_bytes").c_str(), 0);
            dp.n_8bit_bytes = read_scalar_i32(model.wctx, (base + ".n_8bit_bytes").c_str(), 0);
            dp.scale_low = read_scalar_f32_gguf(model.wctx, base + ".scale_low", 1.0f);
            dp.scale_int4 = read_scalar_f32_gguf(model.wctx, base + ".scale_int4", 1.0f);
            dp.scale_int8 = read_scalar_f32_gguf(model.wctx, base + ".scale_int8", 1.0f);
            dp.zp_low = read_scalar_f32_gguf(model.wctx, base + ".zp_low", 1.5f);
            dp.zp_int4 = read_scalar_f32_gguf(model.wctx, base + ".zp_int4", 7.5f);
            dp.zp_int8 = read_scalar_f32_gguf(model.wctx, base + ".zp_int8", 127.5f);

            dp.canonical_base_host = nullptr;
            dp.canonical_base = dp.host_packed_weights;
            dp.canonical_pw = dp.host_packed_weights;
            dp.canonical_pm = dp.host_packed_mask;
            dp.canonical_fv =
                dp.host_fp16_values ? reinterpret_cast<ggml_fp16_t *>(dp.host_fp16_values) : nullptr;

            void * pw_dev = nullptr;
            void * pm_dev = nullptr;
            void * fv_dev = nullptr;
            cudaError_t pw_map = cudaHostGetDevicePointer(&pw_dev, dp.host_packed_weights, 0);
            cudaError_t pm_map = cudaHostGetDevicePointer(&pm_dev, dp.host_packed_mask, 0);
            cudaError_t fv_map = cudaSuccess;
            if (dp.host_fp16_values) {
                fv_map = cudaHostGetDevicePointer(&fv_dev, dp.host_fp16_values, 0);
            }
            if (pw_map == cudaSuccess && pm_map == cudaSuccess && fv_map == cudaSuccess) {
                dp.canonical_pw_dev = pw_dev;
                dp.canonical_pm_dev = pm_dev;
                dp.canonical_fv_dev = fv_dev;
                dp.preloaded = true;
            } else {
                std::cerr << "[bmo_prepare_device_packed_tensors] cudaHostGetDevicePointer canonical map failed for "
                          << base << " pw=" << cudaGetErrorString(pw_map)
                          << " pm=" << cudaGetErrorString(pm_map)
                          << " fv=" << cudaGetErrorString(fv_map) << "\n";
                dp.preloaded = false;
                (void) cudaGetLastError();
            }

            {
                const uint8_t * pm_host_bo = reinterpret_cast<const uint8_t *>(pm->data);
                const int32_t blocks_per_row = cols / block_size;

                std::vector<int32_t> row_c2_host((size_t) rows, 0);
                std::vector<int32_t> row_c4_host((size_t) rows, 0);
                std::vector<int32_t> row_c8_host((size_t) rows, 0);
                std::vector<int32_t> row_c16_host((size_t) rows, 0);

                int32_t c2_bo = 0;
                int32_t c4_bo = 0;
                int32_t c8_bo = 0;
                int32_t c16_bo = 0;

                if (!v5_layout) {
                    if (blocks_per_row <= 0 || (int64_t) rows * (int64_t) blocks_per_row != (int64_t) n_blocks) {
                        std::cerr << "[bmo_prepare_device_packed_tensors] row/block geometry mismatch for " << base
                                  << " rows=" << rows << " cols=" << cols << " block_size=" << block_size
                                  << " n_blocks=" << n_blocks << "\n";
                        continue;
                    }

                    for (int32_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
                        if (blocks_per_row > 0 && (block_idx % blocks_per_row) == 0) {
                            const int32_t r = block_idx / blocks_per_row;
                            if (r >= 0 && r < rows) {
                                row_c2_host[(size_t) r] = c2_bo;
                                row_c4_host[(size_t) r] = c4_bo;
                                row_c8_host[(size_t) r] = c8_bo;
                                row_c16_host[(size_t) r] = c16_bo;
                            }
                        }

                        const uint8_t mbyte = pm_host_bo[(size_t) block_idx / 4];
                        const uint8_t tier_bo = unpack_u2_le(mbyte, block_idx % 4);
                        if (tier_bo == 0) {
                            c16_bo += block_size;
                        } else if (tier_bo == 1) {
                            c8_bo += block_size;
                        } else if (tier_bo == 2) {
                            c4_bo += block_size;
                        } else if (tier_bo == 3) {
                            c2_bo += block_size;
                        }
                    }
                } else {
                    for (int32_t r = 0; r < rows; ++r) {
                        row_c2_host[(size_t) r] = c2_bo;
                        row_c4_host[(size_t) r] = c4_bo;
                        row_c8_host[(size_t) r] = c8_bo;
                        row_c16_host[(size_t) r] = c16_bo;
                        for (int32_t c = 0; c < cols; ++c) {
                            const int64_t ei = (int64_t) r * (int64_t) cols + (int64_t) c;
                            const uint8_t mbyte = pm_host_bo[(size_t) (ei / 4)];
                            const uint8_t tier_bo = unpack_u2_le(mbyte, (int) (ei % 4));
                            if (tier_bo == 0) {
                                ++c16_bo;
                            } else if (tier_bo == 1) {
                                ++c8_bo;
                            } else if (tier_bo == 2) {
                                ++c4_bo;
                            } else if (tier_bo == 3) {
                                ++c2_bo;
                            }
                        }
                    }

                    if ((int64_t) c16_bo != n_fp16) {
                        std::cerr << "[bmo_prepare_device_packed_tensors] " << base
                                  << ": fp16_values count mismatch for v5 mask: n_fp16_tensor=" << n_fp16
                                  << " tier0_elems=" << c16_bo << "\n";
                        continue;
                    }
                }

                const size_t row_tbl_bytes = (size_t) rows * sizeof(int32_t);
                int32_t * d_c2 = nullptr;
                int32_t * d_c4 = nullptr;
                int32_t * d_c8 = nullptr;
                int32_t * d_c16 = nullptr;

                cudaError_t r2_err = cudaMalloc(reinterpret_cast<void **>(&d_c2), row_tbl_bytes);
                if (r2_err != cudaSuccess) {
                    std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc row_c2 failed for " << base
                              << ": " << cudaGetErrorString(r2_err) << "\n";
                    continue;
                }
                cudaError_t r4_err = cudaMalloc(reinterpret_cast<void **>(&d_c4), row_tbl_bytes);
                if (r4_err != cudaSuccess) {
                    cudaFree(d_c2);
                    std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc row_c4 failed for " << base
                              << ": " << cudaGetErrorString(r4_err) << "\n";
                    continue;
                }
                cudaError_t r8_err = cudaMalloc(reinterpret_cast<void **>(&d_c8), row_tbl_bytes);
                if (r8_err != cudaSuccess) {
                    cudaFree(d_c2);
                    cudaFree(d_c4);
                    std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc row_c8 failed for " << base
                              << ": " << cudaGetErrorString(r8_err) << "\n";
                    continue;
                }
                cudaError_t r16_err = cudaMalloc(reinterpret_cast<void **>(&d_c16), row_tbl_bytes);
                if (r16_err != cudaSuccess) {
                    cudaFree(d_c2);
                    cudaFree(d_c4);
                    cudaFree(d_c8);
                    std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc row_c16 failed for " << base
                              << ": " << cudaGetErrorString(r16_err) << "\n";
                    continue;
                }

                r2_err = cudaMemcpy(d_c2, row_c2_host.data(), row_tbl_bytes, cudaMemcpyHostToDevice);
                r4_err = cudaMemcpy(d_c4, row_c4_host.data(), row_tbl_bytes, cudaMemcpyHostToDevice);
                r8_err = cudaMemcpy(d_c8, row_c8_host.data(), row_tbl_bytes, cudaMemcpyHostToDevice);
                r16_err = cudaMemcpy(d_c16, row_c16_host.data(), row_tbl_bytes, cudaMemcpyHostToDevice);
                if (r2_err != cudaSuccess || r4_err != cudaSuccess || r8_err != cudaSuccess ||
                    r16_err != cudaSuccess) {
                    cudaFree(d_c2);
                    cudaFree(d_c4);
                    cudaFree(d_c8);
                    cudaFree(d_c16);
                    std::cerr << "[bmo_prepare_device_packed_tensors] cudaMemcpy row tier bases failed for "
                              << base << ": "
                              << cudaGetErrorString(r2_err != cudaSuccess ? r2_err
                                                       : r4_err != cudaSuccess ? r4_err
                                                       : r8_err != cudaSuccess ? r8_err
                                                                               : r16_err)
                              << "\n";
                    continue;
                }

                dp.row_c2 = d_c2;
                dp.row_c4 = d_c4;
                dp.row_c8 = d_c8;
                dp.row_c16 = d_c16;
            }
#else
            if (!v5_layout) {
                err = cudaMalloc(&dp.packed_weights, pw_bytes);
                if (err != cudaSuccess) {
                    std::cerr << "cudaMalloc packed_weights failed: " << cudaGetErrorString(err) << "\n";
                    continue;
                }
                err = cudaMemcpy(dp.packed_weights, pw->data, pw_bytes, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    std::cerr << "cudaMemcpy packed_weights failed: " << cudaGetErrorString(err) << "\n";
                    free_device_packed_owned_buffers(dp);
                    continue;
                }

                err = cudaMalloc(&dp.packed_mask, pm_bytes);
                if (err != cudaSuccess) {
                    std::cerr << "cudaMalloc packed_mask failed: " << cudaGetErrorString(err) << "\n";
                    free_device_packed_owned_buffers(dp);
                    continue;
                }
                err = cudaMemcpy(dp.packed_mask, pm->data, pm_bytes, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    std::cerr << "cudaMemcpy packed_mask failed: " << cudaGetErrorString(err) << "\n";
                    free_device_packed_owned_buffers(dp);
                    continue;
                }
            } else {
                std::cerr << "[bmo_prepare_device_packed_tensors] " << base
                          << ": packing_version>=5 per-element mask; skipping desktop CUDA staging "
                             "(CPU unpack path will be used until Path B CUDA lands).\n";
            }
#endif

#ifndef BMO_JETSON
            if (!v5_layout) {
                err = cudaMalloc(&dp.block_offset, block_offset.size() * sizeof(int32_t));
                if (err != cudaSuccess) {
                    std::cerr << "cudaMalloc block_offset failed: " << cudaGetErrorString(err) << "\n";
                    free_device_packed_owned_buffers(dp);
                    continue;
                }
                err = cudaMemcpy(dp.block_offset, block_offset.data(), block_offset.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    std::cerr << "cudaMemcpy block_offset failed: " << cudaGetErrorString(err) << "\n";
                    free_device_packed_owned_buffers(dp);
                    continue;
                }

                // fp16 values: may be stored as f32 in gguf; convert if needed
                if (fv->type == GGML_TYPE_F32) {
                    int64_t nf = n_fp16;
                    std::vector<ggml_fp16_t> tmp_f16((size_t) nf);
                    const float * src = reinterpret_cast<const float *>(fv->data);
                    for (int64_t j = 0; j < nf; ++j) tmp_f16[(size_t) j] = ggml_fp32_to_fp16(src[j]);
                    size_t fv_bytes = (size_t) nf * sizeof(ggml_fp16_t);
                    err = cudaMalloc(&dp.fp16_values, fv_bytes);
                    if (err != cudaSuccess) {
                        std::cerr << "cudaMalloc fp16_values failed: " << cudaGetErrorString(err) << "\n";
                        goto cleanup_partial_dp;
                    }
                    err = cudaMemcpy(dp.fp16_values, tmp_f16.data(), fv_bytes, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess) {
                        std::cerr << "cudaMemcpy fp16_values failed: " << cudaGetErrorString(err) << "\n";
                        goto cleanup_partial_dp;
                    }
                } else {
                    size_t fv_bytes = (size_t) ggml_nbytes(fv);
                    err = cudaMalloc(&dp.fp16_values, fv_bytes);
                    if (err != cudaSuccess) {
                        std::cerr << "cudaMalloc fp16_values failed: " << cudaGetErrorString(err) << "\n";
                        goto cleanup_partial_dp;
                    }
                    err = cudaMemcpy(dp.fp16_values, fv->data, fv_bytes, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess) {
                        std::cerr << "cudaMemcpy fp16_values failed: " << cudaGetErrorString(err) << "\n";
                        goto cleanup_partial_dp;
                    }
                }
            }
#endif

#ifdef BMO_JETSON
            // v5 uses fused_dequant_matvec_kernel_proto in bmo_compute; desktop CUDA still skips v5 unpack.
            dp.is_valid = true;
            // H2 diagnostic: verify v5 quantized tensors are registered with distinct device pointers.
            if (dp.is_valid &&
                (base.find("self_attn_in_proj_weight") != std::string::npos ||
                 base.find("gating_linear_in_weight") != std::string::npos ||
                 base.find("gating_linear_out_weight") != std::string::npos)) {
                std::string pretty_name = base;
                auto repl = [&](const char * from, const char * to) {
                    const std::string f(from);
                    const size_t pos = pretty_name.find(f);
                    if (pos != std::string::npos) {
                        pretty_name.replace(pos, f.size(), to);
                    }
                };
                repl("self_attn_in_proj_weight", "self_attn.in_proj_weight");
                repl("gating_linear_in_weight", "gating.linear_in.weight");
                repl("gating_linear_out_weight", "gating.linear_out.weight");

                std::fprintf(stderr,
                             "[h2_diag_register] %s | pv=%d valid=%d "
                             "row_c2=%p row_c4=%p row_c8=%p row_c16=%p "
                             "canonical_int2=%p canonical_int4=%p canonical_int8=%p canonical_fp16=%p "
                             "packed_mask=%p\n",
                             pretty_name.c_str(),
                             (int) dp.packing_version,
                             (int) dp.is_valid,
                             (void *) dp.row_c2,
                             (void *) dp.row_c4,
                             (void *) dp.row_c8,
                             (void *) dp.row_c16,
                             (void *) dp.canonical_pw_dev,  // mapped as "canonical_int2" in this log
                             (void *) dp.canonical_pw_dev,  // mapped as "canonical_int4"
                             (void *) dp.canonical_pw_dev,  // mapped as "canonical_int8"
                             (void *) dp.canonical_fv_dev,  // mapped as "canonical_fp16"
                             (void *) dp.canonical_pm_dev   // mapped as "packed_mask"
                );
            }
#else
            dp.is_valid = !v5_layout;
#endif
            ctx.packed_registry[base] = dp;
#ifndef BMO_JETSON
            if (!v5_layout) {
                max_unpack_elems = std::max(max_unpack_elems, (size_t) rows * (size_t) cols);
            }
#endif
            if (getenv("BMO_LOG_INIT")) {
                std::cout << "[bmo_prepare_device_packed_tensors] registered " << base << " rows=" << rows << " cols=" << cols << " n_fp16=" << n_fp16 << "\n";
            }
            continue;

            cleanup_partial_dp:
                free_device_packed_owned_buffers(dp);
                continue;
        }
    }

#ifdef BMO_JETSON
    {
        ctx.cuda_fused_output_buffer = nullptr;
        ctx.cuda_fused_output_buffer_dev = nullptr;
        ctx.cuda_fused_output_buffer_bytes = 0;
        ctx.cuda_fused_output_owns_raw_malloc = false;

        ctx.cuda_fused_input_buffer = nullptr;
        ctx.cuda_fused_input_buffer_dev = nullptr;
        ctx.cuda_fused_input_owns_raw_malloc = false;

        size_t out_bytes = 22528ULL * sizeof(float);
        out_bytes = (out_bytes + 63) & ~size_t(63);
        void * out_raw = nullptr;
        if (posix_memalign(&out_raw, 64, out_bytes) == 0 && out_raw) {
            cudaError_t out_reg =
                cudaHostRegister(out_raw, out_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
            if (out_reg == cudaSuccess) {
                if (cudaHostGetDevicePointer(&ctx.cuda_fused_output_buffer_dev, out_raw, 0) == cudaSuccess) {
                    ctx.cuda_fused_output_buffer = out_raw;
                    ctx.cuda_fused_output_buffer_bytes = out_bytes;
                    ctx.cuda_fused_output_owns_raw_malloc = true;
                    std::cout << "[bmo_jetson] fused_output_buffer="
                              << (double) out_bytes / (1024.0 * 1024.0) << " MB pinned mapped\n";
                } else {
                    cudaHostUnregister(out_raw);
                    std::free(out_raw);
                    std::cerr << "[bmo_jetson] cudaHostGetDevicePointer fused_output failed\n";
                    (void) cudaGetLastError();
                }
            } else {
                std::cerr << "[bmo_jetson] cudaHostRegister fused_output failed: " << cudaGetErrorString(out_reg)
                          << "\n";
                std::free(out_raw);
                (void) cudaGetLastError();
            }
        } else {
            std::cerr << "[bmo_jetson] posix_memalign fused_output failed\n";
        }

        size_t in_bytes = 11264ULL * sizeof(float);
        in_bytes = (in_bytes + 63) & ~size_t(63);
        void * in_raw = nullptr;
        if (posix_memalign(&in_raw, 64, in_bytes) == 0 && in_raw) {
            cudaError_t in_reg =
                cudaHostRegister(in_raw, in_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
            if (in_reg == cudaSuccess) {
                if (cudaHostGetDevicePointer(&ctx.cuda_fused_input_buffer_dev, in_raw, 0) == cudaSuccess) {
                    ctx.cuda_fused_input_buffer = in_raw;
                    ctx.cuda_fused_input_buffer_bytes = in_bytes;
                    ctx.cuda_fused_input_owns_raw_malloc = true;
                    std::cout << "[bmo_jetson] fused_input_buffer="
                              << (double) in_bytes / (1024.0 * 1024.0) << " MB pinned mapped\n";
                } else {
                    cudaHostUnregister(in_raw);
                    std::free(in_raw);
                    std::cerr << "[bmo_jetson] cudaHostGetDevicePointer fused_input failed\n";
                    (void) cudaGetLastError();
                }
            } else {
                std::cerr << "[bmo_jetson] cudaHostRegister fused_input failed: " << cudaGetErrorString(in_reg)
                          << "\n";
                std::free(in_raw);
                (void) cudaGetLastError();
            }
        } else {
            std::cerr << "[bmo_jetson] posix_memalign fused_input failed\n";
        }
    }
    if (!ctx.q8_scratch_dev) {
        cudaError_t q8_err = cudaMalloc(&ctx.q8_scratch_dev, 65536);
        if (q8_err != cudaSuccess) {
            std::cerr << "[bmo_jetson] cudaMalloc q8_scratch_dev failed: " << cudaGetErrorString(q8_err) << "\n";
            ctx.q8_scratch_dev = nullptr;
        }
    }
#endif
#ifndef BMO_JETSON
    const size_t scratch_bytes = max_unpack_elems * sizeof(float);
    if (scratch_bytes > 0) {
        cudaError_t err = cudaMalloc(&ctx.cuda_unpack_scratch, scratch_bytes);
        ctx.cuda_unpack_scratch_dev = nullptr;
        if (err == cudaSuccess) {
            ctx.cuda_unpack_scratch_dev = ctx.cuda_unpack_scratch;
            ctx.cuda_unpack_scratch_bytes = scratch_bytes;
            std::cout << "[bmo_prepare_device_packed_tensors] cuda_unpack_scratch="
                      << (double) scratch_bytes / (1024.0 * 1024.0) << " MB\n";
        } else {
            ctx.cuda_unpack_scratch = nullptr;
            ctx.cuda_unpack_scratch_dev = nullptr;
            ctx.cuda_unpack_scratch_bytes = 0;
            std::cerr << "[bmo_prepare_device_packed_tensors] cudaMalloc scratch failed: "
                      << cudaGetErrorString(err) << "\n";
        }
    }
#endif
    bmo_print_mem_diag("End Prepare");
#endif
}

void bmo_free_cuda_resources(bmo_context & ctx) {
#ifdef BMO_ENABLE_CUDA
    for (auto & kv : ctx.packed_registry) {
        device_packed_t & dp = kv.second;
#ifdef BMO_JETSON
        if (dp.canonical_base_host) {
            cudaHostUnregister(dp.canonical_base_host);
            std::free(dp.canonical_base_host);
            dp.canonical_base_host = nullptr;
        }
#endif
        free_device_packed_owned_buffers(dp);
    }
    ctx.packed_registry.clear();

#ifndef BMO_JETSON
    if (ctx.cuda_unpack_scratch) {
        cudaFree(ctx.cuda_unpack_scratch);
        ctx.cuda_unpack_scratch = nullptr;
    }
    ctx.cuda_unpack_scratch_dev = nullptr;
    ctx.cuda_unpack_scratch_bytes = 0;
    ctx.cuda_unpack_scratch_managed = false;
    ctx.cuda_unpack_scratch_owns_raw_malloc = false;
#endif
#ifdef BMO_JETSON
    if (ctx.cuda_fused_output_buffer && ctx.cuda_fused_output_owns_raw_malloc) {
        cudaHostUnregister(ctx.cuda_fused_output_buffer);
        std::free(ctx.cuda_fused_output_buffer);
        ctx.cuda_fused_output_buffer = nullptr;
    }
    ctx.cuda_fused_output_buffer_dev = nullptr;
    ctx.cuda_fused_output_buffer_bytes = 0;
    ctx.cuda_fused_output_owns_raw_malloc = false;
    if (ctx.cuda_fused_input_buffer && ctx.cuda_fused_input_owns_raw_malloc) {
        cudaHostUnregister(ctx.cuda_fused_input_buffer);
        std::free(ctx.cuda_fused_input_buffer);
        ctx.cuda_fused_input_buffer = nullptr;
    }
    ctx.cuda_fused_input_buffer_dev = nullptr;
    ctx.cuda_fused_input_buffer_bytes = 0;
    ctx.cuda_fused_input_owns_raw_malloc = false;
    if (ctx.streaming_big_pool_registered && ctx.streaming_big_pool) {
        cudaHostUnregister(ctx.streaming_big_pool);
        ctx.streaming_big_pool_registered = false;
    }
    if (ctx.streaming_big_pool) {
        std::free(ctx.streaming_big_pool);
        ctx.streaming_big_pool = nullptr;
    }
    ctx.streaming_big_pool_size = 0;
    if (ctx.streaming_scalar_pool_registered && ctx.streaming_scalar_pool) {
        cudaHostUnregister(ctx.streaming_scalar_pool);
        ctx.streaming_scalar_pool_registered = false;
    }
    if (ctx.streaming_scalar_pool) {
        std::free(ctx.streaming_scalar_pool);
        ctx.streaming_scalar_pool = nullptr;
    }
    ctx.streaming_scalar_pool_size = 0;
    for (int i = 0; i < gpu_staging_pool::N_SLOTS; ++i) {
        if (ctx.staging.host[i]) {
            cudaHostUnregister(ctx.staging.host[i]);
            std::free(ctx.staging.host[i]);
            ctx.staging.host[i] = nullptr;
        }
        ctx.staging.dev[i] = nullptr;
        ctx.staging.in_use[i] = false;
    }
    if (ctx.q8_scratch_dev) {
        cudaFree(ctx.q8_scratch_dev);
        ctx.q8_scratch_dev = nullptr;
    }
    if (ctx.kv_mem_registered && ctx.kv_mem) {
        cudaHostUnregister(ctx.kv_mem.get());
        ctx.kv_mem_registered = false;
    }
    if (ctx.depth_kv_mem_registered && ctx.depth_kv_mem) {
        cudaHostUnregister(ctx.depth_kv_mem.get());
        ctx.depth_kv_mem_registered = false;
    }
#endif
#else
    (void) ctx;
#endif
}

void bmo_init_kv_cache(bmo_context & ctx, int32_t n_ctx) {
    if (ctx.n_heads <= 0 || ctx.head_dim <= 0 || ctx.n_layers <= 0) {
        throw std::runtime_error("KV cache init requires valid n_layers, n_heads and head_dim in context");
    }

#ifdef BMO_JETSON
    // KV cache scales linearly: at n_ctx=128 we measured ~64 MB total
    // (~0.5 MB/slot across all 32 layers, k+v in fp16). On the 8 GB
    // Orin Nano with ~1 GB free after model load, n_ctx=1024 (≈512 MB)
    // is the safe ceiling. The earlier cap of 128 was overly defensive
    // and broke voice-prompt prefill (a 5-10 s voice prompt alone is
    // 60-125 frames before any generation slots are consumed). Callers
    // can shrink to 128 explicitly via --n-ctx; raising further than
    // 1024 risks pushing RSS past 8 GB during sampling.
    constexpr int BMO_JETSON_KV_MAX = 1024;
    if (n_ctx > BMO_JETSON_KV_MAX) {
        std::cout << "[bmo_jetson] Capping n_ctx from " << n_ctx << " to "
                  << BMO_JETSON_KV_MAX << " (KV cache budget on Orin Nano)\n";
        n_ctx = BMO_JETSON_KV_MAX;
    }
#endif
    ctx.n_ctx = n_ctx;

    // Estimate required memory: two caches (k and v) stored as f16
    const int64_t elems_per_layer = (int64_t) n_ctx * (int64_t) ctx.n_heads * (int64_t) ctx.head_dim;
    const size_t bytes_per_layer = (size_t) elems_per_layer * sizeof(ggml_fp16_t) * 2; // k + v
    const size_t total_bytes = bytes_per_layer * (size_t) ctx.n_layers;

    // If reinitializing KV cache, free previous allocations first.
    if (ctx.kv_ctx) {
        ggml_free(ctx.kv_ctx);
        ctx.kv_ctx = nullptr;
    }
#ifdef BMO_JETSON
    if (ctx.kv_mem_registered && ctx.kv_mem) {
        cudaHostUnregister(ctx.kv_mem.get());
        ctx.kv_mem_registered = false;
    }
#endif
    ctx.kv_mem.reset();

    // Keep KV cache in host memory. The CUDA backend is used only for SEPTQ unpacking.
    const size_t alloc_size = total_bytes + (1 << 20);
    ctx.kv_mem.reset(new uint8_t[alloc_size]);
    ggml_init_params iparams = {
        /*.mem_size   =*/ alloc_size,
        /*.mem_buffer =*/ ctx.kv_mem.get(),
        /*.no_alloc   =*/ false
    };
    ggml_context * kv_ctx = ggml_init(iparams);
    if (!kv_ctx) throw std::runtime_error("Failed to initialize KV ggml_context");

    ctx.kv_ctx = kv_ctx;
    ctx.k_cache = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F16, ctx.head_dim, n_ctx, ctx.n_heads, ctx.n_layers);
    ctx.v_cache = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F16, ctx.head_dim, n_ctx, ctx.n_heads, ctx.n_layers);
    if (!ctx.k_cache || !ctx.v_cache) {
        throw std::runtime_error("Failed to create KV tensors");
    }

#ifdef BMO_JETSON
    {
        cudaError_t kv_reg = cudaHostRegister(ctx.kv_mem.get(), alloc_size, cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (kv_reg == cudaSuccess) {
            ctx.kv_mem_registered = true;
            void * dev_base = nullptr;
            if (cudaHostGetDevicePointer(&dev_base, ctx.kv_mem.get(), 0) == cudaSuccess) {
                size_t k_offset = (size_t)((uint8_t *)ctx.k_cache->data - ctx.kv_mem.get());
                size_t v_offset = (size_t)((uint8_t *)ctx.v_cache->data - ctx.kv_mem.get());
                ctx.k_cache->extra = (uint8_t *)dev_base + k_offset;
                ctx.v_cache->extra = (uint8_t *)dev_base + v_offset;
            }
        } else {
            (void) cudaGetLastError();
        }
    }
#endif

    ctx.kv_bytes = (size_t) ggml_nbytes(ctx.k_cache) + (size_t) ggml_nbytes(ctx.v_cache);

    std::cout << "[bmo_init_kv_cache] Allocated KV cache: " << (double) ctx.kv_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "[bmo_init_kv_cache] per-layer estimate: " << (double) bytes_per_layer / (1024.0 * 1024.0) << " MB\n";

    // Allocate the depth-transformer KV cache. Dimensions are fixed by the
    // Moshi/PersonaPlex depformer: 1024 hidden / 16 heads / 64 head_dim / 6
    // layers, with up to dep_q codebook positions (capped at 16). The cache
    // is tiny (a few hundred KB) so it lives in plain host memory.
    {
        if (ctx.depth_kv_ctx) {
            ggml_free(ctx.depth_kv_ctx);
            ctx.depth_kv_ctx = nullptr;
        }
#ifdef BMO_JETSON
        if (ctx.depth_kv_mem_registered && ctx.depth_kv_mem) {
            cudaHostUnregister(ctx.depth_kv_mem.get());
            ctx.depth_kv_mem_registered = false;
        }
#endif
        ctx.depth_kv_mem.reset();

        ctx.depth_n_heads    = 16;
        ctx.depth_head_dim   = 64;
        ctx.depth_hidden_dim = ctx.depth_n_heads * ctx.depth_head_dim; // 1024
        ctx.depth_n_layers   = 6;
        const int32_t requested_dep_ctx = (ctx.dep_q > 0) ? ctx.dep_q : DEP_Q;
        ctx.depth_n_ctx      = (requested_dep_ctx > DEP_Q) ? DEP_Q : requested_dep_ctx;

        const int64_t depth_elems_per_layer =
            (int64_t) ctx.depth_n_ctx * (int64_t) ctx.depth_n_heads * (int64_t) ctx.depth_head_dim;
        const size_t depth_bytes_per_layer =
            (size_t) depth_elems_per_layer * sizeof(ggml_fp16_t) * 2; // k + v
        const size_t depth_total_bytes = depth_bytes_per_layer * (size_t) ctx.depth_n_layers;
        const size_t depth_alloc_size  = depth_total_bytes + (1 << 16); // small overhead for ggml metadata

        ctx.depth_kv_mem.reset(new uint8_t[depth_alloc_size]);
        ggml_init_params depth_iparams = {
            /*.mem_size   =*/ depth_alloc_size,
            /*.mem_buffer =*/ ctx.depth_kv_mem.get(),
            /*.no_alloc   =*/ false,
        };
        ctx.depth_kv_ctx = ggml_init(depth_iparams);
        if (!ctx.depth_kv_ctx) throw std::runtime_error("Failed to initialize depth KV ggml_context");

        ctx.depth_k_cache = ggml_new_tensor_4d(
            ctx.depth_kv_ctx, GGML_TYPE_F16,
            ctx.depth_head_dim, ctx.depth_n_ctx, ctx.depth_n_heads, ctx.depth_n_layers);
        ctx.depth_v_cache = ggml_new_tensor_4d(
            ctx.depth_kv_ctx, GGML_TYPE_F16,
            ctx.depth_head_dim, ctx.depth_n_ctx, ctx.depth_n_heads, ctx.depth_n_layers);
        if (!ctx.depth_k_cache || !ctx.depth_v_cache) {
            throw std::runtime_error("Failed to create depth KV tensors");
        }

#ifdef BMO_JETSON
        {
            cudaError_t depth_reg = cudaHostRegister(ctx.depth_kv_mem.get(), depth_alloc_size, cudaHostRegisterMapped | cudaHostRegisterPortable);
            if (depth_reg == cudaSuccess) {
                ctx.depth_kv_mem_registered = true;
                void * depth_dev_base = nullptr;
                if (cudaHostGetDevicePointer(&depth_dev_base, ctx.depth_kv_mem.get(), 0) == cudaSuccess) {
                    size_t k_offset = (size_t)((uint8_t *)ctx.depth_k_cache->data - ctx.depth_kv_mem.get());
                    size_t v_offset = (size_t)((uint8_t *)ctx.depth_v_cache->data - ctx.depth_kv_mem.get());
                    ctx.depth_k_cache->extra = (uint8_t *)depth_dev_base + k_offset;
                    ctx.depth_v_cache->extra = (uint8_t *)depth_dev_base + v_offset;
                }
            } else {
                (void) cudaGetLastError();
            }
        }
#endif

        ctx.depth_kv_bytes =
            (size_t) ggml_nbytes(ctx.depth_k_cache) + (size_t) ggml_nbytes(ctx.depth_v_cache);
        bmo_reset_depth_kv(ctx);

        std::cout << "[bmo_init_kv_cache] Allocated depth KV cache: "
                  << (double) ctx.depth_kv_bytes / 1024.0 << " KB"
                  << " (dep_ctx=" << ctx.depth_n_ctx
                  << " heads=" << ctx.depth_n_heads
                  << " head_dim=" << ctx.depth_head_dim
                  << " layers=" << ctx.depth_n_layers << ")\n";
    }

#ifdef BMO_JETSON
    // Allocate the generic GPU staging pool used by the fused-op interceptors
    // (RMSNorm, residual add, ...). Each slot is a fixed-size pinned/mapped
    // host buffer with a device alias suitable for direct kernel reads/writes.
    {
        int allocated = 0;
        for (int i = 0; i < gpu_staging_pool::N_SLOTS; ++i) {
            ctx.staging.host[i] = nullptr;
            ctx.staging.dev[i] = nullptr;
            ctx.staging.in_use[i] = false;

            if (posix_memalign(&ctx.staging.host[i], 64, gpu_staging_pool::SLOT_BYTES) != 0) {
                ctx.staging.host[i] = nullptr;
                std::cerr << "[bmo_init_kv_cache] posix_memalign staging slot " << i << " failed\n";
                continue;
            }
            cudaError_t reg = cudaHostRegister(
                ctx.staging.host[i],
                gpu_staging_pool::SLOT_BYTES,
                cudaHostRegisterMapped | cudaHostRegisterPortable);
            if (reg != cudaSuccess) {
                std::cerr << "[bmo_init_kv_cache] cudaHostRegister staging slot " << i
                          << " failed: " << cudaGetErrorString(reg) << "\n";
                std::free(ctx.staging.host[i]);
                ctx.staging.host[i] = nullptr;
                (void) cudaGetLastError();
                continue;
            }
            if (cudaHostGetDevicePointer(&ctx.staging.dev[i], ctx.staging.host[i], 0) != cudaSuccess) {
                std::cerr << "[bmo_init_kv_cache] cudaHostGetDevicePointer staging slot " << i
                          << " failed\n";
                cudaHostUnregister(ctx.staging.host[i]);
                std::free(ctx.staging.host[i]);
                ctx.staging.host[i] = nullptr;
                ctx.staging.dev[i] = nullptr;
                (void) cudaGetLastError();
                continue;
            }
            ++allocated;
        }
        std::cout << "[bmo_jetson] gpu_staging_pool: " << allocated << "/"
                  << gpu_staging_pool::N_SLOTS << " slots, "
                  << (double) (allocated * gpu_staging_pool::SLOT_BYTES) / 1024.0
                  << " KB pinned mapped\n";
    }
#endif
}

void bmo_reset_depth_kv(bmo_context & ctx) {
    if (ctx.depth_k_cache && ctx.depth_k_cache->data) {
        std::memset(ctx.depth_k_cache->data, 0, (size_t) ggml_nbytes(ctx.depth_k_cache));
    }
    if (ctx.depth_v_cache && ctx.depth_v_cache->data) {
        std::memset(ctx.depth_v_cache->data, 0, (size_t) ggml_nbytes(ctx.depth_v_cache));
    }
}

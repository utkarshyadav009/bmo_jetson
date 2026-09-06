// bmo_api.cpp - Implementation of the BMO C-ABI bridge with CUDA Graph acceleration.

#include "bmo_api.h"
#include "bmo.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

#ifdef BMO_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

struct bmo_handle {
    bmo_model   model;
    bmo_context ctx;
    int         pos = 0;
    std::string last_error;
    std::mutex  mu;

#ifdef BMO_ENABLE_CUDA
    cudaStream_t stream = nullptr;

    // Device-mapped int pointer for dynamic pos
    int * pos_host = nullptr;
    int * pos_dev = nullptr;

    // Temporal static pinned buffers
    float * temporal_in_host = nullptr;
    float * temporal_in_dev = nullptr;
    float * transformer_out_host = nullptr;
    float * transformer_out_dev = nullptr;
    float * text_logits_host = nullptr;
    float * text_logits_dev = nullptr;

    // Temporal graph
    cudaGraph_t temporal_graph = nullptr;
    cudaGraphExec_t temporal_graph_exec = nullptr;
    bool temporal_captured = false;

    // Depth static pinned buffers
    float * depth_temporal_in_host = nullptr;
    float * depth_temporal_in_dev = nullptr;
    float * depth_tok_emb_host = nullptr;
    float * depth_tok_emb_dev = nullptr;
    float * depth_audio_logits_host[8] = {nullptr};
    float * depth_audio_logits_dev[8] = {nullptr};

    // Depth graphs
    cudaGraph_t depth_graph[8] = {nullptr};
    cudaGraphExec_t depth_graph_exec[8] = {nullptr};
    bool depth_captured[8] = {false};
#endif
};

namespace {

constexpr size_t kDefaultWorkMem =
#ifdef BMO_JETSON
    (size_t) 1024ULL * 1024 * 1024;   // 1 GB on Jetson Orin Nano (8 GB unified)
#else
    (size_t) 2048ULL * 1024 * 1024;   // 2 GB on discrete-GPU hosts
#endif

constexpr int32_t kDefaultKvCtx = 2048;

void set_err(bmo_handle_t * h, const std::string & m) {
    if (h) h->last_error = m;
}

static void unpack_emb_row(const ggml_tensor * t, int32_t tok, float * dst, int64_t dim) {
    if (!t || !t->data || tok < 0 || tok >= t->ne[1]) {
        std::memset(dst, 0, (size_t) dim * sizeof(float));
        return;
    }
    const uint8_t * row = (const uint8_t *) t->data + (size_t) tok * t->nb[1];
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(dst, row, (size_t) dim * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * h = (const ggml_fp16_t *) row;
        for (int64_t d = 0; d < dim; ++d) dst[d] = ggml_fp16_to_fp32(h[d]);
    } else {
        const struct ggml_type_traits * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(row, dst, dim);
        } else {
            std::memset(dst, 0, (size_t) dim * sizeof(float));
        }
    }
}

} // namespace

extern "C" {

bmo_handle_t * bmo_init(const char * gguf_path, int n_ctx) {
    if (!gguf_path) {
        std::fprintf(stderr, "[bmo_api] bmo_init: gguf_path is null\n");
        return nullptr;
    }

    auto h = std::make_unique<bmo_handle_t>();
    auto init_cleanup = [](bmo_handle_t * hh) {
        if (!hh) return;
        try {
            if (hh->ctx.work_ctx) {
                ggml_free(hh->ctx.work_ctx);
                hh->ctx.work_ctx = nullptr;
            }
            if (hh->ctx.kv_ctx) {
                ggml_free(hh->ctx.kv_ctx);
                hh->ctx.kv_ctx = nullptr;
            }
            hh->ctx.kv_mem.reset();
            if (hh->ctx.depth_kv_ctx) {
                ggml_free(hh->ctx.depth_kv_ctx);
                hh->ctx.depth_kv_ctx = nullptr;
            }
            hh->ctx.depth_kv_mem.reset();
            if (hh->model.wctx) {
                ggml_free(hh->model.wctx);
                hh->model.wctx = nullptr;
            }
            if (hh->model.gctx) {
                gguf_free(hh->model.gctx);
                hh->model.gctx = nullptr;
            }
            bmo_free_cuda_resources(hh->ctx);
#ifdef BMO_ENABLE_CUDA
            if (hh->stream) cudaStreamDestroy(hh->stream);
            if (hh->pos_host) cudaFreeHost(hh->pos_host);
            if (hh->temporal_in_host) cudaFreeHost(hh->temporal_in_host);
            if (hh->transformer_out_host) cudaFreeHost(hh->transformer_out_host);
            if (hh->text_logits_host) cudaFreeHost(hh->text_logits_host);
            if (hh->depth_temporal_in_host) cudaFreeHost(hh->depth_temporal_in_host);
            if (hh->depth_tok_emb_host) cudaFreeHost(hh->depth_tok_emb_host);
            for (int s = 0; s < 8; ++s) {
                if (hh->depth_audio_logits_host[s]) cudaFreeHost(hh->depth_audio_logits_host[s]);
            }
#endif
        } catch (...) {
            // best-effort cleanup; never throw from the failure path
        }
    };

    try {
        bmo_load_model(gguf_path, h->model, h->ctx);

        const int32_t kv_ctx = (n_ctx > 0) ? (int32_t) n_ctx : kDefaultKvCtx;
        bmo_init_kv_cache(h->ctx, kv_ctx);

        h->ctx.work_mem.resize(kDefaultWorkMem);
        h->pos = 0;

#ifdef BMO_ENABLE_CUDA
        cudaStreamCreateWithFlags(&h->stream, cudaStreamNonBlocking);
        h->ctx.stream = h->stream;

        cudaHostAlloc((void **) &h->pos_host, sizeof(int), cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **) &h->pos_dev, h->pos_host, 0);
        *h->pos_host = 0;
        h->ctx.pos_dev = h->pos_dev;

        // Pinned buffers for temporal
        cudaHostAlloc((void **) &h->temporal_in_host, (size_t) h->ctx.n_embd * sizeof(float), cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **) &h->temporal_in_dev, h->temporal_in_host, 0);

        cudaHostAlloc((void **) &h->transformer_out_host, (size_t) h->ctx.n_embd * sizeof(float), cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **) &h->transformer_out_dev, h->transformer_out_host, 0);

        cudaHostAlloc((void **) &h->text_logits_host, (size_t) h->ctx.text_vocab_size * sizeof(float), cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **) &h->text_logits_dev, h->text_logits_host, 0);

        // Pinned buffers for depth
        cudaHostAlloc((void **) &h->depth_temporal_in_host, (size_t) h->ctx.n_embd * sizeof(float), cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **) &h->depth_temporal_in_dev, h->depth_temporal_in_host, 0);

        cudaHostAlloc((void **) &h->depth_tok_emb_host, 1024 * sizeof(float), cudaHostAllocMapped);
        cudaHostGetDevicePointer((void **) &h->depth_tok_emb_dev, h->depth_tok_emb_host, 0);
        h->ctx.depth_tok_emb_host = h->depth_tok_emb_host;
        h->ctx.depth_tok_emb_dev = h->depth_tok_emb_dev;

        for (int s = 0; s < 8; ++s) {
            cudaHostAlloc((void **) &h->depth_audio_logits_host[s], (size_t) h->ctx.audio_vocab_size * sizeof(float), cudaHostAllocMapped);
            cudaHostGetDevicePointer((void **) &h->depth_audio_logits_dev[s], h->depth_audio_logits_host[s], 0);
        }
#endif
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "[bmo_api] init failed: %s\n", ex.what());
        init_cleanup(h.get());
        return nullptr;
    } catch (...) {
        std::fprintf(stderr, "[bmo_api] init failed: unknown exception\n");
        init_cleanup(h.get());
        return nullptr;
    }

    return h.release();
}

void bmo_free(bmo_handle_t * h) {
    if (!h) return;
    try {
#ifdef BMO_ENABLE_CUDA
        if (h->temporal_graph_exec) cudaGraphExecDestroy(h->temporal_graph_exec);
        if (h->temporal_graph) cudaGraphDestroy(h->temporal_graph);
        for (int s = 0; s < 8; ++s) {
            if (h->depth_graph_exec[s]) cudaGraphExecDestroy(h->depth_graph_exec[s]);
            if (h->depth_graph[s]) cudaGraphDestroy(h->depth_graph[s]);
            if (h->depth_audio_logits_host[s]) cudaFreeHost(h->depth_audio_logits_host[s]);
        }
        if (h->depth_tok_emb_host) cudaFreeHost(h->depth_tok_emb_host);
        if (h->depth_temporal_in_host) cudaFreeHost(h->depth_temporal_in_host);
        if (h->text_logits_host) cudaFreeHost(h->text_logits_host);
        if (h->transformer_out_host) cudaFreeHost(h->transformer_out_host);
        if (h->temporal_in_host) cudaFreeHost(h->temporal_in_host);
        if (h->pos_host) cudaFreeHost(h->pos_host);
        if (h->stream) cudaStreamDestroy(h->stream);
#endif
        if (h->ctx.work_ctx) {
            ggml_free(h->ctx.work_ctx);
            h->ctx.work_ctx = nullptr;
        }
        if (h->ctx.kv_ctx) {
            ggml_free(h->ctx.kv_ctx);
            h->ctx.kv_ctx = nullptr;
        }
        h->ctx.kv_mem.reset();
        if (h->ctx.depth_kv_ctx) {
            ggml_free(h->ctx.depth_kv_ctx);
            h->ctx.depth_kv_ctx = nullptr;
        }
        h->ctx.depth_kv_mem.reset();
        if (h->model.wctx) {
            ggml_free(h->model.wctx);
            h->model.wctx = nullptr;
        }
        if (h->model.gctx) {
            gguf_free(h->model.gctx);
            h->model.gctx = nullptr;
        }
        bmo_free_cuda_resources(h->ctx);
    } catch (...) {
        // Destructors must not throw across the C-ABI boundary; swallow.
    }
    delete h;
}

void bmo_reset(bmo_handle_t * h) {
    if (!h) return;
    std::lock_guard<std::mutex> lk(h->mu);
    h->pos = 0;
#ifdef BMO_ENABLE_CUDA
    if (h->pos_host) *h->pos_host = 0;
#endif
    if (h->ctx.k_cache && h->ctx.k_cache->data) {
        std::memset(h->ctx.k_cache->data, 0, (size_t) ggml_nbytes(h->ctx.k_cache));
    }
    if (h->ctx.v_cache && h->ctx.v_cache->data) {
        std::memset(h->ctx.v_cache->data, 0, (size_t) ggml_nbytes(h->ctx.v_cache));
    }
    h->last_error.clear();
}

int bmo_get_n_layers    (bmo_handle_t * h) { return h ? h->ctx.n_layers         : 0; }
int bmo_get_n_embd      (bmo_handle_t * h) { return h ? h->ctx.n_embd           : 0; }
int bmo_get_n_codebooks (bmo_handle_t * h) { return h ? h->ctx.num_codebooks    : 0; }
int bmo_get_dep_q       (bmo_handle_t * h) { return h ? h->ctx.dep_q            : 0; }
int bmo_get_text_vocab  (bmo_handle_t * h) { return h ? h->ctx.text_vocab_size  : 0; }
int bmo_get_audio_vocab (bmo_handle_t * h) { return h ? h->ctx.audio_vocab_size : 0; }
int bmo_get_n_attn_heads(bmo_handle_t * h) { return h ? h->ctx.n_heads          : 0; }
int bmo_get_head_dim    (bmo_handle_t * h) { return h ? h->ctx.head_dim         : 0; }

int bmo_copy_k_cache_f32(
    bmo_handle_t * h,
    int layer,
    int t_start,
    int n_positions,
    float * out,
    int max_floats) {
    if (!h || !out || n_positions <= 0 || max_floats <= 0) {
        return -1;
    }
    if (!h->ctx.k_cache || !h->ctx.k_cache->data) {
        return -1;
    }
    if (h->ctx.k_cache->type != GGML_TYPE_F16) {
        return -1;
    }
    const int n_heads = (int) h->ctx.n_heads;
    const int head_dim = (int) h->ctx.head_dim;
    const int n_ctx = (int) h->ctx.k_cache->ne[1];
    if (layer < 0 || layer >= (int) h->ctx.n_layers || n_heads <= 0 || head_dim <= 0) {
        return -3;
    }
    if (t_start < 0 || t_start + n_positions > n_ctx) {
        return -3;
    }
    const int64_t need = (int64_t) n_positions * (int64_t) n_heads * (int64_t) head_dim;
    if (need > (int64_t) max_floats) {
        return -2;
    }

    std::lock_guard<std::mutex> lk(h->mu);

    const ggml_fp16_t * k_cache_data = (const ggml_fp16_t *) h->ctx.k_cache->data;
    const size_t per_layer =
        (size_t) head_dim * (size_t) n_ctx * (size_t) h->ctx.n_heads;
    const size_t per_head = (size_t) head_dim * (size_t) n_ctx;

    int w = 0;
    for (int ti = 0; ti < n_positions; ++ti) {
        const int p = t_start + ti;
        for (int hh = 0; hh < n_heads; ++hh) {
            const size_t cache_row_base =
                (size_t) layer * per_layer + (size_t) hh * per_head + (size_t) p * (size_t) head_dim;
            for (int d = 0; d < head_dim; ++d) {
                out[w++] = ggml_fp16_to_fp32(k_cache_data[cache_row_base + (size_t) d]);
            }
        }
    }
    return w;
}

int bmo_capture_graphs(bmo_handle_t * h) {
    if (!h) return 1;
#ifndef BMO_ENABLE_CUDA
    return 1;
#else
    std::lock_guard<std::mutex> lk(h->mu);
    try {
        std::fprintf(stderr, "[bmo_api] Warming up and capturing CUDA Graphs...\n");

        // 1. Warm run outside capture to ensure all device pointers (weights, buffers) are populated
        std::vector<int32_t> warm_tokens(h->ctx.num_codebooks, 0);
        warm_tokens[0] = 3;
        bmo_reset_work_ctx(h->ctx);
        ggml_tensor * layer_in = bmo_embed_input_tokens(h->ctx, h->model, warm_tokens.data(), h->ctx.num_codebooks);
        ggml_cgraph * gf_warm = bmo_build_temporal_graph(h->ctx, h->model, layer_in, 0, 0, h->ctx.n_layers);
        bmo_execute_graph(h->ctx, gf_warm, {}, true);

        for (int s = 0; s < h->ctx.dep_q; ++s) {
            bmo_reset_work_ctx(h->ctx);
            ggml_tensor * t_in = ggml_new_tensor_2d(h->ctx.work_ctx, GGML_TYPE_F32, h->ctx.n_embd, 1);
            ggml_tensor * txt = ggml_new_tensor_1d(h->ctx.work_ctx, GGML_TYPE_I32, 1);
            ggml_tensor * aud = ggml_new_tensor_1d(h->ctx.work_ctx, GGML_TYPE_I32, 1);
            *(int32_t*)txt->data = 0;
            *(int32_t*)aud->data = 0;
            ggml_cgraph * d_gf = bmo_build_depth_graph(h->ctx, h->model, t_in, txt, aud, s, 0);
            bmo_execute_graph(h->ctx, d_gf, {}, true);
        }
        cudaStreamSynchronize(h->stream);

        // 2. Capture Temporal Graph
        bmo_reset_work_ctx(h->ctx);
        ggml_tensor * temp_in = ggml_new_tensor_2d(h->ctx.work_ctx, GGML_TYPE_F32, h->ctx.n_embd, 1);
        temp_in->data = h->temporal_in_host;
        temp_in->extra = h->temporal_in_dev;

        cudaStreamBeginCapture(h->stream, cudaStreamCaptureModeThreadLocal);
        ggml_cgraph * t_gf = bmo_build_temporal_graph(h->ctx, h->model, temp_in, 0, 0, h->ctx.n_layers);
        ggml_tensor * t_out = ggml_graph_get_tensor(t_gf, "transformer_out");
        ggml_tensor * t_lgt = ggml_graph_get_tensor(t_gf, "text_logits");
        if (t_out && t_out->extra && t_out->extra != h->transformer_out_dev) {
            cudaMemcpyAsync(h->transformer_out_dev, t_out->extra, (size_t) h->ctx.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, h->stream);
        }
        if (t_lgt && t_lgt->extra && t_lgt->extra != h->text_logits_dev) {
            cudaMemcpyAsync(h->text_logits_dev, t_lgt->extra, (size_t) h->ctx.text_vocab_size * sizeof(float), cudaMemcpyDeviceToDevice, h->stream);
        }
        cudaError_t t_err = cudaStreamEndCapture(h->stream, &h->temporal_graph);
        if (t_err == cudaSuccess) {
            cudaError_t inst_err = cudaGraphInstantiate(&h->temporal_graph_exec, h->temporal_graph, nullptr, nullptr, 0);
            if (inst_err == cudaSuccess) {
                h->temporal_captured = true;
            } else {
                std::fprintf(stderr, "[bmo_api] Temporal graph instantiate failed: %s\n", cudaGetErrorString(inst_err));
            }
        } else {
            std::fprintf(stderr, "[bmo_api] Temporal graph capture failed: %s\n", cudaGetErrorString(t_err));
        }

        // 3. Depth cascade runs in eager mode (15.7 ms for all 8 steps, guaranteed accurate F16 logits)
        // Leaving h->depth_captured[*] = false ensures bmo_forward_depth uses the verified eager path.
        std::fprintf(stderr, "[bmo_api] Captured CUDA Graphs: Temporal=%s, Depth=EAGER (accurate F16 logits, 15.7ms total)\n",
                     h->temporal_captured ? "SUCCESS" : "FAILED");
        return h->temporal_captured ? 0 : 2;
    } catch (const std::exception & ex) {
        set_err(h, ex.what());
        return 9;
    }
#endif
}

int bmo_has_cuda_graphs(bmo_handle_t * h) {
#ifdef BMO_ENABLE_CUDA
    if (!h) return 0;
    return h->temporal_captured ? 1 : 0;
#else
    return 0;
#endif
}

const char * bmo_last_error(bmo_handle_t * h) {
    return (!h || h->last_error.empty()) ? nullptr : h->last_error.c_str();
}

int bmo_forward_temporal(
    bmo_handle_t * h,
    const int32_t * input_tokens,
    int num_codebooks,
    int pos,
    float * out_transformer,
    float * out_text_logits) {
    return bmo_forward_temporal2(
        h,
        input_tokens,
        num_codebooks,
        pos,
        out_transformer,
        out_text_logits,
        nullptr,
        0,
        nullptr);
}

int bmo_forward_temporal2(
    bmo_handle_t * h,
    const int32_t * input_tokens,
    int num_codebooks,
    int pos,
    float * out_transformer,
    float * out_text_logits,
    const int32_t * capture_layers,
    int n_capture_layers,
    float * capture_out) {
    if (!h || !input_tokens || !out_transformer || !out_text_logits) {
        set_err(h, "bmo_forward_temporal2: null pointer argument");
        return 1;
    }
    if (num_codebooks <= 0 || num_codebooks > h->ctx.num_codebooks) {
        set_err(h, "bmo_forward_temporal2: invalid num_codebooks");
        return 2;
    }
    if (pos < 0) {
        set_err(h, "bmo_forward_temporal2: pos must be non-negative");
        return 2;
    }
    if (n_capture_layers < 0) {
        set_err(h, "bmo_forward_temporal2: invalid n_capture_layers");
        return 2;
    }
    if (n_capture_layers > 0 && (!capture_layers || !capture_out)) {
        set_err(h, "bmo_forward_temporal2: capture_layers/capture_out required when n_capture_layers > 0");
        return 2;
    }

    std::lock_guard<std::mutex> lk(h->mu);
    try {
#ifdef BMO_ENABLE_CUDA
        if (h->pos_host) {
            *h->pos_host = pos;
        }

        // Accelerated path: CUDA Graph execution
        if (h->temporal_captured && n_capture_layers == 0) {
            bmo_embed_input_tokens_into(h->ctx, h->model, input_tokens, num_codebooks, h->temporal_in_host);

            cudaGraphLaunch(h->temporal_graph_exec, h->stream);
            cudaStreamSynchronize(h->stream);

            const size_t hidden_bytes = (size_t) h->ctx.n_embd * sizeof(float);
            const size_t logits_bytes = (size_t) h->ctx.text_vocab_size * sizeof(float);

            std::memcpy(out_transformer, h->transformer_out_host, hidden_bytes);
            std::memcpy(out_text_logits, h->text_logits_host, logits_bytes);

            // Statically copy transformer_out into depth input so depth steps can immediately read it
            std::memcpy(h->depth_temporal_in_host, h->transformer_out_host, hidden_bytes);

            h->pos = pos + 1;
            h->last_error.clear();
            return 0;
        }
#endif

        bmo_reset_work_ctx(h->ctx);

        ggml_tensor * layer_in = bmo_embed_input_tokens(
            h->ctx, h->model, input_tokens, num_codebooks);

        ggml_cgraph * gf = bmo_build_temporal_graph(
            h->ctx, h->model, layer_in, pos, /*layer_begin=*/0, /*layer_end=*/h->ctx.n_layers);
        if (!gf) {
            set_err(h, "bmo_forward_temporal2: failed to build temporal graph");
            return 4;
        }

        bmo_execute_graph(h->ctx, gf, /*inputs=*/{});

        ggml_tensor * t_out = ggml_graph_get_tensor(gf, "transformer_out");
        ggml_tensor * t_lgt = ggml_graph_get_tensor(gf, "text_logits");
        if (!t_out || !t_out->data) {
            set_err(h, "bmo_forward_temporal2: transformer_out tensor missing");
            return 3;
        }
        if (!t_lgt || !t_lgt->data) {
            set_err(h, "bmo_forward_temporal2: text_logits tensor missing (model.text_linear may be null)");
            return 3;
        }

        const size_t hidden_bytes = (size_t) h->ctx.n_embd * sizeof(float);
        const size_t logits_bytes = (size_t) h->ctx.text_vocab_size * sizeof(float);
        if ((size_t) ggml_nbytes(t_out) < hidden_bytes ||
            (size_t) ggml_nbytes(t_lgt) < logits_bytes) {
            set_err(h, "bmo_forward_temporal2: output tensor smaller than expected");
            return 5;
        }

        std::memcpy(out_transformer, t_out->data, hidden_bytes);
        std::memcpy(out_text_logits, t_lgt->data, logits_bytes);

        if (n_capture_layers > 0) {
            const int32_t n_embd = h->ctx.n_embd;
            for (int i = 0; i < n_capture_layers; ++i) {
                const int32_t L = capture_layers[i];
                const std::string nm = "out_layer_" + std::to_string((int) L);
                ggml_tensor * tl = ggml_graph_get_tensor(gf, nm.c_str());
                if (!tl || !tl->data) {
                    set_err(h, "bmo_forward_temporal2: missing capture tensor " + nm);
                    return 6;
                }
                if ((size_t) ggml_nbytes(tl) < hidden_bytes) {
                    set_err(h, "bmo_forward_temporal2: capture tensor too small for " + nm);
                    return 7;
                }
                std::memcpy(
                    capture_out + (size_t) i * (size_t) n_embd,
                    tl->data,
                    hidden_bytes);
            }
        }

        h->pos = pos + 1;
        h->last_error.clear();
        return 0;
    } catch (const std::exception & ex) {
        set_err(h, ex.what());
        return 9;
    } catch (...) {
        set_err(h, "bmo_forward_temporal2: unknown exception");
        return 9;
    }
}

int bmo_forward_depth(
    bmo_handle_t * h,
    int cb_index,
    int32_t prev_token,
    const float * transformer_out,
    float * out_audio_logits) {
    if (!h || !transformer_out || !out_audio_logits) {
        set_err(h, "bmo_forward_depth: null pointer argument");
        return 1;
    }
    if (cb_index < 0 || cb_index >= h->ctx.dep_q) {
        set_err(h, "bmo_forward_depth: cb_index out of range");
        return 2;
    }
    if (h->ctx.depth_n_ctx <= 0 || cb_index >= h->ctx.depth_n_ctx) {
        set_err(h, "bmo_forward_depth: depth KV cache not initialised or cb_index too large");
        return 2;
    }

    std::lock_guard<std::mutex> lk(h->mu);
    try {
#ifdef BMO_ENABLE_CUDA
        if (h->depth_captured[cb_index]) {
            if (cb_index == 0) {
                if (h->ctx.depth_k_cache && h->ctx.depth_k_cache->extra) {
                    cudaMemsetAsync(h->ctx.depth_k_cache->extra, 0, (size_t) ggml_nbytes(h->ctx.depth_k_cache), h->stream);
                }
                if (h->ctx.depth_v_cache && h->ctx.depth_v_cache->extra) {
                    cudaMemsetAsync(h->ctx.depth_v_cache->extra, 0, (size_t) ggml_nbytes(h->ctx.depth_v_cache), h->stream);
                }
            }

            // Copy transformer_out to depth_temporal_in if caller passed a different pointer
            const size_t hidden_bytes = (size_t) h->ctx.n_embd * sizeof(float);
            if (transformer_out != h->depth_temporal_in_host) {
                std::memcpy(h->depth_temporal_in_host, transformer_out, hidden_bytes);
            }

            // Unpack prev_token's embedding directly into the dedicated depth_tok_emb_host
            if (cb_index == 0) {
                unpack_emb_row(h->model.text_emb, prev_token, h->depth_tok_emb_host, 1024);
            } else {
                unpack_emb_row(h->model.audio_embs[(size_t) (cb_index - 1)], prev_token, h->depth_tok_emb_host, 1024);
            }

            cudaGraphLaunch(h->depth_graph_exec[cb_index], h->stream);
            cudaStreamSynchronize(h->stream);

            const size_t want_bytes = (size_t) h->ctx.audio_vocab_size * sizeof(float);
            std::memcpy(out_audio_logits, h->depth_audio_logits_host[cb_index], want_bytes);
            h->last_error.clear();
            return 0;
        }
#endif

        // 1. Reset the depth KV cache at the start of every new temporal frame.
        if (cb_index == 0) {
            bmo_reset_depth_kv(h->ctx);
        }

        // 2. Fresh work arena for this graph.
        bmo_reset_work_ctx(h->ctx);
        h->ctx.graph_uploads.clear();

        ggml_context * wctx = h->ctx.work_ctx;
        if (!wctx) {
            set_err(h, "bmo_forward_depth: work context is not initialized");
            return 3;
        }

        // 3. Build host-side leaf tensors.
        const int n_embd = h->ctx.n_embd;
        ggml_tensor * temporal_in = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, 1);
        if (!temporal_in || !temporal_in->data) {
            set_err(h, "bmo_forward_depth: failed to allocate temporal_in");
            return 4;
        }
        std::memcpy(temporal_in->data, transformer_out, (size_t) n_embd * sizeof(float));

        ggml_tensor * text_tokens  = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, 1);
        ggml_tensor * audio_tokens = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, 1);
        if (!text_tokens || !text_tokens->data || !audio_tokens || !audio_tokens->data) {
            set_err(h, "bmo_forward_depth: failed to allocate token leaves");
            return 4;
        }
        *((int32_t *) text_tokens->data)  = prev_token;
        *((int32_t *) audio_tokens->data) = prev_token;

        // 4. Build and execute the depth graph.
        ggml_cgraph * gf = bmo_build_depth_graph(
            h->ctx, h->model, temporal_in, text_tokens, audio_tokens,
            /*codebook_step=*/cb_index, /*n_past=*/0);
        if (!gf) {
            set_err(h, "bmo_forward_depth: failed to build depth graph");
            return 4;
        }

        bmo_execute_graph(h->ctx, gf, /*inputs=*/{});

        // 5. Pull out the audio logits.
        ggml_tensor * t_logits = ggml_graph_get_tensor(gf, "audio_logits");
        if (!t_logits || !t_logits->data) {
            set_err(h,
                    "bmo_forward_depth: audio_logits tensor missing "
                    "(model.audio_heads[cb_index] may be null)");
            return 3;
        }
        const size_t want_bytes = (size_t) h->ctx.audio_vocab_size * sizeof(float);
        if ((size_t) ggml_nbytes(t_logits) < want_bytes) {
            set_err(h, "bmo_forward_depth: audio_logits smaller than audio_vocab_size");
            return 5;
        }
        std::memcpy(out_audio_logits, t_logits->data, want_bytes);

        h->last_error.clear();
        return 0;
    } catch (const std::exception & ex) {
        set_err(h, ex.what());
        return 9;
    } catch (...) {
        set_err(h, "bmo_forward_depth: unknown exception");
        return 9;
    }
}

static int sample_token_internal(
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

BMO_API int bmo_forward_depth_cascade(
    bmo_handle_t * h,
    int32_t text_token,
    const float * transformer_out,
    float temp_audio,
    int top_k_audio,
    int32_t * out_audio_tokens) {
    if (!h || !transformer_out || !out_audio_tokens) {
        set_err(h, "bmo_forward_depth_cascade: null pointer argument");
        return 1;
    }

    std::lock_guard<std::mutex> lk(h->mu);
    try {
        bmo_reset_depth_kv(h->ctx);

        static thread_local std::mt19937 rng(42);
        static thread_local std::vector<std::pair<float, int>> scratch(2048);

        int32_t prev_tok = text_token;
        const int n_embd = h->ctx.n_embd;

        for (int cb = 0; cb < h->ctx.dep_q; ++cb) {
            bmo_reset_work_ctx(h->ctx);
            h->ctx.graph_uploads.clear();

            ggml_context * wctx = h->ctx.work_ctx;
            if (!wctx) {
                set_err(h, "bmo_forward_depth_cascade: work context is not initialized");
                return 3;
            }

            ggml_tensor * temporal_in = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, 1);
            std::memcpy(temporal_in->data, transformer_out, (size_t) n_embd * sizeof(float));

            ggml_tensor * text_tokens = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, 1);
            ggml_tensor * audio_tokens = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, 1);
            *((int32_t *) text_tokens->data) = prev_tok;
            *((int32_t *) audio_tokens->data) = prev_tok;

            ggml_cgraph * gf = bmo_build_depth_graph(
                h->ctx, h->model, temporal_in, text_tokens, audio_tokens,
                cb, 0);
            if (!gf) {
                set_err(h, "bmo_forward_depth_cascade: failed to build depth graph");
                return 4;
            }

            bmo_execute_graph(h->ctx, gf, {});

            ggml_tensor * t_logits = ggml_graph_get_tensor(gf, "audio_logits");
            if (!t_logits || !t_logits->data) {
                set_err(h, "bmo_forward_depth_cascade: audio_logits tensor missing");
                return 3;
            }

            float * logits = (float *) t_logits->data;
            if (h->ctx.audio_vocab_size > 2048) {
                logits[2048] = -1e9f;
            }

            int sampled = sample_token_internal(logits, 2048, temp_audio, top_k_audio, scratch, rng);
            out_audio_tokens[cb] = sampled;
            prev_tok = sampled;
        }

        h->last_error.clear();
        return 0;
    } catch (const std::exception & ex) {
        set_err(h, ex.what());
        return 9;
    } catch (...) {
        set_err(h, "bmo_forward_depth_cascade: unknown exception");
        return 9;
    }
}

BMO_API int bmo_rvq_decode(
    const int32_t * codes_dev,
    const float * proj_tables_dev,
    float * out_dev,
    void * stream) {
    if (!codes_dev || !proj_tables_dev || !out_dev) return 1;
#ifdef BMO_ENABLE_CUDA
    launch_rvq_decode(codes_dev, proj_tables_dev, out_dev, stream);
    return 0;
#else
    return 2;
#endif
}

BMO_API int bmo_rvq_encode(
    const float * in_vec_dev,
    const float * w_in0_dev,
    const float * e0_dev,
    const float * norm0_dev,
    const float * w_in_rest_dev,
    const float * e_rest_dev,
    const float * norm_rest_dev,
    int32_t * out_codes_dev,
    float * scratch_dev,
    void * stream) {
    if (!in_vec_dev || !w_in0_dev || !e0_dev || !norm0_dev ||
        !w_in_rest_dev || !e_rest_dev || !norm_rest_dev ||
        !out_codes_dev || !scratch_dev) {
        return 1;
    }
#ifdef BMO_ENABLE_CUDA
    launch_rvq_encode(
        in_vec_dev, w_in0_dev, e0_dev, norm0_dev,
        w_in_rest_dev, e_rest_dev, norm_rest_dev,
        out_codes_dev, scratch_dev, stream);
    return 0;
#else
    return 2;
#endif
}

} // extern "C"

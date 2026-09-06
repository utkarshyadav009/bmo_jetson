// bmo_api.cpp - Implementation of the BMO C-ABI bridge.
//
// Wraps bmo_model + bmo_context behind an opaque handle and serialises all
// per-handle entry points through a mutex so the resulting libbmo.so can be
// safely shared across Python threads.

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

struct bmo_handle {
    bmo_model   model;
    bmo_context ctx;
    int         pos = 0;
    std::string last_error;
    std::mutex  mu;
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
        } catch (...) {
            // best-effort cleanup; never throw from the failure path
        }
    };

    try {
        bmo_load_model(gguf_path, h->model, h->ctx);

        // bmo_init_kv_cache takes (ctx, n_ctx) -- the n_ctx caller value is
        // clamped internally on Jetson.
        const int32_t kv_ctx = (n_ctx > 0) ? (int32_t) n_ctx : kDefaultKvCtx;
        bmo_init_kv_cache(h->ctx, kv_ctx);

        h->ctx.work_mem.resize(kDefaultWorkMem);
        h->pos = 0;
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

        // 3. Build host-side leaf tensors. bmo_build_depth_graph is happy to
        //    consume tensors that already have ->data populated; we just need
        //    them to live in the work arena so the graph executor finds them.
        const int n_embd = h->ctx.n_embd;
        ggml_tensor * temporal_in = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, 1);
        if (!temporal_in || !temporal_in->data) {
            set_err(h, "bmo_forward_depth: failed to allocate temporal_in");
            return 4;
        }
        std::memcpy(temporal_in->data, transformer_out, (size_t) n_embd * sizeof(float));

        // text_tokens / audio_tokens are both single-element I32 leaves so
        // bmo_build_depth_graph's per-codebook embedding lookup
        //   (cb_index == 0 -> text_emb[prev_token],
        //    cb_index >  0 -> audio_embs[cb_index - 1][prev_token])
        // can read whichever one matches the current step. We populate both
        // with prev_token; only one is dereferenced per call.
        ggml_tensor * text_tokens  = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, 1);
        ggml_tensor * audio_tokens = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, 1);
        if (!text_tokens || !text_tokens->data || !audio_tokens || !audio_tokens->data) {
            set_err(h, "bmo_forward_depth: failed to allocate token leaves");
            return 4;
        }
        *((int32_t *) text_tokens->data)  = prev_token;
        *((int32_t *) audio_tokens->data) = prev_token;

        // 4. Build and execute the depth graph. n_past is unused inside the
        //    builder (depth attention indexes by codebook_step), so we pass 0.
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

} // extern "C"

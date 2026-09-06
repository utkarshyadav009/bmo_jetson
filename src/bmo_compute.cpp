// Jetson fused linear passes per-row tier bases into fused matvec kernel.
#include "bmo.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
static inline struct ggml_tensor * bmo_safe(struct ggml_tensor * t, const char * msg, int line) {
    if (!t) { fprintf(stderr, "\n[CRITICAL] Tensor '%s' is NULL at line %d!\n", msg, line); exit(1); }
    return t;
}
#define S(x) bmo_safe((x), #x, __LINE__)


#include <algorithm>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef BMO_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef BMO_JETSON
#include <sys/mman.h>
#endif

enum class bmo_dump_kind : uint8_t {
    Layer0, // BMO_DUMP_LAYER0
    Deep,   // BMO_DUMP_DEEP (mid-stack residual, post_out_norm, final_logits)
};

// Numerical harness (opt-in): float32 little-endian dumps when BMO_DUMP_LAYER0 or
// BMO_DUMP_DEEP is set according to `kind`. No effect otherwise.
static void bmo_dump_tensor_f32(struct ggml_tensor * t, const char * name, bmo_dump_kind kind = bmo_dump_kind::Layer0) {
    const bool want_layer0 = getenv("BMO_DUMP_LAYER0") != nullptr;
    const bool want_deep   = getenv("BMO_DUMP_DEEP") != nullptr;
    const bool ok =
        (kind == bmo_dump_kind::Layer0 && want_layer0) || (kind == bmo_dump_kind::Deep && want_deep);
    if (!ok || !name || !t || ggml_nelements(t) <= 0 || !t->data) {
        return;
    }
#ifdef BMO_ENABLE_CUDA
    cudaStreamSynchronize(0);
#endif
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr,
                "[bmo_dump_tensor] skip %s: expected F32, got type=%s\n",
                name,
                ggml_type_name(t->type));
        return;
    }
    const std::string fname = std::string("cpp_") + name + ".bin";
    FILE * f = fopen(fname.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[bmo_dump_tensor] fopen failed for %s\n", fname.c_str());
        return;
    }
    const size_t n = (size_t) ggml_nelements(t);
    const size_t nw = fwrite(t->data, sizeof(float), n, f);
    fclose(f);
    if (nw != n) {
        fprintf(stderr, "[bmo_dump_tensor] short write %s (%zu/%zu floats)\n", fname.c_str(), nw, n);
    }
}

// Path B Diagnostic 2: optional first-8 + L2 taps. Enable with: BMO_H3_TAPS=1
//  - T1,T3,T4,T5,T6: L in {0, 1, 4, 8}.
//  - T2 (residual_out): L in {0,1,4,8,16,24,31}.
//  - T7 (post_out_norm), T8 (final_logits): head path, no layer index.
#ifdef BMO_JETSON
static bool bmo_h3_in_t1_t6_layer_set(int layer) {
    static const int k_t1_t6_layers[] = {0, 1, 4, 8};
    for (int tl : k_t1_t6_layers) {
        if (tl == layer) {
            return true;
        }
    }
    return false;
}

static bool bmo_h3_tap_allow_layer_for_tag(const char * tap_id, int layer) {
    if (std::strcmp(tap_id, "T2") == 0) {
        static const int k_t2_layers[] = {0, 1, 4, 8, 16, 24, 31};
        for (int tl : k_t2_layers) {
            if (tl == layer) {
                return true;
            }
        }
        return false;
    }
    return bmo_h3_in_t1_t6_layer_set(layer);
}

// Full float32 residual row at T2 (device-side tap), for offline cos vs PT.
// Set BMO_H3_DUMP_RESIDUAL_BINS=1; optional BMO_H3_RESIDUAL_BIN_DIR=/abs/path (default cwd).
static void bmo_h3_tap_maybe_dump_t2_residual_bin(int layer, const float * p, int64_t n) {
    if (!getenv("BMO_H3_DUMP_RESIDUAL_BINS")) {
        return;
    }
    if (!bmo_h3_tap_allow_layer_for_tag("T2", layer)) {
        return;
    }
    char path[1024];
    int nc;
    const char * dir = getenv("BMO_H3_RESIDUAL_BIN_DIR");
    if (dir && dir[0]) {
        nc = snprintf(path, sizeof path, "%s/cpp_residual_L%d.bin", dir, layer);
    } else {
        nc = snprintf(path, sizeof path, "cpp_residual_L%d.bin", layer);
    }
    if (nc < 0 || (size_t) nc >= sizeof path) {
        fprintf(stderr, "[h3_dump] path too long or snprintf error (layer %d)\n", layer);
        return;
    }
    FILE * fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[h3_dump] fopen failed %s\n", path);
        return;
    }
    const size_t nw = fwrite(p, sizeof(float), (size_t) n, fp);
    fclose(fp);
    if (nw != (size_t) n) {
        fprintf(stderr, "[h3_dump] short write %s (%zu/%lld floats)\n", path, nw, (long long) n);
    }
}

// Optional full-vector dumps for offline cos vs PT (BMO_H3_DUMP_TAP_BINS=1).
// Directory: BMO_H3_TAP_BIN_DIR, else BMO_H3_RESIDUAL_BIN_DIR, else cwd.
// Skips T2 (use cpp_residual_L{L}.bin from BMO_H3_DUMP_RESIDUAL_BINS).
static void bmo_h3_tap_maybe_dump_tap_bin(const char * tap_id, int layer, const float * p, int64_t n) {
    if (!getenv("BMO_H3_DUMP_TAP_BINS")) {
        return;
    }
    if (std::strcmp(tap_id, "T2") == 0) {
        return;
    }
    if (!bmo_h3_in_t1_t6_layer_set(layer)) {
        return;
    }
    const char * dir = getenv("BMO_H3_TAP_BIN_DIR");
    if (!dir || !dir[0]) {
        dir = getenv("BMO_H3_RESIDUAL_BIN_DIR");
    }
    char path[1024];
    int nc;
    if (dir && dir[0]) {
        nc = snprintf(path, sizeof path, "%s/cpp_h3_%s_L%d.bin", dir, tap_id, layer);
    } else {
        nc = snprintf(path, sizeof path, "cpp_h3_%s_L%d.bin", tap_id, layer);
    }
    if (nc < 0 || (size_t) nc >= sizeof path) {
        fprintf(stderr, "[h3_tap_dump] path error (layer %d %s)\n", layer, tap_id);
        return;
    }
    FILE * fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[h3_tap_dump] fopen failed %s\n", path);
        return;
    }
    const size_t nw = fwrite(p, sizeof(float), (size_t) n, fp);
    fclose(fp);
    if (nw != (size_t) n) {
        fprintf(stderr, "[h3_tap_dump] short write %s (%zu/%lld)\n", path, nw, (long long) n);
    }
}

static void bmo_h3_tap(int layer, const char * tap_id, ggml_tensor * t) {
    static const bool s_enabled = (getenv("BMO_H3_TAPS") != nullptr) ||
                                  (getenv("BMO_H3_DUMP_RESIDUAL_BINS") != nullptr) ||
                                  (getenv("BMO_H3_DUMP_TAP_BINS") != nullptr);
    if (!s_enabled) {
        return;
    }
    const bool want_print = getenv("BMO_H3_TAPS") != nullptr;
    const bool want_t2_bin =
        getenv("BMO_H3_DUMP_RESIDUAL_BINS") != nullptr && std::strcmp(tap_id, "T2") == 0 &&
        bmo_h3_tap_allow_layer_for_tag("T2", layer);
    const bool want_tap_bin = getenv("BMO_H3_DUMP_TAP_BINS") != nullptr;
    const bool tap_bin_eligible =
        want_tap_bin && std::strcmp(tap_id, "T2") != 0 && bmo_h3_in_t1_t6_layer_set(layer);
    if (!want_print && !want_t2_bin && !tap_bin_eligible) {
        return;
    }
    if (want_print && !bmo_h3_tap_allow_layer_for_tag(tap_id, layer)) {
        return;
    }
    if (!t || t->type != GGML_TYPE_F32 || !t->data) {
        if (want_print) {
            fprintf(stderr, "[h3_tap_%s] L=%d <null or non-f32>\n", tap_id, layer);
        }
        return;
    }
    cudaStreamSynchronize(0);
    (void) cudaGetLastError();
    const int64_t n = ggml_nelements(t);
    const float * p = (const float *) t->data;
    if (want_t2_bin) {
        bmo_h3_tap_maybe_dump_t2_residual_bin(layer, p, n);
    }
    bmo_h3_tap_maybe_dump_tap_bin(tap_id, layer, p, n);
    if (!want_print) {
        return;
    }
    double s2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        s2 += (double) p[i] * (double) p[i];
    }
    const float l2 = (float) std::sqrt(s2);
    const int k = (int) std::min<int64_t>(8, n);
    fprintf(stderr, "[h3_tap_%s] L=%d n=%lld l2=%.6f first8=", tap_id, layer, (long long) n, l2);
    for (int i = 0; i < k; ++i) {
        fprintf(stderr, "%s%.6f", (i > 0) ? "," : "", p[i]);
    }
    fprintf(stderr, "\n");
}

// e.g. [h3_tap_T7] post_out_norm n=... l2=... first8=...
static void bmo_h3_tap_head(const char * tag, const char * sublabel, ggml_tensor * t) {
    if (!getenv("BMO_H3_TAPS")) {
        return;
    }
    if (!t || t->type != GGML_TYPE_F32 || !t->data) {
        fprintf(stderr, "[h3_tap_%s] %s <null or non-f32>\n", tag, sublabel);
        return;
    }
    cudaStreamSynchronize(0);
    (void) cudaGetLastError();
    const int64_t n = ggml_nelements(t);
    const float * p = (const float *) t->data;
    double s2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        s2 += (double) p[i] * (double) p[i];
    }
    const float l2 = (float) std::sqrt(s2);
    const int k = (int) std::min<int64_t>(8, n);
    fprintf(stderr, "[h3_tap_%s] %s n=%lld l2=%.6f first8=", tag, sublabel, (long long) n, l2);
    for (int i = 0; i < k; ++i) {
        fprintf(stderr, "%s%.6f", (i > 0) ? "," : "", p[i]);
    }
    fprintf(stderr, "\n");
}
#else
static inline void bmo_h3_tap(int /*layer*/, const char * /*tap_id*/, ggml_tensor * /*t*/) {}
static inline void bmo_h3_tap_head(const char * /*tag*/, const char * /*sublabel*/, ggml_tensor * /*t*/) {}
#endif

#ifdef BMO_ENABLE_CUDA
#ifndef BMO_JETSON
void launch_unpack_kernel(
    const device_packed_t * dp,
    int32_t rows,
    int32_t cols,
    int32_t n_2bit_bytes,
    int32_t n_4bit_bytes,
    int32_t n_8bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    float * out_w);
#endif
#endif

#ifdef BMO_JETSON
#include "bmo_proto_kernels.h"

// ---------- GPU staging pool: borrow/release ----------

struct staging_slot {
    void *              host = nullptr;
    void *              dev  = nullptr;
    int                 idx  = -1;
    gpu_staging_pool *  pool = nullptr;
};

// Path B: packing_version >= 5 uses per-element mask + proto kernel; v4 uses v2 + shared-memory prefix.
static void launch_fused_dequant_matvec_jetson(
    const device_packed_t & dp,
    const char *            tensor_base,
    const void *            kern_pw,
    const void *            kern_pm,
    const void *            kern_fv,
    int                       rows,
    int                       cols,
    int                       block_size_eff,
    const float *           x_dev,
    float *                 y_dev,
    void *                  stream) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    // H2 diagnostic: log every fused dequant->matvec dispatch (register pointers are expected to be distinct per-layer).
    if (0) std::fprintf(stderr,
                 "[h2_diag_matvec] tensor=%s pv=%d valid=%d "
                 "in_dev=%p out_dev=%p "
                 "row_c2=%p row_c4=%p row_c8=%p row_c16=%p "
                 "canonical_int2=%p canonical_fp16=%p\n",
                 tensor_base ? tensor_base : "(null)",
                 (int) dp.packing_version,
                 (int) dp.is_valid,
                 (void *) x_dev,
                 (void *) y_dev,
                 (void *) dp.row_c2,
                 (void *) dp.row_c4,
                 (void *) dp.row_c8,
                 (void *) dp.row_c16,
                 (void *) dp.canonical_pw_dev,
                 (void *) dp.canonical_fv_dev);

    if (dp.packing_version >= 5) {
        if (!dp.row_c16) {
            throw std::runtime_error("packing_version>=5 requires row_c16");
        }
        launch_fused_dequant_matvec_proto(
            kern_pw,
            kern_pm,
            kern_fv,
            dp.row_c2,
            dp.row_c4,
            dp.row_c8,
            dp.row_c16,
            rows,
            cols,
            block_size_eff,
            dp.n_2bit_bytes,
            dp.n_4bit_bytes,
            dp.scale_low,
            dp.scale_int4,
            dp.scale_int8,
            dp.zp_low,
            dp.zp_int4,
            dp.zp_int8,
            x_dev,
            y_dev,
            8,
            s);
    } else {
        launch_fused_dequant_matvec(
            kern_pw,
            kern_pm,
            kern_fv,
            dp.row_c2,
            dp.row_c4,
            dp.row_c8,
            rows,
            cols,
            block_size_eff,
            dp.n_2bit_bytes,
            dp.n_4bit_bytes,
            dp.scale_low,
            dp.scale_int4,
            dp.scale_int8,
            dp.zp_low,
            dp.zp_int4,
            dp.zp_int8,
            x_dev,
            y_dev,
            stream);
    }
}

static int g_staging_next_slot = 0;
static staging_slot borrow_staging(gpu_staging_pool & p) {
    for (int count = 0; count < gpu_staging_pool::N_SLOTS; ++count) {
        int i = (g_staging_next_slot + count) % gpu_staging_pool::N_SLOTS;
        if (!p.in_use[i] && p.host[i] && p.dev[i]) {
            p.in_use[i] = true;
            g_staging_next_slot = (i + 1) % gpu_staging_pool::N_SLOTS;
            return staging_slot { p.host[i], p.dev[i], i, &p };
        }
    }
    throw std::runtime_error("Staging pool exhausted");
}

static void release_all_staging(gpu_staging_pool & p) {
    for (int i = 0; i < gpu_staging_pool::N_SLOTS; ++i) {
        p.in_use[i] = false;
    }
    g_staging_next_slot = 0;
}

// Eagerly executes a fused RMSNorm + element-wise weight multiply on the GPU
// and returns a leaf ggml_tensor whose data field already holds the result.
// Borrows up to two slots from the pool: one to stage the activation (only
// if x->data is not directly mappable) and one for the kernel output. The
// output slot is intentionally kept borrowed until release_all_staging() so
// downstream interceptors in the same layer can re-map y->data via
// cudaHostGetDevicePointer instead of a memcpy.
static ggml_tensor * apply_rmsnorm_gpu(
    bmo_context & ctx,
    ggml_context * wctx,
    ggml_tensor * x,
    ggml_tensor * weight,
    float eps = 1e-8f) {
    if (!x || !weight) {
        throw std::runtime_error("apply_rmsnorm_gpu: null input/weight tensor");
    }
    if (x->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32) {
        throw std::runtime_error("apply_rmsnorm_gpu: requires GGML_TYPE_F32 input and weight");
    }

    const int n_embd = (int) x->ne[0];
    const size_t row_bytes = (size_t) n_embd * sizeof(float);
    if (row_bytes > gpu_staging_pool::SLOT_BYTES) {
        throw std::runtime_error("apply_rmsnorm_gpu: n_embd exceeds staging slot capacity");
    }

    const int64_t n_el = ggml_nelements(x);
    if (n_el <= 0 || (n_el % (int64_t) n_embd) != 0) {
        throw std::runtime_error("apply_rmsnorm_gpu: invalid input shape");
    }
    const int64_t n_tok = n_el / (int64_t) n_embd;

    float * w_dev = (float *) weight->extra;
    if (!w_dev) {
        if (cudaHostGetDevicePointer((void **) &w_dev, weight->data, 0) != cudaSuccess) {
            throw std::runtime_error("apply_rmsnorm_gpu: cudaHostGetDevicePointer(weight) failed");
        }
        weight->extra = w_dev;
    }

    // Resolve / stage the input. If x->extra or x->data is already pinned/mapped, skip memcpy.
    float * x_base_dev = (float *) x->extra;
    bool x_mapped = (x_base_dev != nullptr);
    if (!x_mapped) {
        x_mapped = (cudaHostGetDevicePointer((void **) &x_base_dev, x->data, 0) == cudaSuccess);
    }
    staging_slot in_slot {};
    if (!x_mapped) {
        cudaGetLastError(); // flush stale error from the failed lookup
        in_slot = borrow_staging(ctx.staging);
    }

    staging_slot out_slot = borrow_staging(ctx.staging);
    float * y_dev_slot = (float *) out_slot.dev;
    float * y_host_slot = (float *) out_slot.host;

    ggml_tensor * y = (n_tok == 1)
        ? ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_embd)
        : ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, n_tok);

    for (int64_t t = 0; t < n_tok; ++t) {
        const uint8_t * x_col_host = (const uint8_t *) x->data + (size_t) t * x->nb[1];
        const float * x_col_dev;
        if (x_mapped) {
            x_col_dev = reinterpret_cast<const float *>(
                (const uint8_t *) x_base_dev + (size_t) t * x->nb[1]);
        } else {
            std::memcpy(in_slot.host, x_col_host, row_bytes);
            x_col_dev = (const float *) in_slot.dev;
        }

        (void) cudaGetLastError();

        launch_rmsnorm(x_col_dev, w_dev, eps, n_embd, y_dev_slot, ctx.stream);

        cudaError_t rms_err = cudaGetLastError();
        if (rms_err != cudaSuccess) {
            fprintf(stderr,
                    "[apply_rmsnorm_gpu] CUDA error after launch_rmsnorm: %s | grid=(1,1,1) block=(256,1,1) "
                    "n_embd=%d n_tok=%lld stream=default x_dev=%p w_dev=%p y_dev=%p\n",
                    cudaGetErrorString(rms_err),
                    n_embd,
                    (long long) n_tok,
                    (const void *) x_col_dev,
                    (const void *) w_dev,
                    (void *) y_dev_slot);
            throw std::runtime_error("apply_rmsnorm_gpu: kernel launch error");
        }

        if (n_tok == 1) {
            y->data = y_host_slot;
            y->extra = y_dev_slot;
        } else {
            std::memcpy((uint8_t *) y->data + (size_t) t * y->nb[1], y_host_slot, row_bytes);
        }
    }

    return y;
}

// Eagerly computes y = a + b on the GPU. For single-token inputs the result
// stays in a borrowed staging slot so subsequent interceptors can map it
// directly. For larger inputs we fall back to ggml_add so the graph still
// builds correctly.
static ggml_tensor * apply_residual_gpu(
    bmo_context & ctx,
    ggml_context * wctx,
    ggml_tensor * a,
    ggml_tensor * b) {
    if (!a || !b) {
        throw std::runtime_error("apply_residual_gpu: null input tensor");
    }
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return ggml_add(wctx, a, b);
    }

    const int64_t n_el = ggml_nelements(a);
    if (n_el != ggml_nelements(b) || n_el <= 0) {
        return ggml_add(wctx, a, b);
    }
    const size_t total_bytes = (size_t) n_el * sizeof(float);
    if (total_bytes > gpu_staging_pool::SLOT_BYTES) {
        // Larger than a single slot (multi-token prefill): defer to the graph.
        return ggml_add(wctx, a, b);
    }

    const int n = (int) n_el;

    float * a_dev = (float *) a->extra;
    staging_slot stage_a {};
    if (!a_dev) {
        if (cudaHostGetDevicePointer((void **) &a_dev, a->data, 0) == cudaSuccess) {
            a->extra = a_dev;
        } else {
            cudaGetLastError(); // flush stale error from the failed lookup
            stage_a = borrow_staging(ctx.staging);
            std::memcpy(stage_a.host, a->data, total_bytes);
            a_dev = (float *) stage_a.dev;
        }
    }

    float * b_dev = (float *) b->extra;
    staging_slot stage_b {};
    if (!b_dev) {
        if (cudaHostGetDevicePointer((void **) &b_dev, b->data, 0) == cudaSuccess) {
            b->extra = b_dev;
        } else {
            cudaGetLastError(); // flush stale error from the failed lookup
            stage_b = borrow_staging(ctx.staging);
            std::memcpy(stage_b.host, b->data, total_bytes);
            b_dev = (float *) stage_b.dev;
        }
    }

    staging_slot out = borrow_staging(ctx.staging);
    launch_residual_add(a_dev, b_dev, n, (float *) out.dev, ctx.stream);

    // Deferred sync: see apply_rmsnorm_gpu for rationale.
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("apply_residual_gpu: kernel launch error");
    }

    ggml_tensor * y = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n);
    y->data = out.host;
    y->extra = out.dev;
    return y;
}

// Eagerly applies interleaved RoPE on the GPU and returns a 3D leaf tensor
// whose data field already holds the rotated activations. Falls back to
// ggml_rope for the multi-token (prefill) case where the input view contains
// strided gaps that the contiguous kernel cannot consume.
static ggml_tensor * apply_rope_gpu_interleaved(
    bmo_context & ctx,
    ggml_context * wctx,
    ggml_tensor * x,
    ggml_tensor * pos) {
    if (!x || !pos) {
        throw std::runtime_error("apply_rope_gpu_interleaved: null tensor argument");
    }
    if (x->type != GGML_TYPE_F32 || pos->type != GGML_TYPE_I32) {
        return ggml_rope(wctx, x, pos, (int) x->ne[0], GGML_ROPE_TYPE_NORMAL);
    }

    const int head_dim = (int) x->ne[0];
    const int n_heads = (int) x->ne[1];
    const int n_token = (int) x->ne[2];

    // Multi-token views into a packed QKV tensor are strided across the
    // token axis (ne[2] step crosses K/V slots). The contiguous kernel cannot
    // consume those gaps, so defer to ggml_rope for prefill.
    if (n_token != 1) {
        return ggml_rope(wctx, x, pos, head_dim, GGML_ROPE_TYPE_NORMAL);
    }
    if ((head_dim & 1) != 0) {
        throw std::runtime_error("apply_rope_gpu_interleaved: head_dim must be even");
    }

    const size_t row_bytes = (size_t) head_dim * (size_t) n_heads * sizeof(float);
    if (row_bytes > gpu_staging_pool::SLOT_BYTES) {
        return ggml_rope(wctx, x, pos, head_dim, GGML_ROPE_TYPE_NORMAL);
    }
    if (!pos->data || ggml_nbytes(pos) < (int64_t) sizeof(int32_t)) {
        throw std::runtime_error("apply_rope_gpu_interleaved: invalid pos tensor");
    }
    const int pos_base = ((const int32_t *) pos->data)[0];

    float * x_dev = (float *) x->extra;
    staging_slot in_slot {};
    if (!x_dev) {
        if (cudaHostGetDevicePointer((void **) &x_dev, x->data, 0) != cudaSuccess) {
            cudaGetLastError(); // flush stale error from the failed lookup
            in_slot = borrow_staging(ctx.staging);
            std::memcpy(in_slot.host, x->data, row_bytes);
            x_dev = (float *) in_slot.dev;
        }
    }

    staging_slot out = borrow_staging(ctx.staging);
    launch_rope_interleaved(
        x_dev, n_heads, head_dim, n_token, pos_base, ctx.rope_theta,
        (float *) out.dev, ctx.stream, (const int *) ctx.pos_dev);

    // Deferred sync: see apply_rmsnorm_gpu for rationale.
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("apply_rope_gpu_interleaved: kernel launch error");
    }

    ggml_tensor * y = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, head_dim, n_heads, n_token);
    y->data = out.host;
    y->extra = out.dev;
    out.pool = nullptr; // detach: ownership transfers to release_all_staging at layer end
    return y;
}

// Eagerly executes the SwiGLU split-and-fuse on the GPU:
//   y[i] = silu(h[i]) * h[i + d_ff], for i in [0, d_ff)
// where h_full is the concatenated [gate | up] vector (length 2 * d_ff).
// If h_full->data is already a registered host pointer (i.e. lives in a slot
// from a previous GPU op), we map it directly; otherwise we stage a copy.
static ggml_tensor * apply_swiglu_gpu(
    bmo_context & ctx,
    ggml_context * wctx,
    ggml_tensor * h_full) {
    const int total = (int) ggml_nelements(h_full);
    if (total <= 0 || (total % 2) != 0) {
        throw std::runtime_error("apply_swiglu_gpu: invalid h_full size " + std::to_string(total));
    }
    const int d_ff = total / 2;

    const size_t total_bytes = (size_t) total * sizeof(float);
    const size_t d_ff_bytes  = (size_t) d_ff  * sizeof(float);
    if (total_bytes > gpu_staging_pool::SLOT_BYTES ||
        d_ff_bytes  > gpu_staging_pool::SLOT_BYTES) {
        throw std::runtime_error("apply_swiglu_gpu: tensor exceeds staging slot capacity");
    }

    float * h_dev = (float *) h_full->extra;
    if (!h_dev) {
        if (cudaHostGetDevicePointer((void **) &h_dev, h_full->data, 0) == cudaSuccess) {
            h_full->extra = h_dev;
        } else {
            cudaGetLastError(); // flush stale error from the failed lookup
            staging_slot in_slot = borrow_staging(ctx.staging);
            std::memcpy(in_slot.host, h_full->data, total_bytes);
            h_dev = (float *) in_slot.dev;
        }
    }

    staging_slot out_slot = borrow_staging(ctx.staging);
    launch_swiglu_split(h_dev, d_ff, (float *) out_slot.dev, ctx.stream);

    // Deferred sync: see apply_rmsnorm_gpu for rationale.
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("apply_swiglu_gpu: kernel launch error");
    }

    ggml_tensor * y = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, d_ff);
    y->data = out_slot.host;
    y->extra = out_slot.dev;
    out_slot.pool = nullptr; // detach: ownership transfers to release_all_staging at layer end
    return y;
}

#ifdef BMO_JETSON
static ggml_tensor * apply_dense_q4_0_linear_gpu(
    bmo_context & ctx,
    ggml_context * wctx,
    ggml_tensor * W,
    ggml_tensor * x) {

    if (!W || !x) {
        throw std::runtime_error("apply_dense_q4_0_linear_gpu: null W or x");
    }
    if (W->type != GGML_TYPE_Q4_0) {
        throw std::runtime_error("apply_dense_q4_0_linear_gpu: expected Q4_0 weight");
    }
    if (x->type != GGML_TYPE_F32) {
        throw std::runtime_error("apply_dense_q4_0_linear_gpu: expected F32 activation");
    }

    const int cols = (int) W->ne[0];
    const int rows = (int) W->ne[1];
    const int64_t n_el = ggml_nelements(x);
    if (n_el <= 0 || (n_el % cols) != 0) {
        throw std::runtime_error("apply_dense_q4_0_linear_gpu: invalid x shape for cols=" + std::to_string(cols));
    }
    const int64_t n_tok = n_el / cols;
    if (n_tok != 1) {
        throw std::runtime_error("apply_dense_q4_0_linear_gpu: only single-token decode supported");
    }

    const size_t row_out_bytes = (size_t) rows * sizeof(float);
    if (row_out_bytes > gpu_staging_pool::SLOT_BYTES) {
        throw std::runtime_error("apply_dense_q4_0_linear_gpu: output exceeds staging slot capacity (" +
                                 std::to_string(row_out_bytes) + " > " + std::to_string(gpu_staging_pool::SLOT_BYTES) + ")");
    }

    // Resolve device pointer for x
    float * x_dev = (float *) x->extra;
    if (!x_dev) {
        if (cudaHostGetDevicePointer((void **) &x_dev, x->data, 0) == cudaSuccess) {
            x->extra = x_dev;
        } else {
            cudaGetLastError(); // flush error
            const size_t x_bytes = (size_t) cols * sizeof(float);
            if (x_bytes > ctx.cuda_fused_input_buffer_bytes) {
                throw std::runtime_error("apply_dense_q4_0_linear_gpu: cols exceed fused input buffer capacity");
            }
            std::memcpy(ctx.cuda_fused_input_buffer, x->data, x_bytes);
            x_dev = reinterpret_cast<float *>(ctx.cuda_fused_input_buffer_dev);
        }
    }

    // Resolve device pointer for W
    void * W_dev = W->extra;
    if (!W_dev) {
        if (cudaHostGetDevicePointer(&W_dev, W->data, 0) != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error(std::string("apply_dense_q4_0_linear_gpu: cudaHostGetDevicePointer failed for W: ") +
                                     (W->name[0] ? W->name : "unnamed"));
        }
        W->extra = W_dev;
    }

    staging_slot out_slot = borrow_staging(ctx.staging);
    launch_gemv_q4_0_q8_1(
        W_dev,
        x_dev,
        reinterpret_cast<float *>(out_slot.dev),
        rows,
        cols,
        ctx.q8_scratch_dev,
        ctx.stream);

    cudaError_t k_err = cudaGetLastError();
    if (k_err != cudaSuccess) {
        throw std::runtime_error(std::string("apply_dense_q4_0_linear_gpu: launch_gemv_q4_0_q8_1 error: ") +
                                 cudaGetErrorString(k_err));
    }

    ggml_tensor * y = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, rows);
    y->data = out_slot.host;
    y->extra = out_slot.dev;
    out_slot.pool = nullptr; // detach: released at layer end by release_all_staging
    return y;
}
#endif

// Eagerly compute self-attention for the single-token decode path.
//
// Replaces the lazy ggml_cpy + ggml_flash_attn_ext + ggml_permute + ggml_cont
// chain in the temporal graph. That chain was structurally racy on the Jetson
// eager-build pipeline:
//   - q_rope / k_rope live in staging slots that are recycled by subsequent
//     layers' apply_* eager kernels (release_all_staging at the end of each
//     layer marks slots free), so by the time ggml_graph_compute runs the
//     lazy ggml_cpy / flash_attn it reads garbage Q/K/V.
//   - The next eager op (out_proj's apply_linear_with_transient_unpack) reads
//     attn_cont->data *before* ggml_graph_compute populates it, so on call N+1
//     it consumes call N's garbage attention output and propagates NaN.
//
// This routine fixes both by materialising the attention output synchronously
// during graph build:
//   1. Memcpy current-token K, V (FP32 from staging slots) into the FP16 KV
//      cache at position n_past, for every kv head.
//   2. For each Q head, compute scores = Q . K^T / sqrt(d), softmax (with the
//      max-subtract trick for numerical stability), and the weighted V sum
//      across kv_len = n_past + 1 cached keys.
//   3. Write the result into a fresh leaf tensor [head_dim, n_heads, 1]
//      allocated in work_ctx, whose ->data is fully populated before return.
//
// Only handles n_token == 1; multi-token prefill keeps the lazy flash_attn
// path (out_proj is still racy on that path, but the prefill path isn't
// exercised by bmo_inference.py and bmo_main reads out via ggml_graph_compute
// which sees the lazy result).
static ggml_tensor * apply_attention_eager_decode(
    bmo_context & ctx,
    ggml_context * wctx,
    ggml_tensor * q,        // F32 [head_dim, n_heads,    1] in staging slot (post RoPE)
    ggml_tensor * k,        // F32 [head_dim, n_kv_heads, 1] in staging slot (post RoPE)
    ggml_tensor * v,        // F32 [head_dim, n_kv_heads, 1] view into qkv slot
    int n_past,
    int layer) {
    if (!q || !k || !v || !q->data || !k->data || !v->data) {
        throw std::runtime_error("apply_attention_eager_decode: null tensor data");
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32) {
        throw std::runtime_error("apply_attention_eager_decode: requires F32 q/k/v");
    }
    if (!ctx.k_cache || !ctx.v_cache || !ctx.k_cache->data || !ctx.v_cache->data) {
        throw std::runtime_error("apply_attention_eager_decode: KV cache not initialised");
    }
    if (ctx.k_cache->type != GGML_TYPE_F16 || ctx.v_cache->type != GGML_TYPE_F16) {
        throw std::runtime_error("apply_attention_eager_decode: expected FP16 KV cache");
    }

    const int head_dim   = (int) q->ne[0];
    const int n_heads    = (int) q->ne[1];
    const int n_kv_heads = (int) k->ne[1];
    const int n_token    = (int) q->ne[2];
    const int n_ctx      = ctx.n_ctx;
    const int kv_len     = n_past + n_token;

    if (n_token != 1) {
        throw std::runtime_error("apply_attention_eager_decode: only n_token==1 supported");
    }
    if (kv_len > n_ctx) {
        throw std::runtime_error(
            "apply_attention_eager_decode: kv_len " + std::to_string(kv_len)
            + " exceeds n_ctx " + std::to_string(n_ctx));
    }

    // One-shot log so we can confirm from a real run that this helper is live
    // in the deployed libbmo.so (rather than a stale lazy-attention path).
    {
        static std::atomic<bool> logged{false};
        if (getenv("BMO_LOG_ATTN") && !logged.exchange(true)) {
            std::fprintf(stderr,
                         "[bmo_attn_eager] active: head_dim=%d n_heads=%d "
                         "n_kv_heads=%d n_ctx=%d kv_len=%d layer=%d\n",
                         head_dim, n_heads, n_kv_heads, n_ctx, kv_len, layer);
        }
    }
    if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
        throw std::runtime_error(
            "apply_attention_eager_decode: incompatible n_heads="
            + std::to_string(n_heads) + " n_kv_heads=" + std::to_string(n_kv_heads));
    }

    // KV cache layout (ggml ne ordering): [head_dim, n_ctx, n_heads, n_layers].
    // Element [d, p, h, l] sits at element-offset
    //   d + p*head_dim + h*head_dim*n_ctx + l*head_dim*n_ctx*n_heads.
    // Note: ctx.k_cache is allocated with n_heads (full count), so when
    // n_kv_heads < n_heads the trailing head slots are simply unused.
    ggml_fp16_t * const k_cache_data = (ggml_fp16_t *) ctx.k_cache->data;
    ggml_fp16_t * const v_cache_data = (ggml_fp16_t *) ctx.v_cache->data;
    const float * const k_src = (const float *) k->data;
    const float * const v_src = (const float *) v->data;

    const size_t per_layer = (size_t) head_dim * (size_t) n_ctx * (size_t) ctx.n_heads;
    const size_t per_head  = (size_t) head_dim * (size_t) n_ctx;

    // Diagnostic for the KV-not-felt bug: layer 0, head 0, dump the K cache
    // contents at slot 0 (history) and slot n_past (current write target)
    // BEFORE the write. Then dump again AFTER. This tells us whether previous
    // frames' writes survived to the current call.
    const bool kv_log = (getenv("BMO_LOG_KV") != nullptr) && layer == 0;
    auto dump_k = [&](const char * tag, int slot) {
        const size_t off = (size_t) layer * per_layer + 0 * per_head + (size_t) slot * (size_t) head_dim;
        const ggml_fp16_t * p = k_cache_data + off;
        fprintf(stderr,
                "[bmo_kv] %s layer=%d head=0 slot=%d off=%zu k=[%.4f %.4f %.4f %.4f]\n",
                tag, layer, slot, off,
                ggml_fp16_to_fp32(p[0]), ggml_fp16_to_fp32(p[1]),
                ggml_fp16_to_fp32(p[2]), ggml_fp16_to_fp32(p[3]));
    };
    if (kv_log) {
        fprintf(stderr, "[bmo_kv] --- enter layer=%d n_past=%d kv_len=%d ---\n", layer, n_past, kv_len);
        dump_k("BEFORE_WRITE slot0", 0);
        if (n_past > 0) dump_k("BEFORE_WRITE slotN", n_past);
        // also dump the K we are about to write (head 0 of k_src)
        fprintf(stderr,
                "[bmo_kv] k_src[head0,0..4]=[%.4f %.4f %.4f %.4f]\n",
                k_src[0], k_src[1], k_src[2], k_src[3]);
    }

    // Step 1: write current K, V (FP32) into the FP16 cache at position n_past.
    for (int h = 0; h < n_kv_heads; ++h) {
        const size_t cache_offset =
            (size_t) layer * per_layer +
            (size_t) h * per_head +
            (size_t) n_past * (size_t) head_dim;
        const float * k_h = k_src + (size_t) h * (size_t) head_dim;
        const float * v_h = v_src + (size_t) h * (size_t) head_dim;
        for (int d = 0; d < head_dim; ++d) {
            k_cache_data[cache_offset + (size_t) d] = ggml_fp32_to_fp16(k_h[d]);
            v_cache_data[cache_offset + (size_t) d] = ggml_fp32_to_fp16(v_h[d]);
        }
    }

    if (kv_log) {
        dump_k("AFTER_WRITE  slot0", 0);
        if (n_past > 0) dump_k("AFTER_WRITE  slotN", n_past);
    }

    // Step 2: allocate output [head_dim, n_heads, 1]
#ifdef BMO_JETSON
    staging_slot attn_slot = borrow_staging(ctx.staging);
    ggml_tensor * attn_out = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, head_dim, n_heads, 1);
    attn_out->data = attn_slot.host;
    attn_out->extra = attn_slot.dev;
    attn_slot.pool = nullptr;
#else
    ggml_tensor * attn_out = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, head_dim, n_heads, 1);
#endif
    if (!attn_out || !attn_out->data) {
        throw std::runtime_error("apply_attention_eager_decode: failed to allocate attn_out");
    }
    float * const attn_data = (float *) attn_out->data;

    const float * const q_src = (const float *) q->data;
    const float scale = 1.0f / std::sqrt((float) head_dim);
    const int q_per_kv = n_heads / n_kv_heads; // GQA fold: each kv-head serves q_per_kv q-heads

    if (kv_len == 1) {
        for (int h = 0; h < n_heads; ++h) {
            const int kv_h = (n_kv_heads == n_heads) ? h : (h / q_per_kv);
            const size_t kv_head_base = (size_t) layer * per_layer + (size_t) kv_h * per_head;
            const ggml_fp16_t * v_0 = v_cache_data + kv_head_base;
            float * const out_h = attn_data + (size_t) h * (size_t) head_dim;
            for (int d = 0; d < head_dim; ++d) {
                out_h[d] = ggml_fp16_to_fp32(v_0[d]);
            }
        }
        return attn_out;
    }

    // Per-Q-head attention.
    std::vector<float> scores((size_t) kv_len);
    for (int h = 0; h < n_heads; ++h) {
        const int kv_h = (n_kv_heads == n_heads) ? h : (h / q_per_kv);
        const float * q_h = q_src + (size_t) h * (size_t) head_dim;
        const size_t kv_head_base =
            (size_t) layer * per_layer +
            (size_t) kv_h * per_head;

        // Q . K_t for every cached position.
        for (int t = 0; t < kv_len; ++t) {
            const ggml_fp16_t * k_t = k_cache_data + kv_head_base + (size_t) t * (size_t) head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_h[d] * ggml_fp16_to_fp32(k_t[d]);
            }
            scores[(size_t) t] = dot * scale;
        }

        // Softmax (max-subtract for numerical stability).
        float max_score = scores[0];
        for (int t = 1; t < kv_len; ++t) {
            if (scores[(size_t) t] > max_score) max_score = scores[(size_t) t];
        }
        float sum = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            const float e = std::exp(scores[(size_t) t] - max_score);
            scores[(size_t) t] = e;
            sum += e;
        }
        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int t = 0; t < kv_len; ++t) scores[(size_t) t] *= inv_sum;

        // attn_h = sum_t scores_t * V_t.
        float * const out_h = attn_data + (size_t) h * (size_t) head_dim;
        for (int d = 0; d < head_dim; ++d) out_h[d] = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            const ggml_fp16_t * v_t = v_cache_data + kv_head_base + (size_t) t * (size_t) head_dim;
            const float w = scores[(size_t) t];
            for (int d = 0; d < head_dim; ++d) {
                out_h[d] += w * ggml_fp16_to_fp32(v_t[d]);
            }
        }
    }

    return attn_out;
}

// Depth-tier counterpart of apply_attention_eager_decode. Currently unused:
// the depth qkv tensor is produced by a lazy ggml_mul_mat (its ->data is
// only valid post-graph-compute), so we keep the depth attention chain
// fully lazy and rely on a depth KV cache wired via ggml_cpy/ggml_view +
// ggml_flash_attn_ext. This helper is kept as a reference for a future
// eager-qkv refactor; suppressing the unused-function warning so the
// production build stays warning-clean.
//
// The depth transformer is autoregressive across the codebook dimension: at
// codebook step k it must attend to the K/V written by previous steps 0..k-1
// in the SAME temporal frame. The depth KV cache is reset between temporal
// frames (bmo_reset_depth_kv) and grows from cb_index=0 up to dep_q-1.
//
// Layout matches the temporal helper but uses ctx.depth_k_cache /
// ctx.depth_v_cache, ctx.depth_n_ctx, ctx.depth_n_heads, ctx.depth_head_dim,
// and ctx.depth_n_layers. n_heads == n_kv_heads (no GQA on the depformer).
[[maybe_unused]] static ggml_tensor * apply_depth_attention_eager(
    bmo_context & ctx,
    ggml_context * wctx,
    ggml_tensor * q,        // F32 [head_dim, n_heads, 1] in staging slot (post RoPE)
    ggml_tensor * k,        // F32 [head_dim, n_heads, 1] in staging slot (post RoPE)
    ggml_tensor * v,        // F32 [head_dim, n_heads, 1] view into qkv slot
    int cb_index,
    int layer) {
    if (!q || !k || !v || !q->data || !k->data || !v->data) {
        throw std::runtime_error("apply_depth_attention_eager: null tensor data");
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32) {
        throw std::runtime_error("apply_depth_attention_eager: requires F32 q/k/v");
    }
    if (!ctx.depth_k_cache || !ctx.depth_v_cache ||
        !ctx.depth_k_cache->data || !ctx.depth_v_cache->data) {
        throw std::runtime_error("apply_depth_attention_eager: depth KV cache not initialised");
    }
    if (ctx.depth_k_cache->type != GGML_TYPE_F16 || ctx.depth_v_cache->type != GGML_TYPE_F16) {
        throw std::runtime_error("apply_depth_attention_eager: expected FP16 depth KV cache");
    }

    const int head_dim = (int) q->ne[0];
    const int n_heads  = (int) q->ne[1];
    const int n_token  = (int) q->ne[2];
    const int n_ctx    = ctx.depth_n_ctx;
    const int kv_len   = cb_index + n_token;

    if (n_token != 1) {
        throw std::runtime_error("apply_depth_attention_eager: only n_token==1 supported");
    }
    if (cb_index < 0 || kv_len > n_ctx) {
        throw std::runtime_error(
            "apply_depth_attention_eager: kv_len " + std::to_string(kv_len)
            + " exceeds depth_n_ctx " + std::to_string(n_ctx));
    }
    if (head_dim != ctx.depth_head_dim || n_heads != ctx.depth_n_heads) {
        throw std::runtime_error(
            "apply_depth_attention_eager: shape mismatch q.head_dim="
            + std::to_string(head_dim) + " ctx.depth_head_dim="
            + std::to_string(ctx.depth_head_dim) + " q.n_heads="
            + std::to_string(n_heads) + " ctx.depth_n_heads="
            + std::to_string(ctx.depth_n_heads));
    }

    {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true)) {
            std::fprintf(stderr,
                         "[bmo_depth_attn_eager] active: head_dim=%d n_heads=%d "
                         "n_ctx=%d kv_len=%d layer=%d\n",
                         head_dim, n_heads, n_ctx, kv_len, layer);
        }
    }

    ggml_fp16_t * const k_cache_data = (ggml_fp16_t *) ctx.depth_k_cache->data;
    ggml_fp16_t * const v_cache_data = (ggml_fp16_t *) ctx.depth_v_cache->data;
    const float * const k_src = (const float *) k->data;
    const float * const v_src = (const float *) v->data;

    const size_t per_layer = (size_t) head_dim * (size_t) n_ctx * (size_t) n_heads;
    const size_t per_head  = (size_t) head_dim * (size_t) n_ctx;

    // Step 1: write current K/V into the depth cache at position cb_index.
    for (int h = 0; h < n_heads; ++h) {
        const size_t cache_offset =
            (size_t) layer * per_layer +
            (size_t) h * per_head +
            (size_t) cb_index * (size_t) head_dim;
        const float * k_h = k_src + (size_t) h * (size_t) head_dim;
        const float * v_h = v_src + (size_t) h * (size_t) head_dim;
        for (int d = 0; d < head_dim; ++d) {
            k_cache_data[cache_offset + (size_t) d] = ggml_fp32_to_fp16(k_h[d]);
            v_cache_data[cache_offset + (size_t) d] = ggml_fp32_to_fp16(v_h[d]);
        }
    }

    // Step 2: allocate output [head_dim, n_heads, 1]
#ifdef BMO_JETSON
    staging_slot attn_slot = borrow_staging(ctx.staging);
    ggml_tensor * attn_out = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, head_dim, n_heads, 1);
    attn_out->data = attn_slot.host;
    attn_out->extra = attn_slot.dev;
    attn_slot.pool = nullptr;
#else
    ggml_tensor * attn_out = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, head_dim, n_heads, 1);
#endif
    if (!attn_out || !attn_out->data) {
        throw std::runtime_error("apply_depth_attention_eager: failed to allocate attn_out");
    }
    float * const attn_data = (float *) attn_out->data;

    const float * const q_src = (const float *) q->data;
    if (kv_len == 1) {
        for (int h = 0; h < n_heads; ++h) {
            const size_t kv_head_base = (size_t) layer * per_layer + (size_t) h * per_head;
            const ggml_fp16_t * v_0 = v_cache_data + kv_head_base;
            float * const out_h = attn_data + (size_t) h * (size_t) head_dim;
            for (int d = 0; d < head_dim; ++d) {
                out_h[d] = ggml_fp16_to_fp32(v_0[d]);
            }
        }
        return attn_out;
    }

    const float scale = 1.0f / std::sqrt((float) head_dim);
    std::vector<float> scores((size_t) kv_len);
    for (int h = 0; h < n_heads; ++h) {
        const float * q_h = q_src + (size_t) h * (size_t) head_dim;
        const size_t kv_head_base =
            (size_t) layer * per_layer +
            (size_t) h * per_head;

        for (int t = 0; t < kv_len; ++t) {
            const ggml_fp16_t * k_t = k_cache_data + kv_head_base + (size_t) t * (size_t) head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_h[d] * ggml_fp16_to_fp32(k_t[d]);
            }
            scores[(size_t) t] = dot * scale;
        }

        float max_score = scores[0];
        for (int t = 1; t < kv_len; ++t) {
            if (scores[(size_t) t] > max_score) max_score = scores[(size_t) t];
        }
        float sum = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            const float e = std::exp(scores[(size_t) t] - max_score);
            scores[(size_t) t] = e;
            sum += e;
        }
        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int t = 0; t < kv_len; ++t) scores[(size_t) t] *= inv_sum;

        float * const out_h = attn_data + (size_t) h * (size_t) head_dim;
        for (int d = 0; d < head_dim; ++d) out_h[d] = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            const ggml_fp16_t * v_t = v_cache_data + kv_head_base + (size_t) t * (size_t) head_dim;
            const float w = scores[(size_t) t];
            for (int d = 0; d < head_dim; ++d) {
                out_h[d] += w * ggml_fp16_to_fp32(v_t[d]);
            }
        }
    }

    return attn_out;
}
#endif

namespace {

static void stage_tensor_upload(bmo_context & ctx, ggml_tensor * tensor, const void * data, size_t size) {
    if (tensor && tensor->data && data) {
        std::memcpy(tensor->data, data, size);
    }
    bmo_context::owned_tensor_upload up;
    up.tensor = tensor;
    up.bytes.resize(size);
    std::memcpy(up.bytes.data(), data, size);
    ctx.graph_uploads.push_back(std::move(up));
}

static inline uint8_t unpack_u2_le(uint8_t byte, int lane) {
    return (byte >> (lane * 2)) & 0x3;
}

static ggml_tensor * get_tensor(ggml_context * data_ctx, const std::string & name) {
    return ggml_get_tensor(data_ctx, name.c_str());
}

static int32_t read_scalar_i32(ggml_context * data_ctx, const std::string & name, int32_t fallback = 0) {
    ggml_tensor * t = get_tensor(data_ctx, name);
    if (!t || ggml_nbytes(t) < (int) sizeof(int32_t)) {
        return fallback;
    }
    int32_t out = 0;
    std::memcpy(&out, t->data, sizeof(int32_t));
    return out;
}

static float read_scalar_f32(ggml_context * data_ctx, const std::string & name, float fallback = 0.0f) {
    ggml_tensor * t = get_tensor(data_ctx, name);
    if (!t || ggml_nbytes(t) < (int) sizeof(float)) {
        return fallback;
    }
    float out = 0.0f;
    std::memcpy(&out, t->data, sizeof(float));
    return out;
}

struct packed_linear_ref {
    std::string base;
    ggml_tensor * packed_weights = nullptr;
    ggml_tensor * packed_mask = nullptr;
    ggml_tensor * fp16_indices = nullptr;
    ggml_tensor * fp16_values = nullptr;
    ggml_tensor * dense_weight = nullptr;
    ggml_tensor * dense_bias = nullptr;
    device_packed_t * device_packed = nullptr;
};

static packed_linear_ref resolve_linear(ggml_context * data_ctx, const std::vector<std::string> & bases) {
    packed_linear_ref out;
    for (const auto & base : bases) {
        packed_linear_ref cand;
        cand.base = base;
        cand.packed_weights = get_tensor(data_ctx, base + ".packed_weights");
        cand.packed_mask = get_tensor(data_ctx, base + ".packed_mask");
        cand.fp16_indices = get_tensor(data_ctx, base + ".fp16_indices");
        cand.fp16_values = get_tensor(data_ctx, base + ".fp16_values");
        cand.dense_weight = get_tensor(data_ctx, base + ".weight");
        cand.dense_bias = get_tensor(data_ctx, base + ".bias");

        if (cand.packed_weights || cand.dense_weight) {
            return cand;
        }

        // Some exports use direct key where base itself is already "..._weight".
        cand.dense_weight = get_tensor(data_ctx, base);
        if (cand.dense_weight) {
            return cand;
        }

        // Older dot-style alias.
        std::string dot = base;
        std::replace(dot.begin(), dot.end(), '_', '.');
        cand.dense_weight = get_tensor(data_ctx, dot + ".weight");
        if (cand.dense_weight) {
            cand.base = dot;
            return cand;
        }
    }
    return out;
}

static int parse_temporal_layer_index(const std::string & base) {
    constexpr const char * prefix = "transformer_layers_";
    if (base.compare(0, std::strlen(prefix), prefix) != 0) {
        return -1;
    }

    const size_t start = std::strlen(prefix);
    size_t end = start;
    while (end < base.size() && std::isdigit(static_cast<unsigned char>(base[end]))) {
        ++end;
    }
    if (end == start) {
        return -1;
    }

    try {
        return std::stoi(base.substr(start, end - start));
    } catch (...) {
        return -1;
    }
}

static void unpack_layer_to_f32(
    const uint8_t * packed_weights,
    const uint8_t * packed_mask,
    int32_t rows,
    int32_t cols,
    int32_t n_2bit_bytes,
    int32_t n_4bit_bytes,
    int32_t n_8bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    const int32_t * fp16_indices,
    int64_t n_fp16,
    const ggml_fp16_t * fp16_values,
    float * out_w) {
    const int64_t total = (int64_t) rows * (int64_t) cols;

    const uint8_t * stream2 = packed_weights;
    const uint8_t * stream4 = packed_weights + n_2bit_bytes;
    const uint8_t * stream8 = packed_weights + n_2bit_bytes + n_4bit_bytes;

    int64_t idx2 = 0;
    int64_t idx4 = 0;
    int64_t idx8 = 0;

    for (int64_t pos = 0; pos < total; ++pos) {
        const uint8_t mbyte = packed_mask[(size_t) (pos / 4)];
        const uint8_t tier = unpack_u2_le(mbyte, (int) (pos % 4));

        float v = 0.0f;
        if (tier >= 3) {
            const uint8_t b = stream2[(size_t) (idx2 / 4)];
            const uint8_t q = unpack_u2_le(b, (int) (idx2 % 4));
            ++idx2;
            v = ((float) q - zp_low) * scale_low;
        } else if (tier == 2) {
            const uint8_t b = stream4[(size_t) (idx4 / 2)];
            const uint8_t q = (idx4 % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
            ++idx4;
            v = ((float) q - zp_int4) * scale_int4;
        } else if (tier == 1) {
            const uint8_t q = stream8[(size_t) idx8];
            ++idx8;
            v = ((float) q - zp_int8) * scale_int8;
        }

        out_w[(size_t) pos] = v;
    }

    for (int64_t i = 0; i < n_fp16; ++i) {
        const int32_t pos = fp16_indices[i];
        if (pos >= 0 && (int64_t) pos < total) {
            out_w[(size_t) pos] = ggml_fp16_to_fp32(fp16_values[i]);
        }
    }

    const int64_t used2 = (idx2 + 3) / 4;
    const int64_t used4 = (idx4 + 1) / 2;
    const int64_t used8 = idx8;

    if (used2 != n_2bit_bytes || used4 != n_4bit_bytes || used8 != n_8bit_bytes) {
        fprintf(stderr, "[WARNING] Bypassed stream padding mismatch!\n");
        fprintf(stderr, "[STREAM-DELTA] used2=%lld n2=%lld delta=%lld | used4=%lld n4=%lld delta=%lld | used8=%lld n8=%lld delta=%lld\n",
                (long long)used2, (long long)n_2bit_bytes, (long long)(used2 - n_2bit_bytes),
                (long long)used4, (long long)n_4bit_bytes, (long long)(used4 - n_4bit_bytes),
                (long long)used8, (long long)n_8bit_bytes, (long long)(used8 - n_8bit_bytes));
    }
}

static void unpack_layer_to_f32_blockwise(
    const uint8_t * packed_weights,
    const uint8_t * packed_mask,
    int32_t rows,
    int32_t cols,
    int32_t block_size,
    int32_t packing_version,
    int32_t n_2bit_bytes,
    int32_t n_4bit_bytes,
    int32_t n_8bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    const ggml_fp16_t * fp16_values,
    float * out_w) {
    const int64_t total = (int64_t) rows * (int64_t) cols;
    const uint8_t * stream2 = packed_weights;
    const uint8_t * stream4 = packed_weights + n_2bit_bytes;
    const uint8_t * stream8 = packed_weights + n_2bit_bytes + n_4bit_bytes;

    if (packing_version >= 5) {
        int32_t c2 = 0;
        int32_t c4 = 0;
        int32_t c8 = 0;
        int32_t c16 = 0;
        for (int64_t pos = 0; pos < total; ++pos) {
            const uint8_t mbyte = packed_mask[(size_t) (pos / 4)];
            const uint8_t tier = unpack_u2_le(mbyte, (int) (pos % 4));
            float v = 0.0f;
            if (tier == 0) {
                v = ggml_fp16_to_fp32(fp16_values[(size_t) c16]);
                ++c16;
            } else if (tier == 1) {
                const uint8_t q = stream8[(size_t) c8];
                ++c8;
                v = ((float) q - zp_int8) * scale_int8;
            } else if (tier == 2) {
                const int32_t idx = c4;
                const uint8_t b = stream4[(size_t) (idx / 2)];
                const uint8_t q = (idx % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
                ++c4;
                v = ((float) q - zp_int4) * scale_int4;
            } else {
                const int32_t idx = c2;
                const uint8_t b = stream2[(size_t) (idx / 4)];
                const uint8_t q = unpack_u2_le(b, (int) (idx % 4));
                ++c2;
                v = ((float) q - zp_low) * scale_low;
            }
            out_w[(size_t) pos] = v;
        }
        return;
    }

    const int64_t n_blocks = (total + block_size - 1) / block_size;
    int32_t c2 = 0;
    int32_t c4 = 0;
    int32_t c8 = 0;
    int32_t c16 = 0;
    for (int64_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const uint8_t mbyte = packed_mask[(size_t) (block_idx / 4)];
        const uint8_t tier = unpack_u2_le(mbyte, (int) (block_idx % 4));
        int32_t off = 0;
        if (tier == 0) {
            off = c16;
            c16 += block_size;
        } else if (tier == 1) {
            off = c8;
            c8 += block_size;
        } else if (tier == 2) {
            off = c4;
            c4 += block_size;
        } else {
            off = c2;
            c2 += block_size;
        }

        for (int32_t in_block = 0; in_block < block_size; ++in_block) {
            const int64_t pos = block_idx * block_size + in_block;
            if (pos >= total) {
                break;
            }
            float v = 0.0f;
            if (tier == 0) {
                v = ggml_fp16_to_fp32(fp16_values[(size_t) off + (size_t) in_block]);
            } else if (tier == 1) {
                const uint8_t q = stream8[off + in_block];
                v = ((float) q - zp_int8) * scale_int8;
            } else if (tier == 2) {
                const int32_t idx = off + in_block;
                const uint8_t b = stream4[idx / 2];
                const uint8_t q = (idx % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
                v = ((float) q - zp_int4) * scale_int4;
            } else {
                const int32_t idx = off + in_block;
                const uint8_t b = stream2[idx / 4];
                const uint8_t q = unpack_u2_le(b, (int) (idx % 4));
                v = ((float) q - zp_low) * scale_low;
            }
            out_w[(size_t) pos] = v;
        }
    }
}

static ggml_tensor * apply_linear_with_transient_unpack(
    bmo_context & ctx,
    bmo_model & model,
    ggml_context * wctx,
    ggml_tensor * x,
    const std::vector<std::string> & base_candidates) {

    auto unpack_t0 = std::chrono::steady_clock::now();

    packed_linear_ref linear = resolve_linear(model.wctx, base_candidates);
    // device packed metadata is stored in ctx.packed_registry keyed by base name

    if (!linear.packed_weights && !linear.dense_weight) {
        // Naive correctness mode: if a specific projection is not exported in this GGUF,
        // pass through so the graph can still be built while mapping coverage is expanded.
        return x;
    }

    ggml_tensor * y = nullptr;

    if (linear.packed_weights) {
        const int32_t cols = (int32_t) x->ne[0];

        int32_t rows = read_scalar_i32(model.wctx, linear.base + ".rows", 0);
        if (rows <= 0) {
            rows = read_scalar_i32(model.wctx, linear.base + ".out_features", 0);
        }
        if (rows <= 0) {
            throw std::runtime_error("missing rows/out_features metadata for " + linear.base);
        }

        const int32_t n_2bit_bytes = read_scalar_i32(model.wctx, linear.base + ".n_2bit_bytes", 0);
        const int32_t n_4bit_bytes = read_scalar_i32(model.wctx, linear.base + ".n_4bit_bytes", 0);
        const int32_t n_8bit_bytes = read_scalar_i32(model.wctx, linear.base + ".n_8bit_bytes", 0);

        const float scale_low = read_scalar_f32(model.wctx, linear.base + ".scale_low", 1.0f);
        const float scale_int4 = read_scalar_f32(model.wctx, linear.base + ".scale_int4", 1.0f);
        const float scale_int8 = read_scalar_f32(model.wctx, linear.base + ".scale_int8", 1.0f);

        const float zp_low = read_scalar_f32(model.wctx, linear.base + ".zp_low", 1.5f);
        const float zp_int4 = read_scalar_f32(model.wctx, linear.base + ".zp_int4", 7.5f);
        const float zp_int8 = read_scalar_f32(model.wctx, linear.base + ".zp_int8", 127.5f);

        if (!linear.packed_mask || !linear.fp16_values) {
            throw std::runtime_error("incomplete packed tensor set for " + linear.base);
        }

#ifdef BMO_ENABLE_CUDA
        if (ctx.cuda_backend && ctx.packed_registry.find(linear.base) != ctx.packed_registry.end()) {
            device_packed_t & dp_ref = ctx.packed_registry[linear.base];
            if (dp_ref.is_valid) {
#ifdef BMO_JETSON
                if (x->type != GGML_TYPE_F32) {
                    throw std::runtime_error("Jetson fused linear requires GGML_TYPE_F32 activations for "
                                             + linear.base);
                }
                if (!ctx.cuda_fused_output_buffer || !ctx.cuda_fused_output_buffer_dev) {
                    throw std::runtime_error("Jetson fused output buffer is not allocated");
                }
                if (!ctx.cuda_fused_input_buffer || !ctx.cuda_fused_input_buffer_dev) {
                    throw std::runtime_error("Jetson fused input buffer is not allocated");
                }
                {
                    const size_t max_in_elems = 11264ULL;
                    if ((size_t) cols > max_in_elems) {
                        throw std::runtime_error("Activation cols exceed fused input buffer capacity for "
                                                 + linear.base);
                    }
                }

                const bool has_fv = dp_ref.host_fp16_values && dp_ref.fv_size > 0;
                if (!dp_ref.preloaded || !dp_ref.canonical_pw_dev || !dp_ref.canonical_pm_dev) {
                    throw std::runtime_error("Jetson fused: preloaded canonical pointers missing for "
                                             + linear.base);
                } else if (has_fv && !dp_ref.canonical_fv_dev) {
                    throw std::runtime_error("Jetson fused: preloaded canonical fv missing for "
                                             + linear.base);
                }
                if (!dp_ref.row_c2 || !dp_ref.row_c4 || !dp_ref.row_c8 || !dp_ref.row_c16) {
                    throw std::runtime_error("Jetson fused: row tier-base device buffers missing for "
                                             + linear.base);
                }

                const size_t row_out_bytes = (size_t) rows * sizeof(float);
                if (row_out_bytes > ctx.cuda_fused_output_buffer_bytes) {
                    throw std::runtime_error("Fused output buffer too small for rows in " + linear.base);
                }
                const int64_t n_el = ggml_nelements(x);
                if (n_el <= 0 || (n_el % (int64_t) cols) != 0) {
                    throw std::runtime_error("Invalid activation shape for fused linear " + linear.base);
                }
                const int64_t n_tok = n_el / (int64_t) cols;

                auto e2e_t0 = std::chrono::steady_clock::now();

                const void * kern_pw = dp_ref.canonical_pw_dev;
                const void * kern_pm = dp_ref.canonical_pm_dev;
                const void * kern_fv = has_fv ? static_cast<const void *>(dp_ref.canonical_fv_dev) : nullptr;
                const int    block_size_eff = dp_ref.block_size > 0 ? dp_ref.block_size : 32;
                const size_t x_vec_bytes    = (size_t) cols * sizeof(float);

                if (n_tok == 1) {
                    // -------- Fast path: alias kernel output into a staging slot --------
                    //
                    // The previous design memcpy'd fused_out_host -> wctx after a
                    // per-call cudaStreamSynchronize. Under the deferred-sync
                    // architecture, we cannot host-memcpy a buffer that the GPU is
                    // still writing through cuda_fused_output_buffer_dev. Instead we
                    // borrow a slot from the staging pool, point the kernel at it,
                    // and alias out_lm->data to slot.host. The boundary sync before
                    // the next CPU consumer (flash_attn / silu) ensures the slot is
                    // populated before any host read.
                    if (row_out_bytes > gpu_staging_pool::SLOT_BYTES) {
                        throw std::runtime_error("matvec output exceeds staging slot capacity for "
                                                 + linear.base);
                    }

                    // Map x->data through CUDA when possible (e.g. previous eager op's
                    // staging slot); only fall back to a host->host memcpy when the
                    // input lives in unregistered wctx memory. This avoids racing
                    // against an in-flight upstream kernel writing to x's slot.
                    float * x_dev = (float *) x->extra;
                    if (!x_dev) {
                        if (cudaHostGetDevicePointer((void **) &x_dev, x->data, 0) == cudaSuccess) {
                            x->extra = x_dev;
                        } else {
                            cudaGetLastError(); // flush stale error
                            std::memcpy(ctx.cuda_fused_input_buffer, x->data, x_vec_bytes);
                            x_dev = reinterpret_cast<float *>(ctx.cuda_fused_input_buffer_dev);
                        }
                    }

                    staging_slot out_slot = borrow_staging(ctx.staging);
                    launch_fused_dequant_matvec_jetson(
                        dp_ref,
                        linear.base.c_str(),
                        kern_pw,
                        kern_pm,
                        kern_fv,
                        rows,
                        cols,
                        block_size_eff,
                        x_dev,
                        reinterpret_cast<float *>(out_slot.dev),
                        nullptr);

                    if (cudaGetLastError() != cudaSuccess) {
                        throw std::runtime_error("fused_matvec CUDA error for " + linear.base);
                    }

                    ggml_tensor * out_lm = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, rows);
                    out_lm->data = out_slot.host;
                    out_lm->extra = out_slot.dev;
                    out_slot.pool = nullptr; // detach: released by release_all_staging at layer end

                    y = out_lm;
                    if (linear.dense_bias) {
                        y = ggml_add(wctx, y, linear.dense_bias);
                    }

                    if (getenv("BMO_LOG_PROF")) {
                        auto e2e_t1 = std::chrono::steady_clock::now();
                        const double e2e_ms =
                            std::chrono::duration<double, std::milli>(e2e_t1 - e2e_t0).count();
                        std::fprintf(stderr,
                                     "[prof_prod] base=%s kernel=async e2e_with_graph=%.2fms\n",
                                     linear.base.c_str(), e2e_ms);
                    }

                    return y;
                }

                // -------- Multi-token prefill path: keep per-iteration sync+memcpy --------
                //
                // A single staging slot cannot hold rows*n_tok floats and the
                // shared cuda_fused_output_buffer is overwritten by every kernel,
                // so we synchronize per token and memcpy the result into wctx.
                ggml_tensor * out_lm = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, rows, n_tok);
                float * fused_out_host = reinterpret_cast<float *>(ctx.cuda_fused_output_buffer);
                float * fused_out_dev  = reinterpret_cast<float *>(ctx.cuda_fused_output_buffer_dev);

                for (int64_t t = 0; t < n_tok; ++t) {
                    const float * x_col = reinterpret_cast<const float *>(
                        (const uint8_t *) x->data + (size_t) t * x->nb[1]);
                    std::memcpy(ctx.cuda_fused_input_buffer, x_col, x_vec_bytes);

                    launch_fused_dequant_matvec_jetson(
                        dp_ref,
                            linear.base.c_str(),
                        kern_pw,
                        kern_pm,
                        kern_fv,
                        rows,
                        cols,
                        block_size_eff,
                        reinterpret_cast<const float *>(ctx.cuda_fused_input_buffer_dev),
                        fused_out_dev,
                        nullptr);

                    cudaError_t sync_err = cudaStreamSynchronize(0);
                    if (sync_err != cudaSuccess) {
                        throw std::runtime_error(std::string("CUDA sync failed after fused matvec for ")
                                                 + linear.base + ": "
                                                 + cudaGetErrorString(sync_err));
                    }
                    if (cudaGetLastError() != cudaSuccess) {
                        throw std::runtime_error("fused_matvec CUDA error for " + linear.base);
                    }

                    std::memcpy((uint8_t *) out_lm->data + (size_t) t * out_lm->nb[1],
                                fused_out_host,
                                row_out_bytes);
                }

                y = out_lm;
                if (linear.dense_bias) {
                    y = ggml_add(wctx, y, linear.dense_bias);
                }

                if (getenv("BMO_LOG_PROF")) {
                    auto e2e_t1 = std::chrono::steady_clock::now();
                    const double e2e_ms =
                        std::chrono::duration<double, std::milli>(e2e_t1 - e2e_t0).count();
                    std::fprintf(stderr,
                                 "[prof_prod] base=%s n_tok=%lld kernel=sync e2e_with_graph=%.2fms\n",
                                 linear.base.c_str(), (long long) n_tok, e2e_ms);
                }

                return y;
#endif
#ifndef BMO_JETSON
                ggml_tensor * W = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, cols, rows);

                const int64_t total = (int64_t) rows * (int64_t) cols;
                const size_t total_bytes = (size_t) total * sizeof(float);
                if (!ctx.cuda_unpack_scratch || ctx.cuda_unpack_scratch_bytes < total_bytes) {
                    throw std::runtime_error("CUDA unpack scratch is too small for " + linear.base);
                }

                if ((int64_t) ctx.shared_scratch_w.size() < total) {
                    ctx.shared_scratch_w.resize((size_t) total);
                }

                launch_unpack_kernel(&dp_ref,
                                     rows,
                                     cols,
                                     n_2bit_bytes,
                                     n_4bit_bytes,
                                     n_8bit_bytes,
                                     scale_low,
                                     scale_int4,
                                     scale_int8,
                                     zp_low,
                                     zp_int4,
                                     zp_int8,
                                     reinterpret_cast<float *>(ctx.cuda_unpack_scratch));

                cudaError_t sync_err = cudaDeviceSynchronize();
                if (sync_err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA sync failed after unpack for ") + linear.base
                                             + ": " + cudaGetErrorString(sync_err));
                }

                cudaError_t copy_err =
                    cudaMemcpy(ctx.shared_scratch_w.data(),
                               ctx.cuda_unpack_scratch,
                               total_bytes,
                               cudaMemcpyDeviceToHost);
                if (copy_err != cudaSuccess) {
                    throw std::runtime_error(std::string("cudaMemcpy D2H failed after unpack for ")
                                             + linear.base + ": " + cudaGetErrorString(copy_err));
                }

                std::memcpy(W->data, ctx.shared_scratch_w.data(), total_bytes);

                if (getenv("BMO_LOG_PROF")) {
                    auto unpack_t1 = std::chrono::steady_clock::now();
                    long unpack_us =
                        std::chrono::duration_cast<std::chrono::microseconds>(unpack_t1 - unpack_t0).count();
                    std::fprintf(stderr, "[prof_unpack] base=%s rows=%d cols=%d cuda_d2h_unpack_us=%ld\n",
                                 linear.base.c_str(), rows, cols, unpack_us);
                }

                y = ggml_mul_mat(wctx, S(W), S(x));
                if (linear.dense_bias) {
                    y = ggml_add(wctx, y, linear.dense_bias);
                }
                return y;
#endif
            }
        }
#endif
        {
            ggml_tensor * W = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, cols, rows);

            const int64_t total = (int64_t) rows * (int64_t) cols;
            if ((int64_t) ctx.shared_scratch_w.size() < total) {
                ctx.shared_scratch_w.resize((size_t) total);
            }

            // Memory lifecycle note:
            // 1) We unpack exactly one layer's packed streams into ctx.shared_scratch_w.
            // 2) The same scratch buffer is reused for the next layer, so peak unpacked
            //    host memory is bounded by the largest temporal matrix (~gating linear_in).
            // 3) This keeps us from storing all 32 unpacked F32 matrices at once.
            const uint8_t * pw = reinterpret_cast<const uint8_t *>(linear.packed_weights->data);
            const uint8_t * pm = reinterpret_cast<const uint8_t *>(linear.packed_mask->data);
            const int32_t block_size = read_scalar_i32(model.wctx, linear.base + ".block_size", 0);

            const ggml_fp16_t * fv16 = nullptr;
            std::vector<ggml_fp16_t> tmp_f16;
            if (linear.fp16_values->type == GGML_TYPE_F16) {
                fv16 = reinterpret_cast<const ggml_fp16_t *>(linear.fp16_values->data);
            } else if (linear.fp16_values->type == GGML_TYPE_F32) {
                const float * src = reinterpret_cast<const float *>(linear.fp16_values->data);
                const int64_t n_fp16 = ggml_nbytes(linear.fp16_values) / (int64_t) sizeof(float);
                tmp_f16.resize((size_t) n_fp16);
                for (int64_t i = 0; i < n_fp16; ++i) {
                    tmp_f16[(size_t) i] = ggml_fp32_to_fp16(src[i]);
                }
                fv16 = tmp_f16.data();
            } else {
                throw std::runtime_error("unsupported fp16_values type in " + linear.base);
            }

            if (block_size > 0) {
                const int32_t packing_version =
                    read_scalar_i32(model.wctx, linear.base + ".packing_version", 3);
                unpack_layer_to_f32_blockwise(
                    pw,
                    pm,
                    rows,
                    cols,
                    block_size,
                    packing_version,
                    n_2bit_bytes,
                    n_4bit_bytes,
                    n_8bit_bytes,
                    scale_low,
                    scale_int4,
                    scale_int8,
                    zp_low,
                    zp_int4,
                    zp_int8,
                    fv16,
                    ctx.shared_scratch_w.data());
            } else {
                if (!linear.fp16_indices) {
                    throw std::runtime_error("legacy packed tensor missing fp16_indices for " + linear.base);
                }
                const int32_t * fi = reinterpret_cast<const int32_t *>(linear.fp16_indices->data);
                const int64_t n_fp16 = ggml_nbytes(linear.fp16_indices) / (int64_t) sizeof(int32_t);
                unpack_layer_to_f32(
                    pw,
                    pm,
                    rows,
                    cols,
                    n_2bit_bytes,
                    n_4bit_bytes,
                    n_8bit_bytes,
                    scale_low,
                    scale_int4,
                    scale_int8,
                    zp_low,
                    zp_int4,
                    zp_int8,
                    fi,
                    n_fp16,
                    fv16,
                    ctx.shared_scratch_w.data());
            }

            if (getenv("BMO_LOG_PROF")) {
                auto unpack_t1 = std::chrono::steady_clock::now();
                long unpack_ms = std::chrono::duration_cast<std::chrono::microseconds>(unpack_t1 - unpack_t0).count();
                std::fprintf(stderr, "[prof_unpack] base=%s rows=%d cols=%d unpack_us=%ld\n", linear.base.c_str(), rows, cols, unpack_ms);
            }

            std::memcpy(W->data, ctx.shared_scratch_w.data(), (size_t) total * sizeof(float));

            // CPU / non-CUDA packed path: W only exists in this scope
            y = ggml_mul_mat(wctx, S(W), S(x));
        }
    } else {
#ifdef BMO_JETSON
        if (linear.dense_weight && linear.dense_weight->type == GGML_TYPE_Q4_0 &&
            x->type == GGML_TYPE_F32 && ggml_nelements(x) == linear.dense_weight->ne[0]) {
            y = apply_dense_q4_0_linear_gpu(ctx, wctx, linear.dense_weight, x);
        } else {
            y = ggml_mul_mat(wctx, linear.dense_weight, x);
        }
#else
        y = ggml_mul_mat(wctx, linear.dense_weight, x);
#endif
    }

    if (linear.dense_bias) {
#ifdef BMO_JETSON
        if (y && y->type == GGML_TYPE_F32 && linear.dense_bias->type == GGML_TYPE_F32 &&
            ggml_nelements(y) == ggml_nelements(linear.dense_bias)) {
            y = apply_residual_gpu(ctx, wctx, y, linear.dense_bias);
        } else {
            y = ggml_add(wctx, y, linear.dense_bias);
        }
#else
        y = ggml_add(wctx, y, linear.dense_bias);
#endif
    }

    if (getenv("BMO_LOG_PROF") && !linear.packed_weights) {
        auto unpack_t1 = std::chrono::steady_clock::now();
        long unpack_us = std::chrono::duration_cast<std::chrono::microseconds>(unpack_t1 - unpack_t0).count();
        std::fprintf(stderr, "[prof_unpack] base=%s dense_path_us=%ld\n", linear.base.c_str(), unpack_us);
    }

    return y;
}

} // namespace

void bmo_reset_work_ctx(bmo_context & ctx) {
    if (ctx.work_ctx) {
        ggml_free(ctx.work_ctx);
        ctx.work_ctx = nullptr;
    }
    if (ctx.work_mem.empty()) {
        throw std::runtime_error("bmo_reset_work_ctx: ctx.work_mem is empty; resize() it first");
    }
    ggml_init_params wp = {
        ctx.work_mem.size(),
        ctx.work_mem.data(),
        /*.no_alloc =*/ false,
    };
    ctx.work_ctx = ggml_init(wp);
    if (!ctx.work_ctx) {
        throw std::runtime_error("bmo_reset_work_ctx: ggml_init failed");
    }
}

void bmo_embed_input_tokens_into(
    const bmo_context & ctx,
    const bmo_model & model,
    const int32_t * input_tokens,
    int num_codebooks,
    float * acc) {
    const int64_t n_embd = ctx.n_embd;
    std::memset(acc, 0, (size_t) n_embd * sizeof(float));
    std::vector<float> tmp_row(n_embd);

    auto add_row = [&](const ggml_tensor * t, int32_t tok, int channel_id) {
        if (!t || tok < 0) return;
        const int64_t d = t->ne[0];
        const int64_t vocab = t->ne[1];
        if (d != n_embd) return;
        if (vocab > 0 && tok >= vocab) return;
        const uint8_t * row = (const uint8_t *) t->data + (size_t) tok * t->nb[1];
        switch (t->type) {
            case GGML_TYPE_F32: {
                const float * f = (const float *) row;
                for (int64_t i = 0; i < n_embd; ++i) acc[i] += f[i];
                break;
            }
            case GGML_TYPE_F16: {
                const ggml_fp16_t * h = (const ggml_fp16_t *) row;
                for (int64_t i = 0; i < n_embd; ++i) acc[i] += ggml_fp16_to_fp32(h[i]);
                break;
            }
            default: {
                const struct ggml_type_traits * traits = ggml_get_type_traits(t->type);
                if (traits && traits->to_float) {
                    traits->to_float(row, tmp_row.data(), n_embd);
                    for (int64_t i = 0; i < n_embd; ++i) acc[i] += tmp_row[i];
                }
                break;
            }
        }
    };

    add_row(model.temporal_text_emb, input_tokens[0], /*channel_id=*/0);
    for (int k = 1; k < num_codebooks; ++k) {
        const int audio_idx = k - 1;
        if ((size_t) audio_idx >= model.temporal_audio_embs.size()) break;
        add_row(model.temporal_audio_embs[(size_t) audio_idx], input_tokens[k], k);
    }
}

ggml_tensor * bmo_embed_input_tokens(
    bmo_context & ctx,
    bmo_model & model,
    const int32_t * input_tokens,
    int num_codebooks) {
    if (!input_tokens) {
        throw std::runtime_error("bmo_embed_input_tokens: input_tokens is null");
    }
    int64_t n_embd = ctx.n_embd;
    if (n_embd <= 0) {
        if (model.temporal_text_emb) {
            n_embd = model.temporal_text_emb->ne[0];
        } else {
            for (auto * t : model.temporal_audio_embs) {
                if (t) { n_embd = t->ne[0]; break; }
            }
        }
    }
    if (n_embd <= 0) {
        throw std::runtime_error(
            "bmo_embed_input_tokens: no temporal embedding tables loaded "
            "(emb.{k}.weight / text_emb.weight). Re-export the GGUF.");
    }

    ggml_context * wctx = ctx.work_ctx;
    ggml_tensor * out = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, 1);
    if (!out || !out->data) {
        throw std::runtime_error("bmo_embed_input_tokens: failed to allocate output tensor");
    }
    float * acc = (float *) out->data;
    bmo_embed_input_tokens_into(ctx, model, input_tokens, num_codebooks, acc);

    // Diagnostic: confirm the embedding sum is non-zero. With token text=PAD=3
    // plus audio=0 across all 16 codebooks, the sum should NOT be all zeros.
    // We now also report MEAN (the DC component), which is the single most
    // useful number for the "DC attractor" investigation -- silence inputs
    // should give |mean| ~ 0, non-silence inputs should also give |mean|
    // small if the embedding tables are correctly loaded and balanced.
    if (getenv("BMO_LOG_EMBED")) {
        double s = 0, s2 = 0;
        float vmin = acc[0], vmax = acc[0];
        for (int64_t i = 0; i < n_embd; ++i) {
            s  += (double) acc[i];
            s2 += (double) acc[i] * (double) acc[i];
            if (acc[i] < vmin) vmin = acc[i];
            if (acc[i] > vmax) vmax = acc[i];
        }
        const double mean = s / (double) n_embd;
        const double rms2 = s2 / (double) n_embd;
        const double var  = rms2 - mean * mean;
        const float rms = (float) std::sqrt(rms2);
        const float stdv = (float) std::sqrt(std::max(0.0, var));
        fprintf(stderr,
                "[bmo_embed] tokens=[%d %d %d %d ...] num_cb=%d n_embd=%lld out[0..3]=%.4f %.4f %.4f %.4f mean=%+.5f std=%.5f rms=%.4f min=%.4f max=%.4f\n",
                (int) input_tokens[0],
                num_codebooks > 1 ? (int) input_tokens[1] : -1,
                num_codebooks > 2 ? (int) input_tokens[2] : -1,
                num_codebooks > 3 ? (int) input_tokens[3] : -1,
                num_codebooks, (long long) n_embd,
                acc[0], acc[1], acc[2], acc[3], mean, stdv, rms, vmin, vmax);
        // Full tensor geometry + fixed tok={2,100} fingerprints every call so a
        // single forward's stderr answers "same row?" without needing two pastes.
        if (model.temporal_text_emb && model.temporal_text_emb->data) {
            const ggml_tensor * te = model.temporal_text_emb;
            const size_t esz = ggml_type_size(te->type);
            const size_t expect_nb1 = (size_t) te->ne[0] * esz;
            fprintf(stderr,
                    "[bmo_embed] text_emb_geom ne0=%lld ne1=%lld nb0=%zu nb1=%zu "
                    "expect_nb1(ne0*esz)=%zu type=%s data=%p%s\n",
                    (long long) te->ne[0], (long long) te->ne[1],
                    (size_t) te->nb[0], (size_t) te->nb[1], expect_nb1,
                    ggml_type_name(te->type), te->data,
                    te->nb[1] != expect_nb1 ? "  !! nb1!=expect (non-contiguous layout?)" : "");

            auto row_sqsum_first64 = [&](int tok) -> double {
                if (tok < 0 || (int64_t) tok >= te->ne[1]) return -1.0;
                const uint8_t * row = (const uint8_t *) te->data + (size_t) tok * te->nb[1];
                const int64_t npeek = std::min<int64_t>(64, te->ne[0]);
                double s = 0.0;
                if (te->type == GGML_TYPE_F16) {
                    const ggml_fp16_t * h = (const ggml_fp16_t *) row;
                    for (int64_t i = 0; i < npeek; ++i) {
                        float v = ggml_fp16_to_fp32(h[i]);
                        s += (double) v * (double) v;
                    }
                } else if (te->type == GGML_TYPE_F32) {
                    const float * f = (const float *) row;
                    for (int64_t i = 0; i < npeek; ++i) {
                        s += (double) f[i] * (double) f[i];
                    }
                }
                return s;
            };
            auto peek_row0123 = [&](int tok, float out[4]) {
                for (int i = 0; i < 4; ++i) out[i] = 0.f;
                if (tok < 0 || (int64_t) tok >= te->ne[1]) return;
                const uint8_t * row = (const uint8_t *) te->data + (size_t) tok * te->nb[1];
                if (te->type == GGML_TYPE_F16) {
                    const ggml_fp16_t * h = (const ggml_fp16_t *) row;
                    for (int i = 0; i < 4 && i < (int) te->ne[0]; ++i) {
                        out[i] = ggml_fp16_to_fp32(h[i]);
                    }
                } else if (te->type == GGML_TYPE_F32) {
                    const float * f = (const float *) row;
                    for (int i = 0; i < 4 && i < (int) te->ne[0]; ++i) {
                        out[i] = f[i];
                    }
                }
            };
            const size_t off2   = (size_t) 2 * te->nb[1];
            const size_t off100 = (size_t) 100 * te->nb[1];
            const double s2 = row_sqsum_first64(2);
            const double s100 = row_sqsum_first64(100);
            fprintf(stderr,
                    "[bmo_embed] row_byte_off tok=2 -> %zu tok=100 -> %zu delta=%td\n",
                    off2, off100,
                    (ptrdiff_t) off100 - (ptrdiff_t) off2);
            fprintf(stderr,
                    "[bmo_embed] row_sqsum_first64 tok=2 -> %.10f tok=100 -> %.10f "
                    "same_sqsum=%s\n",
                    s2, s100, (s2 == s100) ? "YES" : "NO");
            float p2[4], p100[4];
            peek_row0123(2, p2);
            peek_row0123(100, p100);
            fprintf(stderr,
                    "[bmo_embed] text_emb_row[2][0..3]=%.6f %.6f %.6f %.6f\n",
                    p2[0], p2[1], p2[2], p2[3]);
            fprintf(stderr,
                    "[bmo_embed] text_emb_row[100][0..3]=%.6f %.6f %.6f %.6f\n",
                    p100[0], p100[1], p100[2], p100[3]);
        }
        // Peek row for the active text token (same as historical diagnostic).
        if (model.temporal_text_emb && model.temporal_text_emb->data
            && input_tokens[0] >= 0 && input_tokens[0] < model.temporal_text_emb->ne[1]) {
            const auto * t = model.temporal_text_emb;
            const uint8_t * row = (const uint8_t *) t->data + (size_t) input_tokens[0] * t->nb[1];
            float r0=0, r1=0, r2=0, r3=0;
            if (t->type == GGML_TYPE_F16) {
                const ggml_fp16_t * h = (const ggml_fp16_t *) row;
                r0 = ggml_fp16_to_fp32(h[0]); r1 = ggml_fp16_to_fp32(h[1]);
                r2 = ggml_fp16_to_fp32(h[2]); r3 = ggml_fp16_to_fp32(h[3]);
            } else if (t->type == GGML_TYPE_F32) {
                const float * f = (const float *) row;
                r0 = f[0]; r1 = f[1]; r2 = f[2]; r3 = f[3];
            }
            fprintf(stderr,
                    "[bmo_embed]   text_emb_row[%d][0..3]=%.4f %.4f %.4f %.4f (type=%s nb1=%zu data=%p)\n",
                    (int) input_tokens[0], r0, r1, r2, r3,
                    ggml_type_name(t->type), t->nb[1], t->data);
        }
    }

    return out;
}

void bmo_execute_graph(bmo_context & ctx, ggml_cgraph * gf, const std::vector<tensor_upload> & inputs, bool sync_cuda) {
    for (const auto & up : inputs) {
        if (up.tensor && up.tensor->data && up.host_data) {
            std::memcpy(up.tensor->data, up.host_data, ggml_nbytes(up.tensor));
        }
    }
    ctx.graph_uploads.clear();

#ifdef BMO_JETSON
    (void) cudaGetLastError();
    if (sync_cuda) {
        if (ctx.stream) {
            cudaStreamSynchronize((cudaStream_t) ctx.stream);
        } else {
            cudaStreamSynchronize(0);
        }
    }
#endif

    bool has_compute_nodes = false;
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);
        if (node && node->op != GGML_OP_NONE) {
            has_compute_nodes = true;
            break;
        }
    }
    if (has_compute_nodes) {
        int n_threads = 6;
        const char * env_threads = std::getenv("BMO_NUM_THREADS");
        if (env_threads) {
            n_threads = std::max(1, std::atoi(env_threads));
        }
        const ggml_status status = ggml_graph_compute_with_ctx(ctx.work_ctx, gf, n_threads);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_graph_compute_with_ctx failed");
        }
    }

    // text_logits is produced by ggml_mul_mat during graph compute; dump after compute so ->data is valid.
    if (getenv("BMO_DUMP_DEEP")) {
#ifdef BMO_ENABLE_CUDA
        cudaStreamSynchronize(0);
#endif
        ggml_tensor * tl = ggml_graph_get_tensor(gf, "text_logits");
        if (tl) {
            bmo_dump_tensor_f32(tl, "final_logits", bmo_dump_kind::Deep);
            if (getenv("BMO_DUMP_DEEP_EXIT")) {
                std::exit(0);
            }
        }
    }
}

ggml_cgraph * bmo_build_temporal_graph(
    bmo_context & ctx,
    bmo_model & model,
    ggml_tensor * input_tokens,
    int n_past,
    int layer_begin,
    int layer_end) {
    if (!input_tokens) {
        throw std::runtime_error("bmo_build_temporal_graph: input_tokens is null");
    }
    if (!ctx.k_cache || !ctx.v_cache) {
        throw std::runtime_error("bmo_build_temporal_graph: KV cache not initialized");
    }
    if (model.temporal_layers.size() != (size_t) ctx.n_layers) {
        throw std::runtime_error("bmo_build_temporal_graph: temporal_layers size mismatch");
    }

    const int64_t n_token = input_tokens->ne[1];
    if (n_token <= 0) {
        throw std::runtime_error("bmo_build_temporal_graph: invalid token count");
    }

    if (!ctx.work_ctx) {
        throw std::runtime_error("failed to initialize temporal work context");
    }

    const int begin = std::max(0, layer_begin);
    const int end = (layer_end < 0) ? ctx.n_layers : std::min(layer_end, ctx.n_layers);
    if (begin >= end) {
        throw std::runtime_error("bmo_build_temporal_graph: invalid layer range");
    }

    ggml_context * wctx = ctx.work_ctx;
    ggml_cgraph * gf = ggml_new_graph(wctx);
    ctx.graph_uploads.clear();

    // Position ids include n_past so RoPE/KV indexing follows autoregressive offset.
    ggml_tensor * pos = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, n_token);
    std::vector<int32_t> pos_host((size_t) n_token);
    for (int64_t t = 0; t < n_token; ++t) {
        pos_host[(size_t) t] = (int32_t) n_past + (int32_t) t;
    }
    stage_tensor_upload(ctx, pos, pos_host.data(), (size_t) n_token * sizeof(int32_t));

    // The temporal graph runs *eager* GPU ops (apply_rmsnorm_gpu,
    // apply_linear_with_transient_unpack, ...) that read x->data at build
    // time. ggml_cont() allocates a fresh, uninitialised buffer for the
    // output and only fills it during ggml_graph_compute_with_ctx (which we
    // run after the build) -- so a naive ggml_cont here would feed garbage
    // into layer 0 and propagate NaNs. When the caller already provided a
    // contiguous leaf (bmo_embed_input_tokens, bmo_main's cascade harness),
    // alias it directly so x->data points at the populated source data.
    // Otherwise we fall back to ggml_cont + an immediate host-side memcpy.
    ggml_tensor * x = nullptr;
    if (ggml_is_contiguous(S(input_tokens))) {
        x = input_tokens;
    } else {
        x = ggml_cont(wctx, S(input_tokens));
        if (x && x->data && input_tokens->data &&
            ggml_nbytes(x) == ggml_nbytes(input_tokens)) {
            std::memcpy(x->data, input_tokens->data, ggml_nbytes(x));
        }
    }

    for (int layer = begin; layer < end; ++layer) {
        const std::string base = "transformer_layers_" + std::to_string(layer);
        const bool is_final_layer = (layer == end - 1);

        static const int dump_deep_layers[] = {0, 1, 8, 15, 23, 31};
        const bool dump_deep_env = getenv("BMO_DUMP_DEEP") != nullptr;
        bool dump_this_layer = false;
        if (dump_deep_env) {
            for (int dl : dump_deep_layers) {
                if (dl == layer) {
                    dump_this_layer = true;
                    break;
                }
            }
        }

        // -------- Attention block --------
        ggml_tensor * residual = x;
        if (layer == 0 && getenv("BMO_DUMP_LAYER0")) {
            bmo_dump_tensor_f32(residual, "embed_sum");
        }
        if (dump_this_layer) {
            bmo_dump_tensor_f32(residual, ("layer" + std::to_string(layer) + "_x_in").c_str(), bmo_dump_kind::Deep);
        }
#ifdef BMO_JETSON
        ggml_tensor * x_norm;
        if (model.temporal_layers[layer].norm1_weight) {
            x_norm = apply_rmsnorm_gpu(ctx, wctx, x, model.temporal_layers[layer].norm1_weight, ctx.norm_eps);
        } else {
            x_norm = ggml_rms_norm(wctx, x, ctx.norm_eps);
        }
#else
        ggml_tensor * x_norm = ggml_rms_norm(wctx, x, ctx.norm_eps);
        if (model.temporal_layers[layer].norm1_weight) {
            x_norm = ggml_mul(wctx, x_norm, model.temporal_layers[layer].norm1_weight);
        }
#endif
        bmo_h3_tap(layer, "T3", x_norm);
        if (layer == 0 && getenv("BMO_DUMP_LAYER0")) {
            bmo_dump_tensor_f32(x_norm, "layer0_post_norm1");
        }
        ggml_tensor * qkv = nullptr;
#ifdef BMO_JETSON
        if (model.temporal_layers[layer].weight && model.temporal_layers[layer].weight->type == GGML_TYPE_Q4_0 && n_token == 1) {
            qkv = apply_dense_q4_0_linear_gpu(ctx, wctx, model.temporal_layers[layer].weight, x_norm);
        } else
#endif
        {
            qkv = apply_linear_with_transient_unpack(
                ctx,
                model,
                wctx,
                x_norm,
                {
                    base + "_self_attn_in_proj_weight",
                    base + "_self_attn_in_proj",
                });
        }
        bmo_h3_tap(layer, "T4", qkv);

        // Diagnostic for the K=0 bug (Phase 4.4): on layer 0, print the input
        // (x_norm) and the QKV output to pinpoint whether zeros come from the
        // input, the linear matmul, or somewhere downstream.
        if (layer == 0 && getenv("BMO_LOG_LIN")) {
#ifdef BMO_JETSON
            cudaStreamSynchronize(0);
#endif
            auto rms = [](const float * p, int n) {
                if (!p) return -1.0f;
                double s = 0.0;
                for (int i = 0; i < n; ++i) s += (double) p[i] * (double) p[i];
                return (float) std::sqrt(s / (double) n);
            };
            const float * xn = x_norm && x_norm->data ? (const float *) x_norm->data : nullptr;
            const float * qv = qkv && qkv->data ? (const float *) qkv->data : nullptr;
            const int q_off = (int) (ctx.n_heads * ctx.head_dim); // 4096
            const int kv_off = q_off + (int) ((qkv ? (int) qkv->ne[0] : 0) - q_off) / 2; // start of V
            fprintf(stderr,
                    "[bmo_lin] L0 x_norm[0..3]=%.4f %.4f %.4f %.4f rms=%.4f\n",
                    xn ? xn[0] : 0.f, xn ? xn[1] : 0.f, xn ? xn[2] : 0.f, xn ? xn[3] : 0.f,
                    rms(xn, ctx.n_embd));
            if (qv) {
                fprintf(stderr,
                        "[bmo_lin] L0 qkv qw=%lld Q[0..3]=%.4f %.4f %.4f %.4f rms_Q=%.4f K[0..3]=%.4f %.4f %.4f %.4f rms_K=%.4f V[0..3]=%.4f %.4f %.4f %.4f rms_V=%.4f\n",
                        (long long) qkv->ne[0],
                        qv[0], qv[1], qv[2], qv[3], rms(qv, q_off),
                        qv[q_off+0], qv[q_off+1], qv[q_off+2], qv[q_off+3], rms(qv + q_off, kv_off - q_off),
                        qv[kv_off+0], qv[kv_off+1], qv[kv_off+2], qv[kv_off+3], rms(qv + kv_off, kv_off - q_off));
                auto mean_abs = [](const float * p, int n) -> float {
                    if (!p || n <= 0) return -1.0f;
                    double s = 0.0;
                    for (int i = 0; i < n; ++i) s += (double) std::fabs((double) p[i]);
                    return (float) (s / (double) n);
                };
                const float ma_q = mean_abs(qv, q_off);
                const float ma_x = mean_abs(xn, ctx.n_embd);
                if (ma_x > 1e-8f && ma_q < 0.5f * ma_x) {
                    fprintf(stderr,
                            "[bmo_lin] WARNING: Q slice mean|.| suspiciously low vs x_norm (mean|Q|=%.6f "
                            "mean|x_norm|=%.6f); check fused matvec / packed weights path.\n",
                            ma_q, ma_x);
                }
            }
        }

            const int64_t q_dim = (int64_t) ctx.n_heads * ctx.head_dim;
            const int64_t qkv_w = qkv->ne[0];
            const bool qkv_valid = (qkv_w > q_dim) && ((qkv_w - q_dim) % 2 == 0);

            if (!qkv_valid) {
                throw std::runtime_error("layer=" + std::to_string(layer) + " invalid qkv width " + std::to_string((long long) qkv_w) + " (q_dim=" + std::to_string((long long) q_dim) + ")");
            }

            const int64_t kv_dim = (qkv_w - q_dim) / 2;
            const int32_t n_kv_heads = (int32_t) (kv_dim / ctx.head_dim);
            if (n_kv_heads <= 0) {
                throw std::runtime_error("layer=" + std::to_string(layer) + " invalid n_kv_heads=" + std::to_string(n_kv_heads) + " from qkv width " + std::to_string((long long) qkv_w));
            }

            const size_t e = qkv->nb[0];
            const size_t nb1_q = (size_t) ctx.head_dim * e;
            const size_t nb2_qkv = qkv->nb[1];

            ggml_tensor * q_raw = ggml_view_3d(wctx, qkv, ctx.head_dim, ctx.n_heads, n_token, nb1_q, nb2_qkv, 0);
            ggml_tensor * k_raw = ggml_view_3d(wctx, qkv, ctx.head_dim, n_kv_heads, n_token, nb1_q, nb2_qkv, (size_t) q_dim * e);
            ggml_tensor * v_raw = ggml_view_3d(wctx, qkv, ctx.head_dim, n_kv_heads, n_token, nb1_q, nb2_qkv, (size_t) (q_dim + kv_dim) * e);
#ifdef BMO_JETSON
            if (qkv->extra) {
                q_raw->extra = (uint8_t *) qkv->extra + 0;
                k_raw->extra = (uint8_t *) qkv->extra + (size_t) q_dim * e;
                v_raw->extra = (uint8_t *) qkv->extra + (size_t) (q_dim + kv_dim) * e;
            }
#endif

            if (layer == 0 && getenv("BMO_DUMP_LAYER0")) {
                bmo_dump_tensor_f32(q_raw, "layer0_q_pre_rope");
                bmo_dump_tensor_f32(k_raw, "layer0_k_pre_rope");
                bmo_dump_tensor_f32(v_raw, "layer0_v");
            }

#ifdef BMO_JETSON
            ggml_tensor * q_rope = apply_rope_gpu_interleaved(ctx, wctx, q_raw, pos);
            ggml_tensor * k_rope = apply_rope_gpu_interleaved(ctx, wctx, k_raw, pos);
#else
            ggml_tensor * q_rope = ggml_rope(wctx, q_raw, pos, ctx.head_dim, GGML_ROPE_TYPE_NORMAL);
            ggml_tensor * k_rope = ggml_rope(wctx, k_raw, pos, ctx.head_dim, GGML_ROPE_TYPE_NORMAL);
#endif

            if (layer == 0 && getenv("BMO_DUMP_LAYER0")) {
                bmo_dump_tensor_f32(q_rope, "layer0_q_post_rope");
                bmo_dump_tensor_f32(k_rope, "layer0_k_post_rope");
            }

            ggml_tensor * attn_2d = nullptr;
#ifdef BMO_JETSON
            // Single-token decode: dispatch directly to GPU decode attention kernel
            // when device pointers are cached, completely eliminating the CPU-GPU stream
            // sync bubble (saves ~45 ms across 32 layers). Fall back to eager CPU attention
            // if device pointers are absent.
            if (n_token == 1) {
                if (ctx.k_cache && ctx.k_cache->extra && ctx.v_cache && ctx.v_cache->extra &&
                    q_rope && q_rope->extra && k_rope && k_rope->extra && v_raw && v_raw->extra) {
                    staging_slot attn_slot = borrow_staging(ctx.staging);
                    launch_decode_attention(
                        (const float *) q_rope->extra,
                        (const float *) k_rope->extra,
                        (const float *) v_raw->extra,
                        ctx.k_cache->extra,
                        ctx.v_cache->extra,
                        (float *) attn_slot.dev,
                        ctx.head_dim, ctx.n_heads, n_kv_heads, ctx.n_ctx,
                        n_past, layer, ctx.stream, (const int *) ctx.pos_dev);
                    attn_slot.pool = nullptr;
                    ggml_tensor * attn_3d = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, ctx.head_dim, ctx.n_heads, 1);
                    attn_3d->data = attn_slot.host;
                    attn_3d->extra = attn_slot.dev;
                    attn_2d = ggml_reshape_2d(wctx, attn_3d, ctx.n_embd, n_token);
                    attn_2d->extra = attn_3d->extra;
                } else {
                    cudaStreamSynchronize(0); // ensure RoPE / qkv kernels finished
                    ggml_tensor * attn_3d = apply_attention_eager_decode(
                        ctx, wctx, q_rope, k_rope, v_raw, n_past, layer);
                    attn_2d = ggml_reshape_2d(wctx, attn_3d, ctx.n_embd, n_token);
                    attn_2d->extra = attn_3d->extra;
                }
                bmo_h3_tap(layer, "T5", attn_2d);
            }
#endif
            if (!attn_2d) {
                // Multi-token prefill (or non-Jetson): keep the lazy graph path.
                ggml_tensor * q_trans = ggml_permute(wctx, q_rope, 0, 2, 1, 3);
                ggml_tensor * k_trans = ggml_permute(wctx, k_rope, 0, 2, 1, 3);
                ggml_tensor * v_trans = ggml_permute(wctx, v_raw,  0, 2, 1, 3);

                ggml_tensor * k_layer = ggml_view_3d(
                    wctx, ctx.k_cache, ctx.head_dim, ctx.n_ctx, n_kv_heads,
                    ctx.k_cache->nb[1], ctx.k_cache->nb[2], (size_t) layer * ctx.k_cache->nb[3]);

                ggml_tensor * v_layer = ggml_view_3d(
                    wctx, ctx.v_cache, ctx.head_dim, ctx.n_ctx, n_kv_heads,
                    ctx.v_cache->nb[1], ctx.v_cache->nb[2], (size_t) layer * ctx.v_cache->nb[3]);

                ggml_tensor * k_slot = ggml_view_3d(
                    wctx, k_layer, ctx.head_dim, n_token, n_kv_heads,
                    k_layer->nb[1], k_layer->nb[2], (size_t) n_past * k_layer->nb[1]);

                ggml_tensor * v_slot = ggml_view_3d(
                    wctx, v_layer, ctx.head_dim, n_token, n_kv_heads,
                    v_layer->nb[1], v_layer->nb[2], (size_t) n_past * v_layer->nb[1]);

                ggml_tensor * k_write = ggml_cpy(wctx, k_trans, k_slot);
                ggml_tensor * v_write = ggml_cpy(wctx, v_trans, v_slot);
                ggml_build_forward_expand(gf, k_write);
                ggml_build_forward_expand(gf, v_write);

                const int64_t kv_len = n_past + n_token;
                ggml_tensor * k_hist = ggml_view_3d(wctx, k_layer, ctx.head_dim, kv_len, n_kv_heads, k_layer->nb[1], k_layer->nb[2], 0);
                ggml_tensor * v_hist = ggml_view_3d(wctx, v_layer, ctx.head_dim, kv_len, n_kv_heads, v_layer->nb[1], v_layer->nb[2], 0);

#ifdef BMO_JETSON
                cudaStreamSynchronize(0); // Sync before CPU Attention
#endif
                ggml_tensor * attn_heads = ggml_flash_attn_ext(
                    wctx, q_trans, ggml_cont(wctx, k_hist), ggml_cont(wctx, v_hist), nullptr,
                    1.0f / std::sqrt((float) ctx.head_dim), 0.0f, 0.0f);

                ggml_tensor * attn_trans_out = ggml_permute(wctx, attn_heads, 0, 2, 1, 3);
                ggml_tensor * attn_cont      = ggml_cont(wctx, attn_trans_out);
                attn_2d                      = ggml_reshape_2d(wctx, attn_cont, ctx.n_embd, n_token);
                bmo_h3_tap(layer, "T5", attn_2d);
            }

            ggml_tensor * attn_out = nullptr;
#ifdef BMO_JETSON
            if (model.temporal_layers[layer].wo && model.temporal_layers[layer].wo->type == GGML_TYPE_Q4_0 && n_token == 1) {
                attn_out = apply_dense_q4_0_linear_gpu(ctx, wctx, model.temporal_layers[layer].wo, attn_2d);
            } else
#endif
            {
                attn_out = apply_linear_with_transient_unpack(
                    ctx,
                    model,
                    wctx,
                    attn_2d,
                    {
                        base + "_self_attn_out_proj_weight",
                        base + "_self_attn_out_proj",
                    });
            }

            if (layer == 0 && getenv("BMO_DUMP_LAYER0")) {
                bmo_dump_tensor_f32(attn_out, "layer0_post_attn");
            }
            if (dump_this_layer) {
                bmo_dump_tensor_f32(
                    attn_out,
                    ("layer" + std::to_string(layer) + "_post_attn").c_str(),
                    bmo_dump_kind::Deep);
            }

#ifdef BMO_JETSON
            x = apply_residual_gpu(ctx, wctx, S(residual), S(attn_out));
#else
            x = ggml_add(wctx, S(residual), S(attn_out));
#endif
            bmo_h3_tap(layer, "T1", x);
            if (dump_this_layer) {
                bmo_dump_tensor_f32(
                    x,
                    ("layer" + std::to_string(layer) + "_post_attn_residual").c_str(),
                    bmo_dump_kind::Deep);
            }

        // -------- Feed-forward block --------
        ggml_tensor * ff_residual = x;
#ifdef BMO_JETSON
        ggml_tensor * ff_norm;
        if (model.temporal_layers[layer].norm2_weight) {
            ff_norm = apply_rmsnorm_gpu(ctx, wctx, x, model.temporal_layers[layer].norm2_weight, ctx.norm_eps);
        } else {
            ff_norm = ggml_rms_norm(wctx, x, ctx.norm_eps);
        }
#else
        ggml_tensor * ff_norm = ggml_rms_norm(wctx, x, ctx.norm_eps);
        if (model.temporal_layers[layer].norm2_weight) {
            ff_norm = ggml_mul(wctx, ff_norm, model.temporal_layers[layer].norm2_weight);
        }
#endif
        if (layer == 0 && getenv("BMO_DUMP_LAYER0")) {
            bmo_dump_tensor_f32(ff_norm, "layer0_post_norm2");
        }
        ggml_tensor * gating_in_w = model.temporal_layers[layer].ffn_in;
        if (!gating_in_w) {
            gating_in_w = get_tensor(model.wctx, base + "_gating_linear_in_weight");
        }
        if (!gating_in_w) {
            gating_in_w = get_tensor(model.wctx, base + "_gating_linear_in");
        }

        ggml_tensor * ff_in = nullptr;
#ifdef BMO_JETSON
        if (gating_in_w && gating_in_w->type == GGML_TYPE_Q4_0 && n_token == 1) {
            ff_in = apply_dense_q4_0_linear_gpu(ctx, wctx, gating_in_w, ff_norm);
        } else
#endif
        if (gating_in_w && !model.temporal_layers[layer].packed_weights) {
            ff_in = ggml_mul_mat(wctx, gating_in_w, ff_norm);
        } else {
            ff_in = apply_linear_with_transient_unpack(
                ctx,
                model,
                wctx,
                ff_norm,
                {
                    base + "_gating_linear_in_weight",
                    base + "_gating_linear_in",
                });
        }

        if (ff_in->ne[0] <= 0 || (ff_in->ne[0] % 2) != 0) {
            throw std::runtime_error("layer=" + std::to_string(layer) + " invalid ff_in width " + std::to_string((long long) ff_in->ne[0]));
        } else {
            // SwiGLU fused projection: split ff_in into gate and up projections
            const int64_t hidden_dim = ff_in->ne[0] / 2;
#ifdef BMO_JETSON
            // SwiGLU is fused on the GPU (apply_swiglu_gpu); the kernel chain
            // stays on stream 0 so no boundary sync is needed before this point.
            ggml_tensor * ff_act = (n_token == 1)
                ? apply_swiglu_gpu(ctx, wctx, ff_in)
                : ggml_mul(wctx, ggml_silu(wctx, ggml_view_2d(wctx, ff_in, hidden_dim, ff_in->ne[1], ff_in->nb[1], 0)), ggml_view_2d(wctx, ff_in, hidden_dim, ff_in->ne[1], ff_in->nb[1], hidden_dim * ggml_type_size(ff_in->type)));
            (void) hidden_dim;
#else
            ggml_tensor * ff_gate = ggml_view_2d(wctx, ff_in, hidden_dim, ff_in->ne[1], ff_in->nb[1], 0);
            ggml_tensor * ff_up   = ggml_view_2d(wctx, ff_in, hidden_dim, ff_in->ne[1], ff_in->nb[1], hidden_dim * ggml_type_size(ff_in->type));
            ggml_tensor * ff_act  = ggml_mul(wctx, ggml_silu(wctx, ff_gate), ff_up);
#endif

            ggml_tensor * gating_out_w = model.temporal_layers[layer].ffn_out;
            if (!gating_out_w) {
                gating_out_w = get_tensor(model.wctx, base + "_gating_linear_out_weight");
            }
            if (!gating_out_w) {
                gating_out_w = get_tensor(model.wctx, base + "_gating_linear_out");
            }

            ggml_tensor * ff_out = nullptr;
#ifdef BMO_JETSON
            if (gating_out_w && gating_out_w->type == GGML_TYPE_Q4_0 && n_token == 1) {
                ff_out = apply_dense_q4_0_linear_gpu(ctx, wctx, gating_out_w, ff_act);
            } else
#endif
            if (gating_out_w && !model.temporal_layers[layer].packed_weights) {
                ff_out = ggml_mul_mat(wctx, gating_out_w, ff_act);
            } else {
                ff_out = apply_linear_with_transient_unpack(
                    ctx,
                    model,
                    wctx,
                    ff_act,
                    {
                        base + "_gating_linear_out_weight",
                        base + "_gating_linear_out",
                    });
            }

            if (layer == 0 && getenv("BMO_DUMP_LAYER0")) {
                bmo_dump_tensor_f32(ff_out, "layer0_post_ffn");
            }
            if (dump_this_layer) {
                bmo_dump_tensor_f32(
                    ff_out,
                    ("layer" + std::to_string(layer) + "_post_ffn").c_str(),
                    bmo_dump_kind::Deep);
            }
            bmo_h3_tap(layer, "T6", ff_out);

            if (ggml_nelements(ff_out) != ggml_nelements(ff_residual)) {
                throw std::runtime_error(
                    "layer=" + std::to_string(layer) +
                    " ff_out elements=" + std::to_string((long long) ggml_nelements(ff_out)) +
                    " residual elements=" + std::to_string((long long) ggml_nelements(ff_residual))
                );
            } else {
#ifdef BMO_JETSON
                x = apply_residual_gpu(ctx, wctx, S(ff_residual), S(ff_out));
#else
                x = ggml_add(wctx, S(ff_residual), S(ff_out));
#endif
            }
            bmo_h3_tap(layer, "T2", x);
        }

        if (layer == 0 && getenv("BMO_DUMP_LAYER0")) {
            bmo_dump_tensor_f32(x, "layer0_residual_out");
            if (getenv("BMO_DUMP_LAYER0_EXIT")) {
                std::exit(0);
            }
        }
        if (dump_this_layer) {
            bmo_dump_tensor_f32(
                x,
                ("layer" + std::to_string(layer) + "_residual_out").c_str(),
                bmo_dump_kind::Deep);
        }

        const std::string out_name = "out_layer_" + std::to_string(layer);
        ggml_set_name(x, out_name.c_str());
        ggml_build_forward_expand(gf, x);

        // ---- Per-layer residual diagnostic (DC-attractor investigation) ----
        // After this layer's attention + FFN, x is the residual stream's post-
        // layer state. On the Jetson eager path (apply_residual_gpu) the value
        // already lives in pinned-mapped memory, so we just sync the stream
        // and read x->data directly. On non-Jetson the residual is a lazy
        // ggml_add and its ->data is unset until graph_compute runs, so we
        // emit only the layer index and skip stats.
        if (getenv("BMO_LOG_LAYER_Z")) {
#ifdef BMO_JETSON
            cudaStreamSynchronize(0);
            const float * p = x ? (const float *) x->data : nullptr;
            const int64_t n = x ? ggml_nelements(x) : 0;
            if (p && n > 0) {
                double s = 0, s2 = 0;
                float mn = p[0], mx = p[0];
                for (int64_t i = 0; i < n; ++i) {
                    s  += (double) p[i];
                    s2 += (double) p[i] * (double) p[i];
                    if (p[i] < mn) mn = p[i];
                    if (p[i] > mx) mx = p[i];
                }
                const double mean = s  / (double) n;
                const double rms2 = s2 / (double) n;
                const double var  = rms2 - mean * mean;
                const double stdv = std::sqrt(std::max(0.0, var));
                const double norm = std::sqrt(s2);
                fprintf(stderr,
                    "[bmo_layer_z] L=%2d n=%lld mean=%+.5f std=%.5f |z|=%.3f min=%+.3f max=%+.3f n_past=%d\n",
                    layer, (long long) n, mean, stdv, norm, mn, mx, n_past);
            } else {
                fprintf(stderr, "[bmo_layer_z] L=%2d (no host-readable data)\n", layer);
            }
#else
            fprintf(stderr, "[bmo_layer_z] L=%2d (lazy graph; data unavailable until compute)\n", layer);
#endif
        }

#ifdef BMO_JETSON
        // All transient pinned slots used during this layer can be reclaimed
        // for the next iteration (the layer output's data still points into
        // a slot, but its contents survive until the next borrow at layer N+1
        // overwrites it -- we copy/consume it earlier in that iteration).
        release_all_staging(ctx.staging);
#endif
    }

    // ---- Public outputs for the C-API ---------------------------------------
    //
    // `out_layer_{i}` names already exist for per-layer cascade / debug paths
    // (see main.cpp). For the C-API we additionally expose:
    //   - "transformer_out": the final-normed hidden state. Moshi/Helium apply
    //     a final RMSNorm (out_norm.alpha) to the residual stream after layer
    //     n_layers-1 and feed the normalised vector into BOTH text_linear and
    //     depformer_in. Skipping the norm makes text_logits collapse to
    //     ~uniform near-zero values (top logit ~0.24 instead of ~10) and
    //     produces gibberish text + miscalibrated audio conditioning. We only
    //     apply it when the caller is running the full stack (end == n_layers).
    //   - "text_logits": text_linear @ final_norm + text_linear.bias (when
    //     present), exposed only when the model carries a text head.
    //
    // For partial cascade builds (end < n_layers, used by main.cpp's
    // per-layer validation harness) we deliberately skip the final norm so
    // bit-exact A/B against the PyTorch per-layer dump still works.
    {
        ggml_tensor * final_x = x;
        if (end == ctx.n_layers && model.out_norm_weight) {
#ifdef BMO_JETSON
            final_x = apply_rmsnorm_gpu(ctx, wctx, x, model.out_norm_weight, ctx.norm_eps);
#else
            ggml_tensor * normed = ggml_rms_norm(wctx, x, ctx.norm_eps);
            final_x = ggml_mul(wctx, normed, model.out_norm_weight);
#endif
            bmo_h3_tap_head("T7", "post_out_norm", final_x);
        }

        // Final residual stats just before LM head. Pairs with the per-layer
        // BMO_LOG_LAYER_Z trace so we can isolate whether out_norm changes
        // the DC component, RMS, or just rescales channels uniformly.
        if (getenv("BMO_LOG_LAYER_Z")) {
#ifdef BMO_JETSON
            cudaStreamSynchronize(0);
            const float * p = final_x ? (const float *) final_x->data : nullptr;
            const int64_t n = final_x ? ggml_nelements(final_x) : 0;
            if (p && n > 0) {
                double s = 0, s2 = 0;
                float mn = p[0], mx = p[0];
                for (int64_t i = 0; i < n; ++i) {
                    s  += (double) p[i];
                    s2 += (double) p[i] * (double) p[i];
                    if (p[i] < mn) mn = p[i];
                    if (p[i] > mx) mx = p[i];
                }
                const double mean = s  / (double) n;
                const double rms2 = s2 / (double) n;
                const double var  = rms2 - mean * mean;
                const double stdv = std::sqrt(std::max(0.0, var));
                const double norm = std::sqrt(s2);
                fprintf(stderr,
                    "[bmo_layer_z] OUTNORM n=%lld mean=%+.5f std=%.5f |z|=%.3f min=%+.3f max=%+.3f n_past=%d\n",
                    (long long) n, mean, stdv, norm, mn, mx, n_past);
            }
#endif
        }

        ggml_tensor * transformer_out = ggml_view_1d(wctx, final_x, ggml_nelements(final_x), 0);
        ggml_set_name(transformer_out, "transformer_out");
        ggml_build_forward_expand(gf, transformer_out);

        if (getenv("BMO_DUMP_DEEP") && end == ctx.n_layers && model.out_norm_weight) {
            bmo_dump_tensor_f32(transformer_out, "post_out_norm", bmo_dump_kind::Deep);
        }

        if (end == ctx.n_layers && model.text_linear) {
            ggml_tensor * text_logits = nullptr;
#ifdef BMO_JETSON
            if (model.text_linear->type == GGML_TYPE_Q4_0 && n_token == 1) {
                text_logits = apply_dense_q4_0_linear_gpu(ctx, wctx, model.text_linear, final_x);
            } else {
                text_logits = ggml_mul_mat(wctx, model.text_linear, final_x);
            }
#else
            text_logits = ggml_mul_mat(wctx, model.text_linear, final_x);
#endif
            if (model.text_linear_bias) {
#ifdef BMO_JETSON
                if (text_logits && text_logits->type == GGML_TYPE_F32 && model.text_linear_bias->type == GGML_TYPE_F32 &&
                    ggml_nelements(text_logits) == ggml_nelements(model.text_linear_bias)) {
                    text_logits = apply_residual_gpu(ctx, wctx, text_logits, model.text_linear_bias);
                } else {
                    text_logits = ggml_add(wctx, text_logits, model.text_linear_bias);
                }
#else
                text_logits = ggml_add(wctx, text_logits, model.text_linear_bias);
#endif
            }
            ggml_set_name(text_logits, "text_logits");
            ggml_build_forward_expand(gf, text_logits);
            bmo_h3_tap_head("T8", "final_logits", text_logits);
        }
    }

    return gf;
}

ggml_cgraph * bmo_build_depth_graph(
    bmo_context & ctx,
    bmo_model & model,
    ggml_tensor * temporal_out,
    ggml_tensor * text_tokens,
    ggml_tensor * audio_tokens,
    int codebook_step,
    int n_past) {
    if (!temporal_out) {
        throw std::runtime_error("bmo_build_depth_graph: temporal_out is null");
    }
    if (!text_tokens) {
        throw std::runtime_error("bmo_build_depth_graph: text_tokens is null");
    }
    if (!audio_tokens) {
        throw std::runtime_error("bmo_build_depth_graph: audio_tokens is null");
    }
    if (!ctx.work_ctx) {
        throw std::runtime_error("bmo_build_depth_graph: work context is not initialized");
    }
    if (model.depth_layers.size() != 6) {
        throw std::runtime_error("bmo_build_depth_graph: expected 6 depth layers");
    }
    if (codebook_step < 0 || codebook_step >= (int) model.depformer_in.size()) {
        throw std::runtime_error("bmo_build_depth_graph: invalid codebook_step " + std::to_string(codebook_step));
    }
    if (!model.depformer_in[(size_t) codebook_step]) {
        throw std::runtime_error("bmo_build_depth_graph: missing depformer_in projection for codebook_step " + std::to_string(codebook_step));
    }
    if (codebook_step > 0 && !model.audio_embs[(size_t) (codebook_step - 1)]) {
        throw std::runtime_error("bmo_build_depth_graph: missing audio embedding table for previous codebook_step " + std::to_string(codebook_step - 1));
    }
    if (!model.text_emb) {
        throw std::runtime_error("bmo_build_depth_graph: missing text embedding table");
    }

    constexpr int64_t hidden_dim = 1024;
    constexpr int64_t depth_num_heads = 16;
    if ((hidden_dim % depth_num_heads) != 0) {
        throw std::runtime_error("bmo_build_depth_graph: hidden_dim must be divisible by depth_num_heads");
    }
    const int64_t head_dim = hidden_dim / depth_num_heads;

    ggml_context * wctx = ctx.work_ctx;
    ggml_cgraph * gf = ggml_new_graph(wctx);
    ctx.graph_uploads.clear();

    // Keep the depth stack causally aligned with the temporal stack interface.
    (void) n_past;

    const int64_t n_token = temporal_out->ne[1];
    const bool debug_step0 = (codebook_step == 0);
    const int64_t step_count = (int64_t) model.depformer_in.size();
    const int64_t qkv_step = 3 * hidden_dim;
    const int64_t out_step = hidden_dim;

    ggml_tensor * z_s = nullptr;
#ifdef BMO_JETSON
    if (model.depformer_in[(size_t) codebook_step]->type == GGML_TYPE_Q4_0 && n_token == 1) {
        z_s = apply_dense_q4_0_linear_gpu(ctx, wctx, model.depformer_in[(size_t) codebook_step], temporal_out);
    } else {
        z_s = ggml_mul_mat(wctx, model.depformer_in[(size_t) codebook_step], temporal_out);
    }
#else
    z_s = ggml_mul_mat(wctx, model.depformer_in[(size_t) codebook_step], temporal_out);
#endif
    if (!z_s) {
        throw std::runtime_error("bmo_build_depth_graph: failed to build depformer input projection");
    }

    ggml_tensor * x = nullptr;
#ifdef BMO_JETSON
    if (n_token == 1) {
        staging_slot last_tok_slot {};
        float * last_tok_host = nullptr;
        void * last_tok_dev = nullptr;
        if (ctx.depth_tok_emb_host && ctx.depth_tok_emb_dev) {
            last_tok_host = (float *) ctx.depth_tok_emb_host;
            last_tok_dev = ctx.depth_tok_emb_dev;
        } else {
            last_tok_slot = borrow_staging(ctx.staging);
            last_tok_host = (float *) last_tok_slot.host;
            last_tok_dev = last_tok_slot.dev;
        }
        auto unpack_emb_row = [](const ggml_tensor * t, int32_t tok, float * dst, int64_t dim) {
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
        };

        if (codebook_step == 0) {
            int32_t tok_id = *(const int32_t *) text_tokens->data;
            unpack_emb_row(model.text_emb, tok_id, last_tok_host, 1024);
        } else {
            int32_t tok_id = *(const int32_t *) audio_tokens->data;
            unpack_emb_row(model.audio_embs[(size_t) (codebook_step - 1)], tok_id, last_tok_host, 1024);
        }
        ggml_tensor * last_tok = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 1024);
        last_tok->data = last_tok_host;
        last_tok->extra = last_tok_dev;
        last_tok_slot.pool = nullptr;
        x = apply_residual_gpu(ctx, wctx, z_s, last_tok);
    } else
#endif
    {
        ggml_tensor * last_tok = nullptr;
        if (codebook_step == 0) {
            last_tok = ggml_get_rows(wctx, model.text_emb, text_tokens);
        } else {
            last_tok = ggml_get_rows(wctx, model.audio_embs[(size_t) (codebook_step - 1)], audio_tokens);
        }

        if (!last_tok) {
            throw std::runtime_error("bmo_build_depth_graph: failed to build token embeddings");
        }

        x = ggml_add(wctx, z_s, ggml_reshape_2d(wctx, last_tok, 1024, 1));
    }
    if (debug_step0) {
        ggml_set_name(x, "depth_x_init");
    }

    for (int i = 0; i < 6; ++i) {
        const std::string prefix = "depformer.layers." + std::to_string(i);
        const std::string base = "depformer_layers_" + std::to_string(i);
        const std::string step_idx = std::to_string(codebook_step);

        const std::string dot_prefix = prefix; // e.g. "depformer.layers.0"
        ggml_tensor * w_massive_in = ((size_t) i < 6) ? model.depth_in_proj[i] : nullptr;
        if (!w_massive_in) {
            std::string dot_prefix = prefix;
            w_massive_in = get_tensor(model.wctx, dot_prefix + ".self_attn.in_proj_weight");
            if (!w_massive_in) {
                std::string ug = std::string("depformer_layers_") + std::to_string(i) + "_self_attn_in_proj_weight";
                w_massive_in = get_tensor(model.wctx, ug);
            }
        }

        ggml_tensor * w_massive_out = ((size_t) i < 6) ? model.depth_out_proj[i] : nullptr;
        if (!w_massive_out) {
            std::string dot_prefix = prefix;
            w_massive_out = get_tensor(model.wctx, dot_prefix + ".self_attn.out_proj.weight");
            if (!w_massive_out) {
                std::string ug_out = std::string("depformer_layers_") + std::to_string(i) + "_self_attn_out_proj_weight";
                w_massive_out = get_tensor(model.wctx, ug_out);
            }
        }

        if (!w_massive_in || !w_massive_out) {
            throw std::runtime_error("bmo_build_depth_graph: missing shared attention weight tensors for layer " + std::to_string(i));
        }

        ggml_tensor * w_slice = ggml_view_2d(
            wctx,
            w_massive_in,
            hidden_dim,
            qkv_step,
            w_massive_in->nb[1],
            (size_t) codebook_step * qkv_step * w_massive_in->nb[1]);
#ifdef BMO_JETSON
        if (w_massive_in && w_massive_in->extra) {
            w_slice->extra = (uint8_t *) w_massive_in->extra + (size_t) codebook_step * qkv_step * w_massive_in->nb[1];
        }
#endif

        ggml_tensor * w_out_slice = ggml_view_2d(
            wctx,
            w_massive_out,
            hidden_dim,
            out_step,
            w_massive_out->nb[1],
            (size_t) codebook_step * out_step * w_massive_out->nb[1]);
#ifdef BMO_JETSON
        if (w_massive_out && w_massive_out->extra) {
            w_out_slice->extra = (uint8_t *) w_massive_out->extra + (size_t) codebook_step * out_step * w_massive_out->nb[1];
        }
#endif

        // Shared attention block.
        ggml_tensor * residual = x;
        if (!model.depth_layers[(size_t) i].norm1_weight) {
            throw std::runtime_error("bmo_build_depth_graph: missing norm1_weight for depth layer " + std::to_string(i));
        }
#ifdef BMO_JETSON
        if (n_token == 1 && w_slice->type == GGML_TYPE_Q4_0 && w_out_slice->type == GGML_TYPE_Q4_0) {
            ggml_tensor * residual = x;
            ggml_tensor * x_norm = apply_rmsnorm_gpu(ctx, wctx, x, model.depth_layers[(size_t) i].norm1_weight, ctx.norm_eps);
            ggml_tensor * qkv = apply_dense_q4_0_linear_gpu(ctx, wctx, w_slice, x_norm);

            const int64_t q_dim = hidden_dim;
            const int64_t kv_dim = hidden_dim;
            const size_t e = qkv->nb[0];
            const size_t nb1_q = (size_t) head_dim * e;
            const size_t nb2_qkv = qkv->nb[1];

            ggml_tensor * q_raw = ggml_view_3d(wctx, qkv, head_dim, depth_num_heads, n_token, nb1_q, nb2_qkv, 0);
            ggml_tensor * k_raw = ggml_view_3d(wctx, qkv, head_dim, depth_num_heads, n_token, nb1_q, nb2_qkv, (size_t) q_dim * e);
            ggml_tensor * v_raw = ggml_view_3d(wctx, qkv, head_dim, depth_num_heads, n_token, nb1_q, nb2_qkv, (size_t) (q_dim + kv_dim) * e);
            if (qkv->extra) {
                q_raw->extra = (uint8_t *) qkv->extra + 0;
                k_raw->extra = (uint8_t *) qkv->extra + (size_t) q_dim * e;
                v_raw->extra = (uint8_t *) qkv->extra + (size_t) (q_dim + kv_dim) * e;
            }

            ggml_tensor * attn_2d = nullptr;
#ifdef BMO_JETSON
            if (ctx.depth_k_cache && ctx.depth_k_cache->extra && ctx.depth_v_cache && ctx.depth_v_cache->extra &&
                q_raw && q_raw->extra && k_raw && k_raw->extra && v_raw && v_raw->extra) {
                staging_slot attn_slot = borrow_staging(ctx.staging);
                launch_decode_attention(
                    (const float *) q_raw->extra,
                    (const float *) k_raw->extra,
                    (const float *) v_raw->extra,
                    ctx.depth_k_cache->extra,
                    ctx.depth_v_cache->extra,
                    (float *) attn_slot.dev,
                    ctx.depth_head_dim, ctx.depth_n_heads, ctx.depth_n_heads, ctx.depth_n_ctx,
                    codebook_step, i, ctx.stream, nullptr);
                attn_slot.pool = nullptr;
                ggml_tensor * attn_3d = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, ctx.depth_head_dim, ctx.depth_n_heads, 1);
                attn_3d->data = attn_slot.host;
                attn_3d->extra = attn_slot.dev;
                attn_2d = ggml_reshape_2d(wctx, attn_3d, hidden_dim, n_token);
                attn_2d->extra = attn_3d->extra;
            } else {
                cudaStreamSynchronize(0); // Ensure QKV kernels finished before CPU attention
                ggml_tensor * attn_3d = apply_depth_attention_eager(
                    ctx, wctx, q_raw, k_raw, v_raw, codebook_step, i);
                attn_2d = ggml_reshape_2d(wctx, attn_3d, hidden_dim, n_token);
                attn_2d->extra = attn_3d->extra;
            }
#else
            ggml_tensor * attn_3d = apply_depth_attention_eager(
                ctx, wctx, q_raw, k_raw, v_raw, codebook_step, i);
            attn_2d = ggml_reshape_2d(wctx, attn_3d, hidden_dim, n_token);
#endif

            ggml_tensor * attn_out = apply_dense_q4_0_linear_gpu(ctx, wctx, w_out_slice, attn_2d);
            x = apply_residual_gpu(ctx, wctx, residual, attn_out);

            // FFN
            ggml_tensor * ff_residual = x;
            ggml_tensor * ff_norm = apply_rmsnorm_gpu(ctx, wctx, x, model.depth_layers[(size_t) i].norm2_weight, ctx.norm_eps);

            ggml_tensor * depth_gating_in_w = ((size_t) i < 6 && (size_t) codebook_step < (size_t) DEP_Q)
                ? model.depth_gating_in[i][codebook_step] : nullptr;
            if (!depth_gating_in_w) {
                std::string in_key = base + "_gating_" + step_idx + "_linear_in_weight";
                depth_gating_in_w = get_tensor(model.wctx, in_key);
                if (!depth_gating_in_w) {
                    depth_gating_in_w = get_tensor(model.wctx, base + "_gating_" + step_idx + "_linear_in");
                }
            }

            ggml_tensor * ff_in = nullptr;
            if (depth_gating_in_w && depth_gating_in_w->type == GGML_TYPE_Q4_0) {
                ff_in = apply_dense_q4_0_linear_gpu(ctx, wctx, depth_gating_in_w, ff_norm);
            } else {
                ff_in = apply_linear_with_transient_unpack(
                    ctx,
                    model,
                    wctx,
                    ff_norm,
                    {
                        base + "_gating_" + step_idx + "_linear_in_weight",
                        base + "_gating_" + step_idx + "_linear_in",
                    });
            }

            ggml_tensor * ff_act = apply_swiglu_gpu(ctx, wctx, ff_in);

            ggml_tensor * depth_gating_out_w = ((size_t) i < 6 && (size_t) codebook_step < (size_t) DEP_Q)
                ? model.depth_gating_out[i][codebook_step] : nullptr;
            if (!depth_gating_out_w) {
                std::string out_key = base + "_gating_" + step_idx + "_linear_out_weight";
                depth_gating_out_w = get_tensor(model.wctx, out_key);
                if (!depth_gating_out_w) {
                    depth_gating_out_w = get_tensor(model.wctx, base + "_gating_" + step_idx + "_linear_out");
                }
            }

            ggml_tensor * ff_out = nullptr;
            if (depth_gating_out_w && depth_gating_out_w->type == GGML_TYPE_Q4_0) {
                ff_out = apply_dense_q4_0_linear_gpu(ctx, wctx, depth_gating_out_w, ff_act);
            } else {
                ff_out = apply_linear_with_transient_unpack(
                    ctx,
                    model,
                    wctx,
                    ff_act,
                    {
                        base + "_gating_" + step_idx + "_linear_out_weight",
                        base + "_gating_" + step_idx + "_linear_out",
                    });
            }

            x = apply_residual_gpu(ctx, wctx, ff_residual, ff_out);
            release_all_staging(ctx.staging);
            continue;
        }
#endif
        // Stay on the LAZY GGML RMSNorm path even on Jetson: the depth path's
        // x feeding this norm is itself lazy (ggml_add of z_s + last_tok at
        // i==0, ggml_add of residual + attn_out for i>0), so apply_rmsnorm_gpu
        // would memcpy uninitialised x->data at graph-build time. Letting
        // graph_compute order this with the upstream adds is correct and the
        // depth tier is small enough that the GPU detour wouldn't help much
        // anyway.
        ggml_tensor * x_norm = ggml_rms_norm(wctx, x, ctx.norm_eps);
        x_norm = ggml_mul(wctx, x_norm, model.depth_layers[(size_t) i].norm1_weight);
        if (debug_step0 && i == 0) {
            ggml_set_name(x_norm, "depth_x_norm");
            ggml_build_forward_expand(gf, x_norm);
        }

        ggml_tensor * qkv = ggml_mul_mat(wctx, S(w_slice), S(x_norm));
        if (debug_step0 && i == 0) {
            ggml_set_name(qkv, "depth_qkv_raw");
            ggml_build_forward_expand(gf, qkv);
        }

        const int64_t q_dim = hidden_dim;
        const int64_t kv_dim = hidden_dim;
        const int64_t qkv_w = qkv->ne[0];
        if (qkv_w != q_dim + kv_dim + kv_dim) {
            throw std::runtime_error(
                "bmo_build_depth_graph: layer=" + std::to_string(i) +
                " invalid qkv width " + std::to_string((long long) qkv_w) +
                " (expected " + std::to_string((long long) (q_dim + 2 * kv_dim)) + ")");
        }

        const int32_t n_kv_heads = (int32_t) (kv_dim / head_dim);
        if (n_kv_heads <= 0) {
            throw std::runtime_error("bmo_build_depth_graph: invalid n_kv_heads for layer " + std::to_string(i));
        }

        const size_t e = qkv->nb[0];
        const size_t nb1_q = (size_t) head_dim * e;
        const size_t nb2_qkv = qkv->nb[1];

        ggml_tensor * q_raw = ggml_view_3d(wctx, qkv, head_dim, depth_num_heads, n_token, nb1_q, nb2_qkv, 0);
        ggml_tensor * k_raw = ggml_view_3d(wctx, qkv, head_dim, n_kv_heads, n_token, nb1_q, nb2_qkv, (size_t) q_dim * e);
        ggml_tensor * v_raw = ggml_view_3d(wctx, qkv, head_dim, n_kv_heads, n_token, nb1_q, nb2_qkv, (size_t) (q_dim + kv_dim) * e);

        if (debug_step0 && i == 0) {
            ggml_tensor * q_dbg = ggml_cont(wctx, q_raw);
            ggml_tensor * k_dbg = ggml_cont(wctx, k_raw);
            ggml_tensor * v_dbg = ggml_cont(wctx, v_raw);
            ggml_set_name(q_dbg, "depth_q_raw");
            ggml_set_name(k_dbg, "depth_k_raw");
            ggml_set_name(v_dbg, "depth_v_raw");
            ggml_build_forward_expand(gf, q_dbg);
            ggml_build_forward_expand(gf, k_dbg);
            ggml_build_forward_expand(gf, v_dbg);
        }

        // Permute (head_dim, n_heads, n_token) -> (head_dim, n_token, n_heads)
        // so the layout matches the depth KV cache view (ne[0]=head_dim,
        // ne[1]=n_token-or-cb_position, ne[2]=n_heads).
        // Kyutai Moshi specifies depformer_pos_emb: "none", so no RoPE is applied.
        ggml_tensor * q_trans = ggml_permute(wctx, q_raw, 0, 2, 1, 3);
        ggml_tensor * k_trans = ggml_permute(wctx, k_raw, 0, 2, 1, 3);
        ggml_tensor * v_trans = ggml_permute(wctx, v_raw,  0, 2, 1, 3);

        // Cross-codebook depth KV cache. Written at slot codebook_step on
        // every call; reset to zero at codebook_step==0 by bmo_reset_depth_kv.
        // Layout matches the temporal cache: [head_dim, depth_n_ctx, n_heads,
        // depth_n_layers] FP16. We slice the per-layer 3-D view, then the
        // single-position write slot, and finally a [0..cb_index+1) history
        // view for the lazy flash_attn.
        if (!ctx.depth_k_cache || !ctx.depth_v_cache) {
            throw std::runtime_error("bmo_build_depth_graph: depth KV cache is not allocated");
        }
        if (i >= ctx.depth_n_layers) {
            throw std::runtime_error("bmo_build_depth_graph: depth layer index exceeds depth_n_layers");
        }
        if (codebook_step + n_token > ctx.depth_n_ctx) {
            throw std::runtime_error(
                "bmo_build_depth_graph: codebook_step+n_token "
                + std::to_string(codebook_step + (int) n_token)
                + " exceeds depth_n_ctx " + std::to_string(ctx.depth_n_ctx));
        }

        ggml_tensor * k_layer_view = ggml_view_3d(
            wctx, ctx.depth_k_cache,
            (int64_t) ctx.depth_head_dim, (int64_t) ctx.depth_n_ctx, (int64_t) n_kv_heads,
            ctx.depth_k_cache->nb[1], ctx.depth_k_cache->nb[2],
            (size_t) i * ctx.depth_k_cache->nb[3]);
        ggml_tensor * v_layer_view = ggml_view_3d(
            wctx, ctx.depth_v_cache,
            (int64_t) ctx.depth_head_dim, (int64_t) ctx.depth_n_ctx, (int64_t) n_kv_heads,
            ctx.depth_v_cache->nb[1], ctx.depth_v_cache->nb[2],
            (size_t) i * ctx.depth_v_cache->nb[3]);

        // Write slot for THIS step (n_token slots starting at codebook_step).
        ggml_tensor * k_slot = ggml_view_3d(
            wctx, k_layer_view,
            (int64_t) ctx.depth_head_dim, (int64_t) n_token, (int64_t) n_kv_heads,
            k_layer_view->nb[1], k_layer_view->nb[2],
            (size_t) codebook_step * k_layer_view->nb[1]);
        ggml_tensor * v_slot = ggml_view_3d(
            wctx, v_layer_view,
            (int64_t) ctx.depth_head_dim, (int64_t) n_token, (int64_t) n_kv_heads,
            v_layer_view->nb[1], v_layer_view->nb[2],
            (size_t) codebook_step * v_layer_view->nb[1]);

        // F32 -> F16 conversion happens inside ggml_cpy.
        ggml_tensor * k_write = ggml_cpy(wctx, k_trans, k_slot);
        ggml_tensor * v_write = ggml_cpy(wctx, v_trans, v_slot);
        ggml_build_forward_expand(gf, k_write);
        ggml_build_forward_expand(gf, v_write);

        // Read history: positions [0, codebook_step + n_token) covers all
        // cached K/V from this and previous codebook steps in the same
        // temporal frame.
        const int64_t kv_len = (int64_t) codebook_step + (int64_t) n_token;
        ggml_tensor * k_hist = ggml_view_3d(
            wctx, k_layer_view,
            (int64_t) ctx.depth_head_dim, kv_len, (int64_t) n_kv_heads,
            k_layer_view->nb[1], k_layer_view->nb[2], 0);
        ggml_tensor * v_hist = ggml_view_3d(
            wctx, v_layer_view,
            (int64_t) ctx.depth_head_dim, kv_len, (int64_t) n_kv_heads,
            v_layer_view->nb[1], v_layer_view->nb[2], 0);

        ggml_tensor * attn_heads = ggml_flash_attn_ext(
            wctx,
            q_trans,
            k_hist,
            v_hist,
            nullptr,
            1.0f / std::sqrt((float) head_dim),
            0.0f,
            0.0f);
        ggml_flash_attn_ext_set_prec(attn_heads, GGML_PREC_F32);

        ggml_tensor * attn_trans_out = ggml_permute(wctx, attn_heads, 0, 2, 1, 3);
        ggml_tensor * attn_cont = ggml_cont(wctx, attn_trans_out);
        ggml_tensor * attn_2d = ggml_reshape_2d(wctx, attn_cont, hidden_dim, n_token);

        ggml_tensor * attn_out = ggml_mul_mat(wctx, S(w_out_slice), S(attn_2d));

        if (debug_step0 && i == 0) {
            ggml_tensor * attn_dbg = ggml_cont(wctx, attn_out);
            ggml_set_name(attn_dbg, "depth_attn_out");
            ggml_build_forward_expand(gf, attn_dbg);
        }

        // Lazy residual add (see norm1 comment for the rationale).
        x = ggml_add(wctx, residual, attn_out);
        if (debug_step0 && i == 0) {
            ggml_set_name(x, "depth_attn_x");
        }

        // Step-specific FFN block. Same lazy-only treatment as the attention
        // block above -- x at this point is a lazy ggml_add, so we cannot
        // route it through apply_rmsnorm_gpu without reading uninitialised
        // data at graph-build time.
        ggml_tensor * ff_residual = x;
        ggml_tensor * ff_norm = ggml_rms_norm(wctx, x, ctx.norm_eps);
        if (model.depth_layers[(size_t) i].norm2_weight) {
            ff_norm = ggml_mul(wctx, ff_norm, model.depth_layers[(size_t) i].norm2_weight);
        }

        std::string in_key = base + "_gating_" + step_idx + "_linear_in_weight";
        ggml_tensor * depth_gating_in_w = get_tensor(model.wctx, in_key);
        if (!depth_gating_in_w) {
            depth_gating_in_w = get_tensor(model.wctx, base + "_gating_" + step_idx + "_linear_in");
        }

        ggml_tensor * ff_in = nullptr;
        if (depth_gating_in_w) {
            ff_in = ggml_mul_mat(wctx, depth_gating_in_w, ff_norm);
        } else {
            ff_in = apply_linear_with_transient_unpack(
                ctx,
                model,
                wctx,
                ff_norm,
                {
                    in_key,
                    base + "_gating_" + step_idx + "_linear_in",
                });
        }

        if (!ff_in) {
            throw std::runtime_error("bmo_build_depth_graph: missing step-specific FFN input for layer " + std::to_string(i));
        }
        if (ff_in->ne[0] <= 0 || (ff_in->ne[0] % 2) != 0) {
            throw std::runtime_error("bmo_build_depth_graph: layer=" + std::to_string(i) + " invalid ff_in width " + std::to_string((long long) ff_in->ne[0]));
        }

        const int64_t ff_hidden = ff_in->ne[0] / 2;
        ggml_tensor * ff_gate = ggml_view_2d(wctx, ff_in, ff_hidden, ff_in->ne[1], ff_in->nb[1], 0);
        ggml_tensor * ff_up   = ggml_view_2d(wctx, ff_in, ff_hidden, ff_in->ne[1], ff_in->nb[1], ff_hidden * ggml_type_size(ff_in->type));
        ggml_tensor * ff_act  = ggml_mul(wctx, ggml_silu(wctx, ff_gate), ff_up);

        std::string out_key = base + "_gating_" + step_idx + "_linear_out_weight";
        ggml_tensor * depth_gating_out_w = get_tensor(model.wctx, out_key);
        if (!depth_gating_out_w) {
            depth_gating_out_w = get_tensor(model.wctx, base + "_gating_" + step_idx + "_linear_out");
        }

        ggml_tensor * ff_out = nullptr;
        if (depth_gating_out_w) {
            ff_out = ggml_mul_mat(wctx, depth_gating_out_w, ff_act);
        } else {
            ff_out = apply_linear_with_transient_unpack(
                ctx,
                model,
                wctx,
                ff_act,
                {
                    out_key,
                    base + "_gating_" + step_idx + "_linear_out",
                });
        }

        if (!ff_out) {
            throw std::runtime_error("bmo_build_depth_graph: missing step-specific FFN output for layer " + std::to_string(i));
        }
        if (ggml_nelements(ff_out) != ggml_nelements(ff_residual)) {
            throw std::runtime_error(
                "bmo_build_depth_graph: layer=" + std::to_string(i) +
                " ff_out elements=" + std::to_string((long long) ggml_nelements(ff_out)) +
                " residual elements=" + std::to_string((long long) ggml_nelements(ff_residual))
            );
        }

        // Lazy FFN residual (see norm1 comment for the rationale).
        x = ggml_add(wctx, ff_residual, ff_out);

#ifdef BMO_JETSON
        // Reclaim all transient pinned slots used during this depth layer.
        release_all_staging(ctx.staging);
#endif
    }

    const std::string out_name = "depth_out_step_" + std::to_string(codebook_step);
    ggml_set_name(x, out_name.c_str());
    ggml_build_forward_expand(gf, x);

    // Audio-logits head for this codebook step. Built only when the model
    // actually carries the per-codebook output projection (linears.{k}.weight)
    // -- when it doesn't, callers can still read depth_out_step_{k} for
    // cascade-style validation runs.
    if ((size_t) codebook_step < model.audio_heads.size() && model.audio_heads[(size_t) codebook_step]) {
        ggml_tensor * head_w = model.audio_heads[(size_t) codebook_step];
        // Defensive shape check: head_w is (audio_vocab, hidden_dim) which
        // ggml stores as ne[0]=hidden_dim, ne[1]=audio_vocab.
        if (head_w->ne[0] != hidden_dim) {
            throw std::runtime_error(
                "bmo_build_depth_graph: audio_heads[" + std::to_string(codebook_step)
                + "] inner dim " + std::to_string((long long) head_w->ne[0])
                + " mismatches depth hidden_dim " + std::to_string((long long) hidden_dim));
        }
        // Normalise x to 2D [hidden_dim, n_token]. On Jetson, the residual
        // helpers may have returned a 1D tensor with hidden_dim elements;
        // on the lazy path, x is already 2D [hidden_dim, n_token]. A reshape
        // with matching nelements is just a cheap view-rewrite either way.
        if (ggml_nelements(x) != hidden_dim * n_token) {
            throw std::runtime_error(
                "bmo_build_depth_graph: depth_out element count "
                + std::to_string((long long) ggml_nelements(x))
                + " mismatches hidden_dim*n_token "
                + std::to_string((long long) (hidden_dim * n_token)));
        }
#ifdef BMO_JETSON
        ggml_tensor * audio_logits = nullptr;
        if (head_w->type == GGML_TYPE_Q4_0 && n_token == 1) {
            audio_logits = apply_dense_q4_0_linear_gpu(ctx, wctx, head_w, x);
        } else {
            ggml_tensor * depth_out_2d = ggml_reshape_2d(wctx, x, hidden_dim, n_token);
            audio_logits = ggml_mul_mat(wctx, S(head_w), S(depth_out_2d));
        }
#else
        ggml_tensor * depth_out_2d = ggml_reshape_2d(wctx, x, hidden_dim, n_token);
        ggml_tensor * audio_logits = ggml_mul_mat(wctx, S(head_w), S(depth_out_2d));
#endif
        ggml_set_name(audio_logits, "audio_logits");
        ggml_build_forward_expand(gf, audio_logits);
    }

    return gf;
}

// bmo_api.h - Stable C ABI for the BMO temporal engine.
//
// This header is the only include required by external consumers (Python via
// ctypes/cffi, other native languages, etc.). It deliberately contains no C++
// types so that libbmo.so can be loaded and used without a C++ runtime.

#ifndef BMO_API_H
#define BMO_API_H

#include <stdint.h>

// Visibility / export macro. The shared library is built with
// `-fvisibility=hidden` so every C-ABI entry point must be marked explicitly
// or it will be invisible to dlopen / ctypes.
#if defined(_WIN32)
  #if defined(BMO_BUILDING_SHARED)
    #define BMO_API __declspec(dllexport)
  #else
    #define BMO_API __declspec(dllimport)
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  #define BMO_API __attribute__((visibility("default")))
#else
  #define BMO_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque runtime handle. Callers must obtain one via bmo_init() and free it
// with bmo_free().
typedef struct bmo_handle bmo_handle_t;

// Loads a GGUF model and prepares the inference runtime. n_ctx is the maximum
// KV-cache window length (clamped internally on Jetson). Returns NULL on
// failure; bmo_last_error() is unavailable for failed init paths because the
// handle was not successfully constructed -- check stderr instead.
BMO_API bmo_handle_t * bmo_init(const char * gguf_path, int n_ctx);

// Tears down the runtime, frees the model, and releases all CUDA / pinned
// host buffers associated with the handle. Safe to call with a NULL handle.
BMO_API void bmo_free(bmo_handle_t * h);

// Resets the autoregressive position to 0 and zeroes the K/V cache so a
// fresh decode session can start. Thread-safe (acquires the handle mutex).
BMO_API void bmo_reset(bmo_handle_t * h);

// Configuration accessors. Return 0 if the handle is NULL.
BMO_API int bmo_get_n_layers     (bmo_handle_t * h);
BMO_API int bmo_get_n_embd       (bmo_handle_t * h);
BMO_API int bmo_get_n_codebooks  (bmo_handle_t * h);
BMO_API int bmo_get_dep_q        (bmo_handle_t * h);
BMO_API int bmo_get_text_vocab   (bmo_handle_t * h);
BMO_API int bmo_get_audio_vocab  (bmo_handle_t * h);
BMO_API int bmo_get_n_attn_heads (bmo_handle_t * h);
BMO_API int bmo_get_head_dim     (bmo_handle_t * h);

// Copies a slice of the temporal K-cache out as float32.
// Returns the number of floats written, or negative on error.
BMO_API int bmo_copy_k_cache_f32(
    bmo_handle_t * h,
    int layer,
    int t_start,
    int n_positions,
    float * out,
    int max_floats);

// Captures CUDA graphs for the temporal and depth passes.
// Returns 0 on success, non-zero on failure.
BMO_API int bmo_capture_graphs(bmo_handle_t * h);
BMO_API int bmo_has_cuda_graphs(bmo_handle_t * h);

// Runs the temporal transformer for a single decode step.
//
// input_tokens     : array of length num_codebooks; one int32 codebook index
//                    per audio stream for the current step.
// num_codebooks    : must be <= bmo_get_n_codebooks(h).
// pos              : autoregressive position (0-based) of this step.
// out_transformer  : caller-provided buffer of >= n_embd       floats.
// out_text_logits  : caller-provided buffer of >= text_vocab   floats.
//
// Returns 0 on success, non-zero on error; call bmo_last_error() for the
// human-readable description. Thread-safe per handle (serialised by the
// handle's mutex).
BMO_API int bmo_forward_temporal(
    bmo_handle_t * h,
    const int32_t * input_tokens,
    int num_codebooks,
    int pos,
    float * out_transformer,
    float * out_text_logits);

// Same as bmo_forward_temporal, plus optional capture of post-layer hidden states
// (graph tensors named out_layer_{L}, shape [n_embd] for single-token decode).
// If n_capture_layers > 0, capture_layers must point to n_capture_layers int32
// layer indices and capture_out must hold n_capture_layers * n_embd floats.
// If n_capture_layers == 0, capture_layers and capture_out may be NULL.
BMO_API int bmo_forward_temporal2(
    bmo_handle_t * h,
    const int32_t * input_tokens,
    int num_codebooks,
    int pos,
    float * out_transformer,
    float * out_text_logits,
    const int32_t * capture_layers,
    int n_capture_layers,
    float * capture_out);

// Runs the depformer (depth) transformer for a single codebook step.
//
// cb_index         : depformer codebook index in [0, dep_q).
// prev_token       : previously sampled token (text token at cb_index==0,
//                    audio token at cb_index>0).
// transformer_out  : caller-provided buffer of n_embd floats from the most
//                    recent bmo_forward_temporal call.
// out_audio_logits : caller-provided buffer of >= audio_vocab floats.
//
// Returns 0 on success, non-zero on error; call bmo_last_error() for the
// description. Thread-safe per handle.
//
// NOTE (Phase 4.2): symbol is exported so the Python ctypes wrapper can bind
// it, but the body is currently a stub that returns rc=10. The full
// depformer KV-cache + audio-logits head is scheduled for Phase 4.3.
BMO_API int bmo_forward_depth(
    bmo_handle_t * h,
    int cb_index,
    int32_t prev_token,
    const float * transformer_out,
    float * out_audio_logits);

// Returns a NUL-terminated string describing the most recent error on this
// handle, or NULL if there is no pending error. The pointer is owned by the
// handle and is valid until the next API call on the same handle.
BMO_API const char * bmo_last_error(bmo_handle_t * h);

// Fused Mimi RVQ decode: sums 8 vectors of dimension 512 from projected tables
BMO_API int bmo_rvq_decode(
    const int32_t * codes_dev,
    const float * proj_tables_dev,
    float * out_dev,
    void * stream);

// Fused Mimi RVQ encode: projects [512] and searches 8 codebook nearest neighbors
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
    void * stream);

#ifdef __cplusplus
}
#endif

#endif // BMO_API_H

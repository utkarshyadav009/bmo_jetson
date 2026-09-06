// bmo.h - core data structures for BMO model runtime
#pragma once
// Jetson fused matvec: per-row tier-stream bases (row_c2/c4/c8) plus an in-kernel
// in-row prefix scan; fp16 base is derived from row * blocks_per_row minus other tiers.

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

// Set depth steps and output heads to 8 for standard Moshi (was 16 in PersonaPlex)
constexpr int DEP_Q = 8;
constexpr int NUM_CODEBOOKS = 8;

// Device-side packed tensor metadata (for CUDA unpacking)
struct device_packed_t {
#ifdef BMO_JETSON
    void * host_packed_weights = nullptr;
    size_t pw_size = 0;
    void * host_packed_mask = nullptr;
    size_t pm_size = 0;
    void * host_fp16_values = nullptr;
    size_t fv_size = 0;
    int32_t n_2bit_bytes = 0;
    int32_t n_4bit_bytes = 0;
    int32_t n_8bit_bytes = 0;
    float scale_low = 1.0f;
    float scale_int4 = 1.0f;
    float scale_int8 = 1.0f;
    float zp_low = 1.5f;
    float zp_int4 = 7.5f;
    float zp_int8 = 127.5f;
    void * canonical_base = nullptr;
    void * canonical_base_host = nullptr;
    // Host pointers (for memcpy into GGML tensors / debug)
    void * canonical_pw = nullptr;
    void * canonical_pm = nullptr;
    ggml_fp16_t * canonical_fv = nullptr;
    // Device pointers (for fused kernel reads)
    void * canonical_pw_dev = nullptr;
    void * canonical_pm_dev = nullptr;
    void * canonical_fv_dev = nullptr;
    bool preloaded = false;
    // Device: tier-stream offsets at start of each row (global walk); fp16 base derived in kernel.
    int32_t * row_c2 = nullptr;
    int32_t * row_c4 = nullptr;
    int32_t * row_c8 = nullptr;
    // v5 (per-element mask): cumulative FP16 stream element offset at each row start (tier==0).
    int32_t * row_c16 = nullptr;
#else
    void * packed_weights = nullptr;      // device ptr to packed 2/4/8 streams
    void * packed_mask = nullptr;         // device ptr to tier mask (v4: uint2 per block; v5: per element)
    void * fp16_indices = nullptr;        // legacy v1 fp16 index array
    void * fp16_values = nullptr;         // device ptr to fp16 block values / legacy overrides
    int32_t * block_offset = nullptr;     // v3: per-block element offset into that block's tier stream
#endif
    int32_t rows = 0;
    int32_t cols = 0;
    int32_t block_size = 0;
    int32_t n_blocks = 0;
    int64_t n_fp16 = 0;                   // v2: fp16 values count; legacy: override count
    int32_t packing_version = 3;          // GGUF packing_version scalar (5 = per-element mask)
    bool is_blockwise = false;
    bool is_valid = false;                // flag indicating successful allocation
};

struct tensor_upload {
    ggml_tensor * tensor = nullptr;
    const void * host_data = nullptr;
};

// A single layer which may be either a temporal (multi-tier quantized)
// or a standard unquantized depth layer. Fields not used for a given
// layer type are left nullptr.
struct bmo_layer {
    std::string name;

    // Temporal multi-tier (packed) artifacts
    ggml_tensor * packed_weights = nullptr; // packed 2/4/8 stream (bytes)
    ggml_tensor * packed_mask = nullptr;    // uint8 packed mask (4 x uint2 per byte)
    ggml_tensor * scale_low = nullptr;      // scalar f32 or f16
    ggml_tensor * scale_int4 = nullptr;
    ggml_tensor * scale_int8 = nullptr;
    ggml_tensor * fp16_indices = nullptr;   // int32 indices
    ggml_tensor * fp16_values = nullptr;    // f16 or f32 values

    // Unquantized artifacts (dense weights)
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;

    // Optional attention / ffn sub-weights for downstream compute
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wo = nullptr;

    ggml_tensor * ffn_in = nullptr;
    ggml_tensor * ffn_out = nullptr;

    // Learned RMSNorm scale weights (gamma)
    ggml_tensor * norm1_weight = nullptr;  // attention pre-norm
    ggml_tensor * norm2_weight = nullptr;  // FFN pre-norm

    // Note: device-side packed pointers are now kept in the runtime registry
    // keyed by the matrix base name. Do not store device pointers here.

    bmo_layer() = default;
};

// Model container: arrays of layers and global heads/embeddings
struct bmo_model {
    std::vector<bmo_layer> temporal_layers;
    std::vector<bmo_layer> depth_layers;

    // Depth-tier (depformer_dim) audio embedding tables and depformer input
    // projections. These are consumed exclusively by bmo_build_depth_graph.
    std::vector<ggml_tensor *> audio_embs;        // depformer_emb.{k}.weight (1024-dim)
    std::vector<ggml_tensor *> depformer_in;      // depformer_in.{k}.weight
    // Per-codebook output heads: audio_logits = linears[cb_index] @ depth_out
    // Each tensor is (audio_vocab, depth_hidden_dim) -- usually (2049, 1024).
    std::vector<ggml_tensor *> audio_heads;       // linears.{k}.weight

    // Pre-resolved depth transformer weights (6 layers x 8 codebook steps)
    ggml_tensor * depth_gating_in[6][DEP_Q] = {{nullptr}};
    ggml_tensor * depth_gating_out[6][DEP_Q] = {{nullptr}};
    ggml_tensor * depth_in_proj[6] = {nullptr};
    ggml_tensor * depth_out_proj[6] = {nullptr};

    ggml_tensor * text_emb = nullptr;             // depformer_text_emb.weight (1024-dim)
    ggml_tensor * text_linear = nullptr;          // shared temporal text head
    ggml_tensor * text_linear_bias = nullptr;     // text_linear.bias (text_vocab-dim, fp32)

    // Final temporal RMSNorm gain applied to the residual stream after layer
    // n_layers-1 and before either text_linear OR depformer_in. Stored as a
    // 1D (n_embd,) tensor in the GGUF (key "out_norm_weight"; flattened from
    // PyTorch's (1,1,4096) "out_norm.alpha"). Skipping it makes text_logits
    // collapse to ~uniform near-zero values and feeds an unnormalised vector
    // into the depth conditioning -- both produce gibberish output.
    ggml_tensor * out_norm_weight = nullptr;

    // Temporal-tier (n_embd) embedding tables consumed by bmo_embed_input_tokens
    // when constructing the temporal transformer's input.
    std::vector<ggml_tensor *> temporal_audio_embs; // emb.{k}.weight (n_embd-dim)
    ggml_tensor * temporal_text_emb = nullptr;      // text_emb.weight (n_embd-dim)

    // embeddings and head tensors (may be stored as ggml tensors inside the
    // weights ggml_context that gguf provides)
    ggml_tensor * token_embedding = nullptr;
    ggml_tensor * output_head = nullptr;

    // Keep a reference to the gguf/ggml data context in which weight tensors live.
    gguf_context * gctx = nullptr;     // owns the data memory
    ggml_context * wctx = nullptr;     // ggml context used by gguf to store weight tensors

    // Optional memory-mapped GGUF backing (Linux).
    void * gguf_mmap = nullptr;
    size_t gguf_mmap_size = 0;
};

// Generic pool of pinned/mapped GPU staging slots. Each slot is a fixed-size
// page-locked block whose host and device aliases stay alive for the lifetime
// of the runtime context. Borrowed transiently inside layer iterations and
// bulk-released at the end of each iteration via release_all_staging().
struct gpu_staging_pool {
    static constexpr int    N_SLOTS    = 32;
    // Sized for the largest matvec output we alias into a slot
    // (text_linear produces 32000 floats = 128 KB on this model).
    static constexpr size_t SLOT_BYTES = 32000 * sizeof(float);

    void * host[N_SLOTS]   = {};
    void * dev[N_SLOTS]    = {};
    bool   in_use[N_SLOTS] = {};
};

// Runtime context that holds allocation contexts and runtime params (KV cache, etc.)
struct bmo_context {
    // GGML contexts
    ggml_context * kv_ctx = nullptr;   // dedicated KV cache context

    // CUDA backend (if available)
    void * cuda_backend = nullptr;     // ggml_backend_t (opaque, to avoid ggml-backend.h)
#ifndef BMO_JETSON
    void * cuda_unpack_scratch = nullptr;
    void * cuda_unpack_scratch_dev = nullptr;
    size_t cuda_unpack_scratch_bytes = 0;
    bool cuda_unpack_scratch_managed = false;
    bool cuda_unpack_scratch_owns_raw_malloc = false;
#endif
#ifdef BMO_JETSON
    void * cuda_fused_output_buffer = nullptr;
    void * cuda_fused_output_buffer_dev = nullptr;
    size_t cuda_fused_output_buffer_bytes = 0;
    bool cuda_fused_output_owns_raw_malloc = false;
    void * cuda_fused_input_buffer = nullptr;
    void * cuda_fused_input_buffer_dev = nullptr;
    size_t cuda_fused_input_buffer_bytes = 0;
    bool cuda_fused_input_owns_raw_malloc = false;
    void * streaming_big_pool = nullptr;
    size_t streaming_big_pool_size = 0;
    bool streaming_big_pool_registered = false;
    void * streaming_scalar_pool = nullptr;
    size_t streaming_scalar_pool_size = 0;
    bool streaming_scalar_pool_registered = false;
    void * q8_scratch_dev = nullptr;
#endif

    // Generic ring buffer of pinned/mapped GPU staging slots used by the fused
    // GPU op interceptors (RMSNorm, residual add, ...). Slots are borrowed
    // during a layer iteration and bulk-released at its end.
    gpu_staging_pool staging;

    // Registry mapping a matrix base name (e.g. "transformer_layers_0_self_attn_in_proj_weight")
    // to device-side packed metadata allocated by bmo_prepare_device_packed_tensors.
    std::unordered_map<std::string, device_packed_t> packed_registry;

    // Model hyperparameters filled at load time
    int32_t n_ctx = 0;
    int32_t n_layers = 0;
    int32_t n_heads = 0;
    int32_t n_embd = 0;
    int32_t head_dim = 0;

    // Vocabulary / codebook geometry parsed at load time. Populated from the
    // embedding tables in bmo_model and exposed through the C-API.
    int32_t num_codebooks    = 0; // K = n_q + 1 (text + audio) channels in temporal input
    int32_t dep_q            = 0; // count of non-null depformer input projections
    int32_t text_vocab_size  = 0; // text_linear (or text_emb) output dim
    int32_t audio_vocab_size = 0; // temporal_audio_embs[k] output dim (assumed uniform)

    // RoPE base frequency (theta). Parsed from the GGUF metadata at load time.
    float rope_theta = 10000.0f;

    // RMSNorm epsilon. Parsed from the GGUF metadata at load time (default 1e-8f for Moshi).
    float norm_eps = 1e-8f;

    // Physical KV cache tensors allocated in kv_ctx
    ggml_tensor * k_cache = nullptr;
    ggml_tensor * v_cache = nullptr;
    std::unique_ptr<uint8_t[]> kv_mem;
    bool kv_mem_registered = false;

    // Depth-transformer KV cache.
    //
    // The depth transformer is autoregressive across the codebook dimension:
    // for cb_index = k it attends to the K/V written by previous calls
    // cb_index = 0..k-1 (all within the SAME temporal frame). The cache is
    // therefore tiny (max 16 codebooks) and gets reset between temporal
    // frames via bmo_reset_depth_kv() at cb_index == 0.
    //
    // Layout matches the temporal cache: [head_dim, n_ctx, n_heads, n_layers]
    // FP16, with n_ctx = depth_n_ctx, n_heads = depth_n_heads, n_layers = 6.
    ggml_context * depth_kv_ctx = nullptr;
    ggml_tensor *  depth_k_cache = nullptr;
    ggml_tensor *  depth_v_cache = nullptr;
    std::unique_ptr<uint8_t[]> depth_kv_mem;
    bool depth_kv_mem_registered = false;
    int32_t depth_n_ctx     = 0;   // = max codebook count (typically 16 = dep_q)
    int32_t depth_n_heads   = 0;   // = 16 in Moshi/PersonaPlex depformer
    int32_t depth_head_dim  = 0;   // = 64
    int32_t depth_n_layers  = 0;   // = 6
    int32_t depth_hidden_dim = 0;  // = depth_n_heads * depth_head_dim = 1024
    size_t  depth_kv_bytes = 0;

    // Track memory usage
    size_t weights_bytes = 0;
    size_t kv_bytes = 0;
    
    // Inference compute arenas
    ggml_context * work_ctx = nullptr;
    std::vector<uint8_t> work_mem;
    std::vector<float> shared_scratch_w;
    void * stream = nullptr;           // Active CUDA stream (cudaStream_t)
    void * pos_dev = nullptr;          // Device-mapped int pointer for dynamic pos/past in CUDA graphs
    void * depth_tok_emb_host = nullptr; // Dedicated static buffer for depth token embeddings
    void * depth_tok_emb_dev = nullptr;  // Device alias for depth_tok_emb_host
    struct owned_tensor_upload {
        ggml_tensor * tensor = nullptr;
        std::vector<uint8_t> bytes;
    };
    std::vector<owned_tensor_upload> graph_uploads;
};

// Loader and allocator APIs
void bmo_load_model(const char * fname, bmo_model & model, bmo_context & ctx);
void bmo_init_kv_cache(bmo_context & ctx, int32_t n_ctx);
void bmo_prepare_device_packed_tensors(bmo_model & model, bmo_context & ctx);
void bmo_free_cuda_resources(bmo_context & ctx);
void bmo_print_mem_diag(const std::string & phase);
// Zeroes the depth-transformer KV cache. Called at cb_index == 0 of every
// new temporal frame so the depth transformer's cross-codebook attention
// starts from a clean slate.
void bmo_reset_depth_kv(bmo_context & ctx);

// Compute graph builder
struct ggml_cgraph * bmo_build_temporal_graph(
    bmo_context & ctx,
    bmo_model & model,
    struct ggml_tensor * input_tokens,
    int n_past,
    int layer_begin = 0,
    int layer_end = -1);

struct ggml_cgraph * bmo_build_depth_graph(
    bmo_context & ctx,
    bmo_model & model,
    struct ggml_tensor * temporal_out,
    struct ggml_tensor * text_tokens,
    struct ggml_tensor * audio_tokens,
    int codebook_step,
    int n_past);

void bmo_execute_graph(bmo_context & ctx, struct ggml_cgraph * gf, const std::vector<tensor_upload> & inputs = {}, bool sync_cuda = true);

// (Re)initializes ctx.work_ctx using the host-side ctx.work_mem buffer.
// Must be called before each new graph build (before bmo_embed_input_tokens
// and bmo_build_temporal_graph) so that transient ggml allocations live in
// a fresh arena.
void bmo_reset_work_ctx(bmo_context & ctx);

// Builds the input-token embedding for a single decode step using the
// temporal-tier (n_embd-wide) embedding tables. Token layout follows Moshi:
//   y[d] =  temporal_text_emb[input_tokens[0]][d]
//        + sum_{k=1..K-1} temporal_audio_embs[k-1][input_tokens[k]][d]
// where K = num_codebooks. Indices < 0 (Moshi's "no token yet" convention)
// or out-of-vocab values silently skip their channel; if all channels skip,
// returns a zeroed n_embd vector so the caller still sees a well-shaped input.
//
// Returns a graph node with shape [n_embd, 1] that can be fed directly to
// bmo_build_temporal_graph as its `input_tokens` argument. Allocates inside
// ctx.work_ctx (must already be initialized via bmo_reset_work_ctx).
struct ggml_tensor * bmo_embed_input_tokens(
    bmo_context & ctx,
    bmo_model & model,
    const int32_t * input_tokens,
    int num_codebooks);

void bmo_embed_input_tokens_into(
    const bmo_context & ctx,
    const bmo_model & model,
    const int32_t * input_tokens,
    int num_codebooks,
    float * acc);

#ifdef BMO_ENABLE_CUDA
void launch_fused_dequant_matvec(
    const void * pw,
    const void * pm,
    const void * fp16_vals,
    const int32_t * row_c2,
    const int32_t * row_c4,
    const int32_t * row_c8,
    int rows,
    int cols,
    int block_size,
    int n_2bit_bytes,
    int n_4bit_bytes,
    float scale_low,
    float scale_int4,
    float scale_int8,
    float zp_low,
    float zp_int4,
    float zp_int8,
    const float * x,
    float * y,
    void * stream = nullptr);

void launch_rmsnorm(
    const float * x_dev,
    const float * weight_dev,
    float eps,
    int n_embd,
    float * y_dev,
    void * stream = nullptr);

void launch_residual_add(
    const float * a_dev,
    const float * b_dev,
    int n,
    float * y_dev,
    void * stream = nullptr);

void launch_rope_interleaved(
    const float * x_dev,
    int n_heads,
    int head_dim,
    int n_token,
    int pos_base,
    float theta_base,
    float * y_dev,
    void * stream = nullptr,
    const int * pos_dev = nullptr);

void launch_swiglu_split(
    const float * h_dev,
    int d_ff,
    float * y_dev,
    void * stream = nullptr);

void launch_gemv_q4_0_q8_1(
    const void * W_dev,
    const float * x_dev,
    float * y_dev,
    int rows,
    int cols,
    void * q8_scratch,
    void * stream = nullptr);

void launch_decode_attention(
    const float * q_dev,
    const float * k_dev,
    const float * v_dev,
    void * k_cache_dev,
    void * v_cache_dev,
    float * out_dev,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int n_ctx,
    int n_past,
    int layer,
    void * stream = nullptr,
    const int * pos_dev = nullptr);

void launch_rvq_decode(
    const int32_t * codes_dev,
    const float * proj_tables_dev,
    float * out_dev,
    void * stream = nullptr);

void launch_rvq_encode(
    const float * in_vec_dev,
    const float * w_in0_dev,
    const float * e0_dev,
    const float * norm0_dev,
    const float * w_in_rest_dev,
    const float * e_rest_dev,
    const float * norm_rest_dev,
    int32_t * out_codes_dev,
    float * scratch_dev,
    void * stream = nullptr);
#endif

#include "bmo.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static void reset_work_ctx(bmo_context & ctx, size_t work_mem_size) {
    if (ctx.work_ctx) {
        ggml_free(ctx.work_ctx);
        ctx.work_ctx = nullptr;
    }

    ggml_init_params wp = {
        work_mem_size,
        ctx.work_mem.data(),
        false,
    };
    ctx.work_ctx = ggml_init(wp);
    if (!ctx.work_ctx) {
        throw std::runtime_error("Failed to initialize work context");
    }
}

static void copy_tensor_to_host(bmo_context & ctx, ggml_tensor * t, void * dst, size_t nbytes) {
    (void) ctx;
    if (!t || !t->data) throw std::runtime_error("Missing graph tensor or null data");
    std::memcpy(dst, t->data, nbytes);
}

static void dump_tensor(bmo_context & ctx, const char * path, ggml_tensor * t) {
    std::vector<uint8_t> host((size_t) ggml_nbytes(t));
    copy_tensor_to_host(ctx, t, host.data(), host.size());

    FILE * out_file = fopen(path, "wb");
    if (!out_file) {
        throw std::runtime_error(std::string("Failed to open ") + path + " for writing");
    }

    fwrite(host.data(), 1, host.size(), out_file);
    fclose(out_file);
}

int main(int argc, char ** argv) {
    std::string fname = "bmo_weights.gguf";
    std::string mode = "depth_cascade"; // default to depth for backwards compat
    bool debug_dumps = false;
    int n_iterations = 100;
    int n_threads = 32;
    int layer_begin = 0;
    int layer_end = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--n-iterations" && i + 1 < argc) {
            n_iterations = std::atoi(argv[++i]);
        } else if (arg == "--n-threads" && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        } else if (arg == "--layer_begin" && i + 1 < argc) {
            layer_begin = std::atoi(argv[++i]);
        } else if (arg == "--layer_end" && i + 1 < argc) {
            layer_end = std::atoi(argv[++i]);
        } else if (arg == "--debug-dumps") {
            debug_dumps = true;
        } else if (arg[0] != '-') {
            fname = arg;
        }
    }

    bmo_model model;
    bmo_context ctx;

    try {
        std::cout << "[bmo_main] === BMO Forward Pass Test ===\n";
        std::cout << "[bmo_main] Loading model from: " << fname << "\n";
        bmo_load_model(fname.c_str(), model, ctx);

        std::cout << "[bmo_main] weights_bytes = " << (double) ctx.weights_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
        std::cout << "[bmo_main] n_layers=" << ctx.n_layers << " n_heads=" << ctx.n_heads
                  << " n_embd=" << ctx.n_embd << " head_dim=" << ctx.head_dim << "\n";

        std::cout << "[bmo_main] Initializing KV cache for 2048 tokens...\n";
        bmo_init_kv_cache(ctx, 2048);
        std::cout << "[bmo_main] kv_bytes      = " << (double) ctx.kv_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";

        std::cout << "[bmo_main] Initializing compute arenas...\n";
        const size_t work_mem_size =
#ifdef BMO_TARGET_JETSON
            (size_t) 256ULL * 1024 * 1024;
#else
            (size_t) 2048ULL * 1024 * 1024;
#endif
        ctx.work_mem.resize(work_mem_size);
        std::cout << "[bmo_main] Allocated work_mem: "
                  << (double) work_mem_size / (1024.0 * 1024.0 * 1024.0) << " GB\n";

        if (mode == "temporal_cascade") {
            std::cout << "[bmo_main] Running Temporal Cascade...\n";
            
            // Initialize the cascade input with ones, matching verify_all_layers.py
            std::vector<float> current_x(ctx.n_embd, 1.0f);

            const int begin = std::max(0, layer_begin);
            const int end = (layer_end < 0) ? ctx.n_layers : std::min(layer_end, ctx.n_layers);
            for (int i = begin; i < end; ++i) {
                reset_work_ctx(ctx, work_mem_size);
                
                ggml_tensor * layer_in = ggml_new_tensor_2d(ctx.work_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
                if (!layer_in || !layer_in->data) {
                    throw std::runtime_error("temporal_cascade: failed to allocate layer_in");
                }
                // Eagerly populate layer_in->data BEFORE graph build. The Jetson
                // path runs apply_rmsnorm_gpu / apply_linear_with_transient_unpack
                // during graph construction, which read x->data immediately. The
                // tensor_upload memcpy inside bmo_execute_graph happens after build
                // and is too late on this path; on dGPU the lazy graph compute
                // hides the issue, but on Jetson it produced NaNs.
                std::memcpy(layer_in->data, current_x.data(),
                            (size_t) ctx.n_embd * sizeof(float));
                std::vector<tensor_upload> inputs = { { layer_in, current_x.data() } };

                // Build only layer i (from i to i+1)
                ggml_cgraph * gf = bmo_build_temporal_graph(ctx, model, layer_in, 0, i, i + 1);
                
                bmo_print_mem_diag("Before Graph Exec");
                bmo_execute_graph(ctx, gf, inputs);

                std::string out_name = "out_layer_" + std::to_string(i);
                ggml_tensor * layer_out = ggml_graph_get_tensor(gf, out_name.c_str());
                if (!layer_out) {
                    throw std::runtime_error("Missing graph tensor: " + out_name);
                }

                // Copy output back to current_x for the next layer's input
                copy_tensor_to_host(ctx, layer_out, current_x.data(), (size_t) ctx.n_embd * sizeof(float));

                if (debug_dumps) {
                    std::string dump_path = "cpp_out_layer_" + std::to_string(i) + ".bin";
                    dump_tensor(ctx, dump_path.c_str(), layer_out);
                    std::cout << "[bmo_main] Dumped " << dump_path << "\n";
                }
            }
            std::cout << "[SUCCESS] Temporal validation cascade completed!\n";

        } else if (mode == "depth_cascade") {
            std::cout << "[bmo_main] Running Depth Cascade (Step 0)...\n";
            reset_work_ctx(ctx, work_mem_size);

            ggml_tensor * temporal_out = ggml_new_tensor_2d(ctx.work_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
            ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx.work_ctx, GGML_TYPE_I32, 1);
            ggml_tensor * audio_tokens = ggml_new_tensor_1d(ctx.work_ctx, GGML_TYPE_I32, 1);
            if (!temporal_out || !text_tokens || !audio_tokens) {
                throw std::runtime_error("Failed to allocate depth validation tensors");
            }

            std::vector<float> temporal_host((size_t) ggml_nelements(temporal_out), 1.0f);
            int32_t text_host = 0;
            int32_t audio_host = 0;
            std::memcpy(temporal_out->data, temporal_host.data(), (size_t) ggml_nelements(temporal_out) * sizeof(float));
            *((int32_t *) text_tokens->data) = text_host;
            *((int32_t *) audio_tokens->data) = audio_host;
            std::vector<tensor_upload> inputs = {
                { temporal_out, temporal_host.data() },
                { text_tokens, &text_host },
                { audio_tokens, &audio_host },
            };

            ggml_cgraph * gf = bmo_build_depth_graph(ctx, model, temporal_out, text_tokens, audio_tokens, 0, 0);
            if (!gf) {
                throw std::runtime_error("Failed to build depth graph");
            }

            std::cout << "[bmo_main] Depth graph has " << ggml_graph_n_nodes(gf) << " nodes\n";

            bmo_print_mem_diag("Before Graph Exec");
            bmo_execute_graph(ctx, gf, inputs);

            ggml_tensor * out_tensor = ggml_graph_get_tensor(gf, "depth_out_step_0");
            if (!out_tensor) {
                throw std::runtime_error("Missing graph tensor: depth_out_step_0");
            }

            if (debug_dumps) {
                const std::string dump_path = "cpp_depth_out.bin";
                dump_tensor(ctx, dump_path.c_str(), out_tensor);
                std::cout << "[bmo_main] Dumped depth output (" << ggml_nbytes(out_tensor)
                          << " bytes) to " << dump_path << "\n";

                auto dump_named = [&](const char * name, const char * path) {
                    ggml_tensor * t = ggml_graph_get_tensor(gf, name);
                    if (!t) {
                        std::cout << "[bmo_main] Missing debug tensor: " << name << "\n";
                        return;
                    }
                    dump_tensor(ctx, path, t);
                    std::cout << "[bmo_main] Dumped " << name << " to " << path << "\n";
                };

                dump_named("depth_x_init", "cpp_depth_x_init.bin");
                dump_named("depth_x_norm", "cpp_depth_x_norm.bin");
                dump_named("depth_qkv_raw", "cpp_depth_qkv_raw.bin");
                dump_named("depth_q_raw", "cpp_depth_q_raw.bin");
                dump_named("depth_k_raw", "cpp_depth_k_raw.bin");
                dump_named("depth_v_raw", "cpp_depth_v_raw.bin");
                dump_named("depth_attn_out", "cpp_depth_attn_out.bin");
                dump_named("depth_attn_x", "cpp_depth_attn_x.bin");

                // DEBUG: Build separate graph for intermediate values
                {
                    std::cout << "[bmo_main] Computing debug intermediate values...\n";
                    reset_work_ctx(ctx, work_mem_size);
                    ggml_context * dbg_ctx = ctx.work_ctx;
                    ggml_cgraph * dbg_gf = ggml_new_graph(dbg_ctx);

                    // Recreate input tensors
                    ggml_tensor * dbg_temporal_out = ggml_new_tensor_2d(dbg_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
                    ggml_tensor * dbg_text_tokens = ggml_new_tensor_1d(dbg_ctx, GGML_TYPE_I32, 1);
                    std::vector<float> dbg_temporal_host((size_t) ggml_nelements(dbg_temporal_out), 1.0f);
                    int32_t dbg_text_host = 0;
                    std::vector<tensor_upload> dbg_inputs = {
                        { dbg_temporal_out, dbg_temporal_host.data() },
                        { dbg_text_tokens, &dbg_text_host },
                    };

                    // Compute z_s (depformer_in projection)
                    ggml_tensor * dbg_z_s = ggml_mul_mat(dbg_ctx, model.depformer_in[0], dbg_temporal_out);
                    ggml_set_name(dbg_z_s, "debug_z_s");

                    // Get text embedding
                    ggml_tensor * dbg_text_emb = ggml_get_rows(dbg_ctx, model.text_emb, dbg_text_tokens);
                    ggml_set_name(dbg_text_emb, "debug_text_emb");

                    // Add them
                    ggml_tensor * dbg_x_init = ggml_add(dbg_ctx, dbg_z_s, ggml_reshape_2d(dbg_ctx, dbg_text_emb, 1024, 1));
                    ggml_set_name(dbg_x_init, "debug_x_init");

                    ggml_build_forward_expand(dbg_gf, dbg_z_s);
                    ggml_build_forward_expand(dbg_gf, dbg_text_emb);
                    ggml_build_forward_expand(dbg_gf, dbg_x_init);

                    bmo_execute_graph(ctx, dbg_gf, dbg_inputs);

                    // Dump debug values
                    ggml_tensor * out_z_s = ggml_graph_get_tensor(dbg_gf, "debug_z_s");
                    ggml_tensor * out_text_emb = ggml_graph_get_tensor(dbg_gf, "debug_text_emb");
                    ggml_tensor * out_x_init = ggml_graph_get_tensor(dbg_gf, "debug_x_init");

                    if (out_z_s) dump_tensor(ctx, "cpp_debug_z_s.bin", out_z_s);
                    if (out_text_emb) dump_tensor(ctx, "cpp_debug_text_emb.bin", out_text_emb);
                    if (out_x_init) dump_tensor(ctx, "cpp_debug_x_init.bin", out_x_init);

                    std::cout << "[bmo_main] Dumped debug tensors (z_s, text_emb, x_init)\n";
                }
            }

            std::cout << "[SUCCESS] Depth-step 0 validation completed!\n";
        } else if (mode == "stress_test") {
            std::cout << "[bmo_main] Running Stress Test (" << n_iterations << " iterations)...\n";

            std::vector<float> temporal_state(ctx.n_embd, 1.0f);
            auto loop_start = std::chrono::steady_clock::now();
            for (int iter = 0; iter < n_iterations; ++iter) {
                // Temporal pass over all layers in full 32-layer stack.
                reset_work_ctx(ctx, work_mem_size);
                ggml_tensor * layer_in = ggml_new_tensor_2d(ctx.work_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
                if (!layer_in || !layer_in->data) {
                    throw std::runtime_error("stress_test: failed to allocate layer_in");
                }
#ifdef BMO_JETSON
                if (ctx.cuda_fused_input_buffer) {
                    std::memcpy(ctx.cuda_fused_input_buffer, temporal_state.data(),
                                (size_t) ctx.n_embd * sizeof(float));
                    layer_in->data = ctx.cuda_fused_input_buffer;
                    layer_in->extra = ctx.cuda_fused_input_buffer_dev;
                } else {
                    std::memcpy(layer_in->data, temporal_state.data(),
                                (size_t) ctx.n_embd * sizeof(float));
                }
#else
                std::memcpy(layer_in->data, temporal_state.data(),
                            (size_t) ctx.n_embd * sizeof(float));
#endif
                std::vector<tensor_upload> temporal_inputs = { { layer_in, temporal_state.data() } };

                auto build_t0_temp = std::chrono::steady_clock::now();
                ggml_cgraph * temporal_gf = bmo_build_temporal_graph(ctx, model, layer_in, 0, 0, ctx.n_layers);
                auto build_t1_temp = std::chrono::steady_clock::now();
                long build_ms_temp = std::chrono::duration_cast<std::chrono::milliseconds>(build_t1_temp - build_t0_temp).count();
                std::fprintf(stderr, "[prof_build] iter=%d phase=temporal build_ms=%ld\n", iter, build_ms_temp);

                auto t0_temp = std::chrono::steady_clock::now();
                bmo_execute_graph(ctx, temporal_gf, temporal_inputs);
                auto t1_temp = std::chrono::steady_clock::now();

                long compute_ms_temp = std::chrono::duration_cast<std::chrono::milliseconds>(t1_temp - t0_temp).count();
                std::fprintf(stderr, "[prof] iter=%d phase=temporal compute_ms=%ld\n", iter, compute_ms_temp);

                ggml_tensor * temporal_out = ggml_graph_get_tensor(temporal_gf, "transformer_out");
                if (!temporal_out) {
                    throw std::runtime_error("Stress test: missing temporal output at iteration " + std::to_string(iter));
                }
                copy_tensor_to_host(ctx, temporal_out, temporal_state.data(), (size_t) ctx.n_embd * sizeof(float));

                ggml_graph_clear(temporal_gf);
                ggml_free(ctx.work_ctx);
                ctx.work_ctx = nullptr;

                // Reset depth KV cache at start of frame
                bmo_reset_depth_kv(ctx);

                // Depth pass for each codebook step (8 steps for standard Moshi).
                // Sized to 16 MB allocated once for all depth steps to eliminate per-step arena reinit overhead.
                reset_work_ctx(ctx, 16 * 1024 * 1024);
                for (int step = 0; step < ctx.dep_q; ++step) {
                    ggml_tensor * depth_in = ggml_new_tensor_2d(ctx.work_ctx, GGML_TYPE_F32, ctx.n_embd, 1);
                    ggml_tensor * text_tokens = ggml_new_tensor_1d(ctx.work_ctx, GGML_TYPE_I32, 1);
                    ggml_tensor * audio_tokens = ggml_new_tensor_1d(ctx.work_ctx, GGML_TYPE_I32, 1);
                    if (!depth_in || !text_tokens || !audio_tokens) {
                        throw std::runtime_error("Stress test: failed to allocate depth inputs at iteration " + std::to_string(iter));
                    }

                    int32_t text_host = 0;
                    int32_t audio_host = 0;
#ifdef BMO_JETSON
                    if (ctx.cuda_fused_input_buffer) {
                        if (step == 0) {
                            std::memcpy(ctx.cuda_fused_input_buffer, temporal_state.data(), (size_t) ctx.n_embd * sizeof(float));
                        }
                        depth_in->data = ctx.cuda_fused_input_buffer;
                        depth_in->extra = ctx.cuda_fused_input_buffer_dev;
                    } else {
                        std::memcpy(depth_in->data, temporal_state.data(), (size_t) ctx.n_embd * sizeof(float));
                    }
#else
                    std::memcpy(depth_in->data, temporal_state.data(), (size_t) ctx.n_embd * sizeof(float));
#endif
                    *((int32_t *) text_tokens->data) = text_host;
                    *((int32_t *) audio_tokens->data) = audio_host;
                    std::vector<tensor_upload> depth_inputs;
                    if (step == 0) {
                        depth_inputs.push_back({ depth_in, temporal_state.data() });
                    }
                    depth_inputs.push_back({ text_tokens, &text_host });
                    depth_inputs.push_back({ audio_tokens, &audio_host });

                    auto build_t0_depth = std::chrono::steady_clock::now();
                    ggml_cgraph * depth_gf = bmo_build_depth_graph(ctx, model, depth_in, text_tokens, audio_tokens, step, 0);
                    auto build_t1_depth = std::chrono::steady_clock::now();
                    long build_ms_depth = std::chrono::duration_cast<std::chrono::milliseconds>(build_t1_depth - build_t0_depth).count();
                    std::fprintf(stderr, "[prof_build] iter=%d phase=depth step=%d build_ms=%ld\n", iter, step, build_ms_depth);

                    const bool is_last_step = (step == ctx.dep_q - 1);
                    auto t0_depth = std::chrono::steady_clock::now();
                    bmo_execute_graph(ctx, depth_gf, depth_inputs, is_last_step);
                    auto t1_depth = std::chrono::steady_clock::now();

                    long compute_ms_depth = std::chrono::duration_cast<std::chrono::milliseconds>(t1_depth - t0_depth).count();
                    std::fprintf(stderr, "[prof] iter=%d phase=depth step=%d compute_ms=%ld\n", iter, step, compute_ms_depth);

                    ggml_graph_clear(depth_gf);
                }
                ggml_free(ctx.work_ctx);
                ctx.work_ctx = nullptr;

                {
                    auto now = std::chrono::steady_clock::now();
                    long iter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - loop_start).count();
                    loop_start = now;

                    long rss_kb = 0;
                    FILE * f = fopen("/proc/self/status", "r");
                    if (f) {
                        char line[256];
                        while (fgets(line, sizeof(line), f)) {
                            if (strncmp(line, "VmRSS:", 6) == 0) {
                                sscanf(line + 6, "%ld", &rss_kb);
                                break;
                            }
                        }
                        fclose(f);
                    }

                    const size_t work_mem_mb = ctx.work_mem.size() / (1024 * 1024);
                    const size_t scratch_mb = ctx.shared_scratch_w.size() * sizeof(float) / (1024 * 1024);
                    std::cout << "[stress] iter=" << (iter + 1)
                              << " iter_ms=" << iter_ms
                              << " rss_mb=" << (rss_kb / 1024)
                              << " work_mem_mb=" << work_mem_mb
                              << " scratch_mb=" << scratch_mb
                              << std::endl;
                }
            }

            std::cout << "[SUCCESS] Stress test completed!\n";
        } else {
            throw std::runtime_error("Unknown mode: " + mode);
        }
        std::cout << "[bmo_main] Cleaning up...\n";
        if (ctx.work_ctx) {
            ggml_free(ctx.work_ctx);
            ctx.work_ctx = nullptr;
        }
        if (model.wctx) {
            ggml_free(model.wctx);
            model.wctx = nullptr;
        }
        if (model.gctx) {
            gguf_free(model.gctx);
            model.gctx = nullptr;
        }
        bmo_free_cuda_resources(ctx);
        std::cout << "[bmo_main] Test completed successfully!\n";

        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "[bmo_main] ERROR: " << ex.what() << std::endl;
        if (ctx.work_ctx) {
            ggml_free(ctx.work_ctx);
            ctx.work_ctx = nullptr;
        }
        if (model.wctx) {
            ggml_free(model.wctx);
            model.wctx = nullptr;
        }
        if (model.gctx) {
            gguf_free(model.gctx);
            model.gctx = nullptr;
        }
        bmo_free_cuda_resources(ctx);
        return 1;
    }
}

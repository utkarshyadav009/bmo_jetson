"""TensorRT-accelerated Mimi audio codec wrapper for Jetson Orin Nano.

Replaces SEANet convolutional layers with FP16 TensorRT execution contexts
and optimizes RVQ distance search via cuBLAS matrix multiplication.
"""

import os
import torch
import numpy as np

try:
    import tensorrt as trt
    _HAS_TRT = True
except ImportError:
    _HAS_TRT = False


class TRTEngineRunner:
    def __init__(self, engine_path: str, input_name: str, output_name: str):
        if not _HAS_TRT:
            raise RuntimeError("TensorRT Python package not available")
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = input_name
        self.output_name = output_name
        self.stream = torch.cuda.Stream()

    def run(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, output_tensor.data_ptr())
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()


def optimize_quantizer(mimi):
    """Replaces iterative torch.cdist with fast GEMM + argmax."""
    layers = [mimi.quantizer.rvq_first.vq.layers[0]] + list(mimi.quantizer.rvq_rest.vq.layers)
    for layer in layers:
        cb = layer._codebook
        emb = cb.embedding.detach()
        norm_sq = 0.5 * (emb ** 2).sum(dim=1)
        cb._fast_emb = emb
        cb._fast_norm_sq = norm_sq

        def make_fast_quantize(c):
            return lambda x: (x @ c._fast_emb.T - c._fast_norm_sq).argmax(dim=-1)

        cb._quantize = make_fast_quantize(cb)


import ctypes

try:
    _libbmo_path = os.path.join(os.path.dirname(__file__), "build", "libbmo.so")
    if os.path.isfile(_libbmo_path):
        _libbmo = ctypes.CDLL(_libbmo_path)
    else:
        _libbmo = ctypes.CDLL("libbmo.so")
    _libbmo.bmo_rvq_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _libbmo.bmo_rvq_decode.restype = ctypes.c_int
    _libbmo.bmo_rvq_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    _libbmo.bmo_rvq_encode.restype = ctypes.c_int
    _HAS_LIBBMO_RVQ = True
except Exception as e:
    _libbmo = None
    _HAS_LIBBMO_RVQ = False


class TRTMimiCodec:
    """High-performance hybrid TensorRT + Fused CUDA RVQ Mimi wrapper."""

    def __init__(self, mimi_model, encoder_engine_path: str = "seanet_encoder.engine", decoder_engine_path: str = "seanet_decoder.engine"):
        self.mimi = mimi_model
        self.mimi.eval()
        self.mimi.streaming_forever(1)
        optimize_quantizer(self.mimi)

        self.has_trt_enc = False
        self.has_trt_dec = False

        if _HAS_TRT and os.path.isfile(encoder_engine_path):
            try:
                self.trt_encoder = TRTEngineRunner(encoder_engine_path, "audio", "latent")
                self.has_trt_enc = True
                print(f"[+] Loaded TensorRT SEANet Encoder: {encoder_engine_path}")
            except Exception as e:
                print(f"[!] Could not load TRT encoder ({e}), using PyTorch fallback")

        if _HAS_TRT and os.path.isfile(decoder_engine_path):
            try:
                self.trt_decoder = TRTEngineRunner(decoder_engine_path, "latent", "audio")
                self.has_trt_dec = True
                print(f"[+] Loaded TensorRT SEANet Decoder: {decoder_engine_path}")
            except Exception as e:
                print(f"[!] Could not load TRT decoder ({e}), using PyTorch fallback")

        # Fused RVQ Precomputed Tables & Static Buffers
        self.has_fused_rvq = _HAS_LIBBMO_RVQ
        if self.has_fused_rvq:
            try:
                # 1. Precomputed Decode Projection Tables: [8, 2048, 512]
                W0 = self.mimi.quantizer.rvq_first.output_proj.weight.squeeze(-1) # [512, 256]
                E0 = self.mimi.quantizer.rvq_first.vq.layers[0]._codebook.embedding # [2048, 256]
                proj_E0 = (E0 @ W0.T).unsqueeze(0) # [1, 2048, 512]

                W_rest = self.mimi.quantizer.rvq_rest.output_proj.weight.squeeze(-1) # [512, 256]
                proj_E_rest = torch.stack([
                    self.mimi.quantizer.rvq_rest.vq.layers[i]._codebook.embedding @ W_rest.T
                    for i in range(7)
                ]) # [7, 2048, 512]
                self.rvq_proj_tables = torch.cat([proj_E0, proj_E_rest], dim=0).contiguous()

                # 2. Encode Weights & Norms
                self.w_in0 = self.mimi.quantizer.rvq_first.input_proj.weight.squeeze(-1).contiguous()
                self.e0 = self.mimi.quantizer.rvq_first.vq.layers[0]._codebook.embedding.contiguous()
                self.norm0 = (0.5 * (self.e0 ** 2).sum(dim=1)).contiguous()

                self.w_in_rest = self.mimi.quantizer.rvq_rest.input_proj.weight.squeeze(-1).contiguous()
                self.e_rest = torch.stack([
                    self.mimi.quantizer.rvq_rest.vq.layers[i]._codebook.embedding for i in range(7)
                ]).contiguous()
                self.norm_rest = torch.stack([
                    0.5 * (self.e_rest[i] ** 2).sum(dim=1) for i in range(7)
                ]).contiguous()

                # Scratch & I/O buffers
                self.rvq_scratch = torch.zeros(512, dtype=torch.float32, device="cuda")
                self.rvq_codes_out = torch.zeros(8, dtype=torch.int32, device="cuda")
                self.rvq_dec_out = torch.zeros((1, 512, 1), dtype=torch.float32, device="cuda")
                print("[+] Loaded Fused CUDA Mimi RVQ Quantizer (sub-millisecond latency)")
            except Exception as e:
                print(f"[!] Could not setup fused RVQ ({e}), using fast GEMM fallback")
                self.has_fused_rvq = False

        # Static preallocated GPU buffers
        self.enc_out_buf = torch.empty((1, 512, 2), dtype=torch.float32, device="cuda")
        self.dec_out_buf = torch.empty((1, 1, 1920), dtype=torch.float32, device="cuda")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode PCM audio (1, 1, 1920) -> (1, 8, 1) codebooks."""
        state = self.mimi._streaming_state
        if self.has_trt_enc:
            self.trt_encoder.run(x, self.enc_out_buf)
            emb = self.enc_out_buf
        else:
            emb = state.graphed_encoder(x).clone()

        if self.mimi.encoder_transformer is not None:
            (emb,) = state.graphed_tr_enc(emb)
        emb = self.mimi._to_framerate(emb)

        if self.has_fused_rvq:
            _libbmo.bmo_rvq_encode(
                emb.data_ptr(),
                self.w_in0.data_ptr(),
                self.e0.data_ptr(),
                self.norm0.data_ptr(),
                self.w_in_rest.data_ptr(),
                self.e_rest.data_ptr(),
                self.norm_rest.data_ptr(),
                self.rvq_codes_out.data_ptr(),
                self.rvq_scratch.data_ptr(),
                None
            )
            return self.rvq_codes_out.view(1, 8, 1)
        else:
            codes = self.mimi.quantizer.encode(emb)
            return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode (1, 8, 1) codebooks -> PCM audio (1, 1, 1920)."""
        state = self.mimi._streaming_state
        if self.has_fused_rvq:
            codes_i32 = codes.view(8).to(torch.int32).contiguous()
            _libbmo.bmo_rvq_decode(
                codes_i32.data_ptr(),
                self.rvq_proj_tables.data_ptr(),
                self.rvq_dec_out.data_ptr(),
                None
            )
            emb = self.rvq_dec_out
        else:
            emb = self.mimi.decode_latent(codes)

        emb = self.mimi._to_encoder_framerate(emb)
        if self.mimi.decoder_transformer is not None:
            (emb,) = state.graphed_tr_dec(emb)

        if self.has_trt_dec:
            self.trt_decoder.run(emb, self.dec_out_buf)
            out = self.dec_out_buf
        else:
            out = state.graphed_decoder(emb).clone()
        return out

    def reset_streaming(self):
        self.mimi.reset_streaming()

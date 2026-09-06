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


class TRTMimiCodec:
    """High-performance hybrid TensorRT + PyTorch Mimi wrapper."""

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
        codes = self.mimi.quantizer.encode(emb)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode (1, 8, 1) codebooks -> PCM audio (1, 1, 1920)."""
        state = self.mimi._streaming_state
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

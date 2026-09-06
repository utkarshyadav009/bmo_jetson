#!/usr/bin/env python3
"""Export Mimi SEANet encoder and decoder to ONNX and compile to TensorRT FP16 engines."""

import os
import sys
import subprocess
import argparse

os.environ.setdefault("NO_TORCH_COMPILE", "1")
import torch
torch.set_num_threads(2)

from moshi.models import get_mimi


def find_trtexec() -> str:
    candidates = [
        "trtexec",
        "/usr/src/tensorrt/bin/trtexec",
        os.path.expanduser("~/.local/bin/trtexec"),
        "/usr/local/bin/trtexec",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
        import shutil
        p = shutil.which(c)
        if p:
            return p
    raise FileNotFoundError("trtexec binary not found in standard paths (/usr/src/tensorrt/bin/trtexec)")


def export_onnx(mimi_path: str, enc_onnx: str, dec_onnx: str):
    print(f"[+] Loading Mimi model from {mimi_path} on CUDA...")
    mimi = get_mimi(mimi_path, device="cuda")
    mimi.eval()

    print(f"[+] Exporting SEANet encoder to {enc_onnx}...")
    x = torch.zeros((1, 1, 1920), device="cuda")
    torch.onnx.export(
        mimi.encoder,
        x,
        enc_onnx,
        input_names=["audio"],
        output_names=["latent"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"    SEANet encoder exported ({os.path.getsize(enc_onnx) / 1024 / 1024:.1f} MB)")

    print(f"[+] Exporting SEANet decoder to {dec_onnx}...")
    codes = torch.zeros((1, 8, 1), dtype=torch.long, device="cuda")
    d1 = mimi.decode_latent(codes)
    d2 = mimi._to_encoder_framerate(d1)
    if mimi.decoder_transformer is not None:
        (d3,) = mimi.decoder_transformer(d2)
    else:
        d3 = d2
    torch.onnx.export(
        mimi.decoder,
        d3,
        dec_onnx,
        input_names=["latent"],
        output_names=["audio"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"    SEANet decoder exported ({os.path.getsize(dec_onnx) / 1024 / 1024:.1f} MB)")


def compile_trt(trtexec_bin: str, onnx_path: str, engine_path: str):
    print(f"\n[+] Compiling {onnx_path} -> {engine_path} with TensorRT FP16...")
    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--memPoolSize=workspace:1024M",
    ]
    print(f"    Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"    Successfully generated {engine_path} ({os.path.getsize(engine_path) / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export and compile Mimi TensorRT engines on Jetson")
    parser.add_argument("--mimi-path", default="/home/bmo/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors")
    parser.add_argument("--enc-onnx", default="seanet_encoder.onnx")
    parser.add_argument("--dec-onnx", default="seanet_decoder.onnx")
    parser.add_argument("--enc-engine", default="seanet_encoder.engine")
    parser.add_argument("--dec-engine", default="seanet_decoder.engine")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export if files already exist")
    args = parser.parse_args()

    trtexec_bin = find_trtexec()
    print(f"[+] Found trtexec at: {trtexec_bin}")

    if not args.skip_onnx:
        export_onnx(args.mimi_path, args.enc_onnx, args.dec_onnx)

    compile_trt(trtexec_bin, args.enc_onnx, args.enc_engine)
    compile_trt(trtexec_bin, args.dec_onnx, args.dec_engine)

    print("\n[+] All TensorRT Mimi engines compiled and ready for deployment!")


if __name__ == "__main__":
    main()

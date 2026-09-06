"""ctypes wrapper around libbmo.so (the optimized C++ Temporal & Depth engine).

Exposes a Pythonic ``BMOEngine`` that mirrors the C-API in ``bmo_api.h``.
"""

import ctypes
import os
import numpy as np
from typing import Tuple, Optional, List

_CANDIDATE_PATHS = [
    os.environ.get("BMO_SO_PATH"),
    os.path.join(os.path.dirname(__file__), "build", "libbmo.so"),
    os.path.join(os.path.dirname(__file__), "build_jetson", "libbmo.so"),
    "./build/libbmo.so",
    "./build_jetson/libbmo.so",
    "libbmo.so",
]

_LIB_PATH = None
for p in _CANDIDATE_PATHS:
    if p and os.path.isfile(p):
        _LIB_PATH = p
        break

if not _LIB_PATH:
    _LIB_PATH = os.environ.get("BMO_SO_PATH", "./build/libbmo.so")

_lib = ctypes.CDLL(_LIB_PATH)

_lib.bmo_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib.bmo_init.restype  = ctypes.c_void_p
_lib.bmo_free.argtypes = [ctypes.c_void_p]
_lib.bmo_free.restype  = None
_lib.bmo_reset.argtypes = [ctypes.c_void_p]
_lib.bmo_reset.restype  = None

for _name in (
    "bmo_get_n_layers",
    "bmo_get_n_embd",
    "bmo_get_n_codebooks",
    "bmo_get_dep_q",
    "bmo_get_text_vocab",
    "bmo_get_audio_vocab",
    "bmo_get_n_attn_heads",
    "bmo_get_head_dim",
    "bmo_capture_graphs",
    "bmo_has_cuda_graphs",
):
    if hasattr(_lib, _name):
        _f = getattr(_lib, _name)
        _f.argtypes = [ctypes.c_void_p]
        _f.restype = ctypes.c_int

if hasattr(_lib, "bmo_copy_k_cache_f32"):
    _lib.bmo_copy_k_cache_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    _lib.bmo_copy_k_cache_f32.restype = ctypes.c_int

_lib.bmo_forward_temporal.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
_lib.bmo_forward_temporal.restype = ctypes.c_int

if hasattr(_lib, "bmo_forward_temporal2"):
    _lib.bmo_forward_temporal2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    _lib.bmo_forward_temporal2.restype = ctypes.c_int

_lib.bmo_forward_depth.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
_lib.bmo_forward_depth.restype = ctypes.c_int

_lib.bmo_last_error.argtypes = [ctypes.c_void_p]
_lib.bmo_last_error.restype  = ctypes.c_char_p


class BMOEngine:
    def __init__(self, gguf_path: str, n_ctx: int = 1024):
        self._h = _lib.bmo_init(gguf_path.encode("utf-8"), n_ctx)
        if not self._h:
            raise RuntimeError(f"bmo_init returned NULL for path: {gguf_path}")
        self.n_layers     = _lib.bmo_get_n_layers(self._h)
        self.n_embd       = _lib.bmo_get_n_embd(self._h)
        self.n_codebooks  = _lib.bmo_get_n_codebooks(self._h)
        self.dep_q        = _lib.bmo_get_dep_q(self._h)
        self.text_vocab   = _lib.bmo_get_text_vocab(self._h)
        self.audio_vocab  = _lib.bmo_get_audio_vocab(self._h)
        self.n_attn_heads = _lib.bmo_get_n_attn_heads(self._h) if hasattr(_lib, "bmo_get_n_attn_heads") else 32
        self.head_dim     = _lib.bmo_get_head_dim(self._h) if hasattr(_lib, "bmo_get_head_dim") else 128
        self._pos         = 0

        self._buf_z       = np.empty(self.n_embd,      dtype=np.float32)
        self._buf_text    = np.empty(self.text_vocab,  dtype=np.float32)
        self._buf_audio   = np.empty(self.audio_vocab, dtype=np.float32)

    def reset(self) -> None:
        _lib.bmo_reset(self._h)
        self._pos = 0

    @property
    def pos(self) -> int:
        return self._pos

    @pos.setter
    def pos(self, value: int) -> None:
        self._pos = int(value)

    def capture_graphs(self) -> int:
        if hasattr(_lib, "bmo_capture_graphs"):
            return _lib.bmo_capture_graphs(self._h)
        return -1

    def has_cuda_graphs(self) -> bool:
        if hasattr(_lib, "bmo_has_cuda_graphs"):
            return bool(_lib.bmo_has_cuda_graphs(self._h))
        return False

    def forward_temporal(self, tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert tokens.dtype == np.int32 and tokens.shape == (self.n_codebooks,), (
            f"Expected int32 array of shape ({self.n_codebooks},), got {tokens.dtype} shape {tokens.shape}"
        )
        if not tokens.flags['C_CONTIGUOUS']:
            tokens = np.ascontiguousarray(tokens)
        rc = _lib.bmo_forward_temporal(
            self._h,
            tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            int(tokens.size),
            int(self._pos),
            self._buf_z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._buf_text.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            err = _lib.bmo_last_error(self._h)
            raise RuntimeError(f"forward_temporal rc={rc}: {err.decode() if err else 'unknown'}")
        self._pos += 1
        return self._buf_z.copy(), self._buf_text.copy()

    def forward_temporal2(
        self,
        tokens: np.ndarray,
        capture_layers: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        assert tokens.dtype == np.int32 and tokens.shape == (self.n_codebooks,), (
            f"Expected int32 array of shape ({self.n_codebooks},), got {tokens.dtype} shape {tokens.shape}"
        )
        if not tokens.flags['C_CONTIGUOUS']:
            tokens = np.ascontiguousarray(tokens)

        if not capture_layers:
            z, text_logits = self.forward_temporal(tokens)
            return z, text_logits, None

        n_cap = len(capture_layers)
        cap_arr = np.array(capture_layers, dtype=np.int32)
        cap_out = np.empty((n_cap, self.n_embd), dtype=np.float32)

        rc = _lib.bmo_forward_temporal2(
            self._h,
            tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            int(tokens.size),
            int(self._pos),
            self._buf_z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._buf_text.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cap_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            int(n_cap),
            cap_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            err = _lib.bmo_last_error(self._h)
            raise RuntimeError(f"forward_temporal2 rc={rc}: {err.decode() if err else 'unknown'}")
        self._pos += 1
        return self._buf_z.copy(), self._buf_text.copy(), cap_out

    def forward_depth(self, cb_index: int, prev_token: int, transformer_out: np.ndarray) -> np.ndarray:
        assert transformer_out.dtype == np.float32 and transformer_out.shape == (self.n_embd,), (
            f"Expected float32 array of shape ({self.n_embd},), got {transformer_out.dtype} shape {transformer_out.shape}"
        )
        if not transformer_out.flags['C_CONTIGUOUS']:
            transformer_out = np.ascontiguousarray(transformer_out)
        rc = _lib.bmo_forward_depth(
            self._h,
            int(cb_index),
            int(prev_token),
            transformer_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._buf_audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            err = _lib.bmo_last_error(self._h)
            raise RuntimeError(f"forward_depth rc={rc}: {err.decode() if err else 'unknown'}")
        return self._buf_audio.copy()

    def get_k_cache_f32(self, layer: int, t_start: int, n_positions: int) -> np.ndarray:
        """Temporal K-cache slice as float32, layout (n_positions, n_heads, head_dim)."""
        if not hasattr(_lib, "bmo_copy_k_cache_f32"):
            raise RuntimeError("libbmo.so was built without bmo_copy_k_cache_f32.")
        nh = self.n_attn_heads
        hd = self.head_dim
        if nh <= 0 or hd <= 0:
            raise RuntimeError("bmo_get_n_attn_heads / bmo_get_head_dim returned invalid geometry.")
        n_el = int(n_positions) * int(nh) * int(hd)
        buf = np.empty(n_el, dtype=np.float32)
        n_written = _lib.bmo_copy_k_cache_f32(
            self._h,
            int(layer),
            int(t_start),
            int(n_positions),
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(n_el),
        )
        if n_written < 0:
            raise RuntimeError(f"bmo_copy_k_cache_f32 failed with code {n_written}")
        if n_written != n_el:
            raise RuntimeError(f"bmo_copy_k_cache_f32 expected {n_el} floats, got {n_written}")
        return buf.reshape(int(n_positions), int(nh), int(hd))

    def __del__(self):
        if getattr(self, "_h", None):
            _lib.bmo_free(self._h)
            self._h = None

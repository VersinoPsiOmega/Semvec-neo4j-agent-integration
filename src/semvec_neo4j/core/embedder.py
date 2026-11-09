"""Embedder adapters consumed by Semvec's SessionManager.

Semvec is embedding-agnostic: it expects a duck-typed object with
``get_embedding(text) -> np.ndarray`` and ``get_dimension() -> int``.
This module provides two adapters:

- :class:`HashEmbedder` — a deterministic, dependency-free stand-in
  used by tests and offline demos.
- :class:`SentenceTransformerEmbedder` — production-grade adapter
  around ``sentence-transformers`` with lazy model loading.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np


class HashEmbedder:
    """Deterministic, dependency-free embedder for tests and offline runs.

    Given the same text, always returns the same unit-norm vector. The
    output is *not* semantically meaningful — it's a stable surrogate
    for situations where shipping a real model is overkill.
    """

    def __init__(self, dimension: int = 16):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    def get_dimension(self) -> int:
        return self._dimension

    def get_embedding(self, text: str) -> np.ndarray:
        seed_bytes = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(seed_bytes, "big") % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dimension).astype(np.float32)
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            v[0] = 1.0
            norm = 1.0
        return v / norm


class SentenceTransformerEmbedder:
    """Adapter around ``sentence-transformers`` models.

    Defaults to ``all-mpnet-base-v2`` (768-d) — best general-purpose
    recall on the SBERT benchmark. Override via ``model_name=`` for a
    smaller/faster model (e.g. ``all-MiniLM-L6-v2`` at 384-d) when
    inference latency or model size matters more than retrieval quality.

    The model is loaded lazily on first use so importing this module
    is cheap. Pass ``device`` to pin to ``"cuda"`` / ``"cpu"`` etc.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2", device: Optional[str] = None):
        self._model_name = model_name
        self._device = device
        self._model = None  # type: ignore[assignment]
        self._dimension: Optional[int] = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                # Prefer the local cache: this skips HuggingFace's per-start
                # Hub freshness check, so repeat runs load in well under a
                # second and work fully offline once the model is cached.
                self._model = SentenceTransformer(
                    self._model_name, device=self._device, local_files_only=True
                )
            except Exception:
                # Not cached yet (or cache incomplete) — fetch once from the
                # Hub; subsequent runs take the fast local path above.
                self._model = SentenceTransformer(self._model_name, device=self._device)
            # sentence-transformers renamed this method; prefer the new name
            # and fall back to the legacy one for older releases.
            get_dim = getattr(
                self._model,
                "get_embedding_dimension",
                self._model.get_sentence_embedding_dimension,
            )
            self._dimension = int(get_dim())
        return self._model

    def get_dimension(self) -> int:
        if self._dimension is None:
            self._load()
        assert self._dimension is not None
        return self._dimension

    def get_embedding(self, text: str) -> np.ndarray:
        model = self._load()
        v = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

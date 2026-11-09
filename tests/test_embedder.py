"""Tests for the embedder adapters that feed Semvec's SessionManager.

Semvec needs an object exposing get_embedding(text) and get_dimension().
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from semvec_neo4j.core.embedder import HashEmbedder, SentenceTransformerEmbedder


@pytest.fixture(autouse=True)
def clean_test_data():
    """Override the conftest autouse fixture — these tests need no Neo4j."""
    yield


class TestHashEmbedder:
    def test_dimension_default(self):
        emb = HashEmbedder()
        assert emb.get_dimension() == 16

    def test_dimension_configurable(self):
        emb = HashEmbedder(dimension=32)
        assert emb.get_dimension() == 32

    def test_returns_numpy_float32_vector(self):
        v = HashEmbedder().get_embedding("hello world")
        assert isinstance(v, np.ndarray)
        assert v.dtype == np.float32
        assert v.shape == (16,)

    def test_unit_norm(self):
        v = HashEmbedder().get_embedding("anything")
        assert math.isclose(float(np.linalg.norm(v)), 1.0, rel_tol=1e-5)

    def test_deterministic(self):
        a = HashEmbedder().get_embedding("same text")
        b = HashEmbedder().get_embedding("same text")
        assert np.allclose(a, b)

    def test_different_texts_differ(self):
        a = HashEmbedder().get_embedding("text one")
        b = HashEmbedder().get_embedding("text two")
        # cosine should not be ~1
        cos = float(np.dot(a, b))
        assert cos < 0.99

    def test_empty_text_safe(self):
        v = HashEmbedder().get_embedding("")
        assert v.shape == (16,)
        assert math.isclose(float(np.linalg.norm(v)), 1.0, rel_tol=1e-5)


class TestSentenceTransformerEmbedder:
    """Light tests — heavy model load is opt-in to keep CI fast."""

    def test_lazy_load_does_not_raise(self):
        # Constructing should not download anything.
        SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    @pytest.mark.integration
    def test_real_model_round_trip(self):
        emb = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        dim = emb.get_dimension()
        assert dim == 384
        v = emb.get_embedding("Patient diagnosed with diabetes.")
        assert isinstance(v, np.ndarray)
        assert v.shape == (384,)
        assert v.dtype == np.float32

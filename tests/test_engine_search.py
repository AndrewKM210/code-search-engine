from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cse.search_engine.engine import CodeSearchEngine


def _make_engine(search_results):
    """Builds a CodeSearchEngine with mocked dependencies, bypassing __init__
    so no real embedding model or Qdrant instance is loaded."""
    engine = object.__new__(CodeSearchEngine)
    engine.db_collection = "test_collection"
    # MagicMock() is a stand-in object: any attribute/method accessed on it
    # (e.g. embed_query) is created automatically, no real class needed.
    engine.embeddings_model = MagicMock()
    # Fixes what calling embed_query(...) returns, instead of running a real model.
    engine.embeddings_model.embed_query.return_value = [0.1, 0.2, 0.3]
    engine.qdrant_client = MagicMock()
    engine.qdrant_client.search.return_value = search_results
    return engine


def test_search_formats_results_as_ranked_dicts():
    scored_points = [
        SimpleNamespace(id=2, score=0.9, payload={"code_content": "def foo(): pass"}),
        SimpleNamespace(id=5, score=0.5, payload={"code_content": "def bar(): pass"}),
    ]
    engine = _make_engine(scored_points)

    results = engine.search("find foo")

    assert results == [
        {"code_id": 2, "score": 0.9, "payload": {"code_content": "def foo(): pass"}},
        {"code_id": 5, "score": 0.5, "payload": {"code_content": "def bar(): pass"}},
    ]


def test_search_embeds_query_and_passes_params_to_qdrant():
    engine = _make_engine([])

    engine.search("a query", k=7, hnsw_ef=64)

    # Mocks record every call they receive; this checks embed_query was
    # called exactly once, with "a query" as its only argument.
    engine.embeddings_model.embed_query.assert_called_once_with("a query")
    # call_args holds the (args, kwargs) of the most recent call; we only
    # need the keyword args qdrant_client.search() was called with.
    _, kwargs = engine.qdrant_client.search.call_args
    assert kwargs["collection_name"] == "test_collection"
    assert kwargs["limit"] == 7
    assert kwargs["query_vector"] == [0.1, 0.2, 0.3]
    assert kwargs["search_params"].hnsw_ef == 64


def test_search_raises_if_not_initialized():
    engine = object.__new__(CodeSearchEngine)
    engine.qdrant_client = None

    with pytest.raises(AssertionError):
        engine.search("anything")
from unittest.mock import patch

import pytest

from cse.search_engine import evaluation


class _FakeEngine:
    """Stands in for CodeSearchEngine so metric math is tested without a
    real embedding model or Qdrant instance."""

    _RESULTS = {
        "hit": [{"code_id": 0, "score": 1.0, "payload": {}}],
        "miss": [{"code_id": 1, "score": 1.0, "payload": {}}, {"code_id": 2, "score": 0.5, "payload": {}}],
    }

    def __init__(self, *args, **kwargs):
        pass

    def index_corpus(self, corpus):
        pass

    def search(self, query, k=10, hnsw_ef=128):
        return self._RESULTS[query]


def test_run_evaluation_computes_mrr_ndcg_recall():
    corpus = {0: "a", 1: "b", 2: "c"}
    # "hit" finds its ground truth at rank 1; "miss" never finds id 99.
    eval_queries = {"hit": [0], "miss": [99]}

    with patch.object(evaluation, "CodeSearchEngine", _FakeEngine):
        metrics = evaluation.run_evaluation(
            model_name="fake-model",
            corpus=corpus,
            eval_queries=eval_queries,
            db_collection="fake_collection",
            db_path="unused",
            quiet=True,
        )

    assert metrics["MRR@10"] == pytest.approx(0.5)
    assert metrics["Recall@10"] == pytest.approx(0.5)
    assert metrics["nDCG@10"] == pytest.approx(0.5)
    assert metrics["Avg. Query Time (ms)"] >= 0
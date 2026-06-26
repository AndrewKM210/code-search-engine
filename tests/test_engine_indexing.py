from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import cse.search_engine.engine as engine_module
from cse.search_engine.engine import CodeSearchEngine


def _make_indexable_engine():
    """Builds a CodeSearchEngine with mocked dependencies, bypassing __init__
    so no real embedding model or Qdrant instance is loaded."""
    engine = object.__new__(CodeSearchEngine)
    engine.db_collection = "test_collection"
    engine.quiet = True
    engine.embeddings_model = MagicMock()
    engine.embeddings_model.embed_documents.return_value = [[0.1], [0.2]]
    engine.qdrant_client = MagicMock()
    return engine


def test_index_from_directory_loads_each_glob_pattern():
    engine = _make_indexable_engine()
    doc_py = Document(
        page_content="def foo(): pass", metadata={"source": "foo.py"}
    )
    doc_yaml = Document(
        page_content="key: value", metadata={"source": "config.yaml"}
    )

    def _fake_loader(dir_path, glob, **kwargs):
        loader = MagicMock()
        loader.load.return_value = [doc_py] if glob == "**/*.py" else [doc_yaml]
        return loader

    with patch.object(
        engine_module, "DirectoryLoader", side_effect=_fake_loader
    ):
        engine.index_from_directory(".", glob=["**/*.py", "**/*.yaml"])

    _, kwargs = engine.qdrant_client.upsert.call_args
    sources = {p.payload["source"] for p in kwargs["points"]}
    assert sources == {"foo.py", "config.yaml"}


def test_index_from_directory_assigns_collision_free_ids_across_patterns():
    engine = _make_indexable_engine()
    doc_py = Document(
        page_content="def foo(): pass", metadata={"source": "foo.py"}
    )
    doc_yaml = Document(
        page_content="key: value", metadata={"source": "config.yaml"}
    )

    def _fake_loader(dir_path, glob, **kwargs):
        loader = MagicMock()
        loader.load.return_value = [doc_py] if glob == "**/*.py" else [doc_yaml]
        return loader

    with patch.object(
        engine_module, "DirectoryLoader", side_effect=_fake_loader
    ):
        engine.index_from_directory(".", glob=["**/*.py", "**/*.yaml"])

    _, kwargs = engine.qdrant_client.upsert.call_args
    ids = [p.id for p in kwargs["points"]]
    assert ids == list(range(len(ids)))
    assert len(set(ids)) == len(ids)


def test_index_from_directory_accepts_a_single_glob_string():
    engine = _make_indexable_engine()
    doc_py = Document(
        page_content="def foo(): pass", metadata={"source": "foo.py"}
    )

    with patch.object(engine_module, "DirectoryLoader") as mock_loader_cls:
        mock_loader_cls.return_value.load.return_value = [doc_py]
        engine.index_from_directory(".", glob="**/*.py")

    mock_loader_cls.assert_called_once()
    assert mock_loader_cls.call_args.kwargs["glob"] == "**/*.py"

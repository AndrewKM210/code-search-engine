from typing import Any, Dict, List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


class CodeSearchEngine:
    """
    An embeddings-based search engine using Qdrant for vector storage.
    """

    def __init__(
        self,
        model_name: str,
        db_collection: str,
        db_path: str,
        db_recreate: bool = False,
        hnsw_config: Dict = {},
        optimizers_config: Dict = {},
        quiet: bool = False,
    ):
        """
        Initializes the search engine.

        Args:
            model_name (str): The name or path of the Hugging Face sentence-transformer model.
            db_collection (str): The name of the Qdrant collection.
            db_path (str): The path for Qdrant's on-disk storage.
            db_recreate (bool): Recreate Qdrant collection even if it exists.
            hnsw_config (dict): Parameters of HNSW index.
            optimizers_config (dict): Parameters of index optimizer.
            quiet (bool): Do not print to output.
        """
        if not quiet:
            print(f"Initializing engine with model: {model_name} and collection: {db_collection}")
        self.model_name = model_name
        self.db_collection = db_collection
        self.quiet = quiet

        # Use HuggingFaceEmbeddings for compatibility with LangChain loaders
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
        )

        # Use SentenceTransformer directly for batch encoding and getting dimensions
        self._sbert_model = SentenceTransformer(model_name, device="cuda")
        self.embedding_dim = self._sbert_model.get_sentence_embedding_dimension()

        # Initialize Qdrant client and parameters
        self.qdrant_client = QdrantClient(path=db_path)
        self.hnsw_config = hnsw_config
        self.optimizers_config = optimizers_config

        # Create or recreate the collection
        self._create_collection(db_recreate)

    def _create_collection(self, recreate: bool):
        """
        Creates or recreates the Qdrant collection with the correct configuration.

        Args:
            recreate (bool): Recreate the collection even if it already exists.
        """
        if not self.qdrant_client.collection_exists(self.db_collection) or recreate:
            try:
                self.qdrant_client.recreate_collection(
                    collection_name=self.db_collection,
                    vectors_config=models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE),
                    hnsw_config=models.HnswConfigDiff(**self.hnsw_config),
                    optimizers_config=models.OptimizersConfigDiff(**self.optimizers_config),
                )
                if not self.quiet:
                    print(f"Collection '{self.db_collection}' created successfully.")
            except Exception as e:
                print(f"Could not recreate collection: {e}. It might already exist with compatible settings.")
        else:
            if not self.quiet:
                print(f"Collection '{self.db_collection}' already exists. Not recreating collection.")

    def index_from_directory(self, dir_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Loads, splits, and indexes all documents from a specified directory.

        Args:
            dir_path (str): Path to directory with documents.
            chunk_size (int): Size of indexed text chunks.
            chunk_overlap (int): Overlap in characters between chunks.
        """
        if not self.quiet:
            print(f"Indexing documents from directory: {dir_path}")

        # Load documents
        loader = DirectoryLoader(
            dir_path,
            glob="**/*.*",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()

        if not documents:
            print(f"No documents found in {dir_path}.")
            return

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        if not self.quiet:
            print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        # Embed and prepare points
        points = []
        chunk_texts = [chunk.page_content for chunk in chunks]

        if not self.quiet:
            print(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embeddings_model.embed_documents(chunk_texts)

        for i, chunk in enumerate(chunks):
            points.append(
                models.PointStruct(
                    id=i,
                    vector=embeddings[i],
                    payload={"content": chunk.page_content, "source": chunk.metadata.get("source", "unknown")},
                )
            )

        # Upsert to Qdrant
        self.qdrant_client.upsert(collection_name=self.db_collection, points=points, wait=True)
        if not self.quiet:
            print(f"Successfully indexed {len(points)} chunks into '{self.db_collection}'.")

    def index_corpus(self, corpus: Dict[str, str]):
        """
        Indexes a pre-defined corpus from a dictionary.

        Args:
            corpus (Dict[str, str]): A dictionary mapping a unique code_id -> code_content
        """
        if not self.quiet:
            print(f"Indexing corpus with {len(corpus)} entries...")

        contents = list(corpus.values())

        # Embed in batches
        if not self.quiet:
            print(f"Embedding {len(contents)} code snippets...")
        embeddings = self._sbert_model.encode(contents, batch_size=32, show_progress_bar=not self.quiet)

        # Prepare points
        points = []
        for i, (code_id, content) in enumerate(corpus.items()):
            points.append(
                models.PointStruct(id=code_id, vector=embeddings[i].tolist(), payload={"code_content": content})
            )

        # Upsert to Qdrant
        self.qdrant_client.upsert(collection_name=self.db_collection, points=points, wait=True)
        if not self.quiet:
            print(f"Successfully indexed {len(points)} snippets into '{self.db_collection}'.")

    def search(self, text_query: str, k: int = 10, hnsw_ef: int = 128) -> List[Dict[str, Any]]:
        """
        Searches over the collection by text query.

        Args:
            text_query (str): The natural language query.
            k (int): The number of top results to return.
            hnsw_ef (int): Query search range.

        Returns:
            list: A list of dictionaries, each containing 'code_id', 'score', and 'payload'.
        """
        assert self.qdrant_client, "Search engine is not initialized."

        # Embed the query
        query_embedding = self.embeddings_model.embed_query(text_query)

        # Perform search
        search_results = self.qdrant_client.search(
            collection_name=self.db_collection,
            query_vector=query_embedding,
            limit=k,
            with_payload=True,
            with_vectors=False,
            search_params=models.SearchParams(hnsw_ef=hnsw_ef, exact=False),
        )

        # Format results
        formatted_results = []
        for scored_point in search_results:
            formatted_results.append(
                {"code_id": scored_point.id, "score": scored_point.score, "payload": scored_point.payload}
            )

        return formatted_results

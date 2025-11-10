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

    def __init__(self, model_name: str, db_collection: str, db_path: str):
        """
        Initializes the search engine.

        Args:
            model_name (str): The name or path of the Hugging Face sentence-transformer model.
            db_collection (str): The name of the Qdrant collection.
            db_path (str): The path for Qdrant's on-disk storage.
        """
        print(f"Initializing engine with model: {model_name} and collection: {db_collection}")
        self.model_name = model_name
        self.db_collection = db_collection

        # Use HuggingFaceEmbeddings for compatibility with LangChain loaders
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # TODO: try with cuda?
        )

        # Use SentenceTransformer directly for batch encoding and getting dimensions
        self._sbert_model = SentenceTransformer(model_name)
        self.embedding_dim = self._sbert_model.get_sentence_embedding_dimension()

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(path=db_path)

        # Create or recreate the collection
        self._create_collection()

    def _create_collection(self):
        """
        Creates or recreates the Qdrant collection with the correct configuration.
        """
        try:
            self.qdrant_client.recreate_collection(
                collection_name=self.db_collection,
                vectors_config=models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE),
            )
            print(f"Collection '{self.db_collection}' created successfully.")
        except Exception as e:
            print(f"Could not recreate collection: {e}. It might already exist with compatible settings.")

    def index_from_directory(self, dir_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Loads, splits, and indexes all documents from a specified directory.
        """
        print(f"Indexing documents from directory: {dir_path}")

        # Load Documents
        loader = DirectoryLoader(
            dir_path,
            glob="**/*.*",  # TODO: is it possible for only code files?
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()

        if not documents:
            print(f"No documents found in {dir_path}.")
            return

        # Split Documents
        # TODO: is it best option?
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        # Embed and Prepare Points
        points = []
        chunk_texts = [chunk.page_content for chunk in chunks]

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
        print(f"Successfully indexed {len(points)} chunks into '{self.db_collection}'.")

    def search(self, text_query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Searches over the collection by text query.

        Args:
            text_query (str): The natural language query.
            k (int): The number of top results to return.

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
        )

        # Format results
        formatted_results = []
        for scored_point in search_results:
            formatted_results.append(
                {"code_id": scored_point.id, "score": scored_point.score, "payload": scored_point.payload}
            )

        return formatted_results

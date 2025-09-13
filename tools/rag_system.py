import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class RAGSystem:
    """
    A Retrieval-Augmented Generation system using ChromaDB and sentence-transformers.

    This class handles the storage and retrieval of documents from a persistent
    vector database, providing the long-term memory for the multi-agent system.
    """
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "research_project"):
        """
        Initializes the RAG system.

        Args:
            db_path: The path to the directory where the ChromaDB database will be stored.
            collection_name: The name of the collection to use within ChromaDB.
        """
        print("Initializing RAG System...")
        try:
            # 1. Initialize a persistent ChromaDB client
            self.client = chromadb.PersistentClient(path=db_path)

            # 2. Initialize the embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # 3. Get or create the ChromaDB collection
            self.collection = self.client.get_or_create_collection(name=collection_name)

            # 4. Initialize a document ID counter
            # In a real-world scenario, you might use a more robust ID generation scheme (e.g., UUIDs)
            # or retrieve the max ID from the DB on startup.
            self.doc_id_counter = self.collection.count()

            print(f"RAG System initialized successfully. Collection '{collection_name}' has {self.doc_id_counter} documents.")
        except Exception as e:
            print(f"Error initializing RAG System: {e}")
            raise

    def add_document(self, content: str, metadata: Dict[str, Any]) -> None:
        """
        Adds a document to the vector store.

        Args:
            content: The text content of the document.
            metadata: A dictionary of metadata associated with the document.
        """
        try:
            # Generate a unique ID for the document
            doc_id = f"doc_{self.doc_id_counter}"

            # Encode the content into a vector embedding
            embedding = self.embedding_model.encode(content).tolist()

            # Add the generated doc_id to the metadata so it's retrieved in queries
            metadata_with_id = metadata.copy()
            metadata_with_id['doc_id'] = doc_id

            # Add the document, embedding, and metadata to the collection
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[metadata_with_id],
                documents=[content]
            )

            # Increment the document ID counter
            self.doc_id_counter += 1
            print(f"Added document {doc_id} to collection.")
        except Exception as e:
            print(f"Error adding document to RAG system: {e}")

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the vector store for similar documents.

        Args:
            query_text: The text to search for.
            k: The number of documents to return.

        Returns:
            A list of document dictionaries, each containing the content and metadata.
        """
        print(f"Querying RAG system for: '{query_text}'")
        try:
            # Encode the query into a vector
            query_embedding = self.embedding_model.encode(query_text).tolist()

            # Perform the similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            # Parse the results and return them in a clean format
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]

            if not documents:
                return []

            formatted_results = [
                {"content": doc, "metadata": meta}
                for doc, meta in zip(documents, metadatas)
            ]

            print(f"Found {len(formatted_results)} results from RAG query.")
            return formatted_results
        except Exception as e:
            print(f"Error querying RAG system: {e}")
            return []

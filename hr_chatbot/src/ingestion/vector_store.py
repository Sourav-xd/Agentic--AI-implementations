#hr_chatbot/src/ingestion/vector_store.py
import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class ChromaManager:
    """
    Manages interactions with ChromaDB using Open Source Embeddings.
    """

    def __init__(self, db_path_str: str, embedding_config_dct: Dict[str, str]):
        """
        Initialize ChromaDB manager.

        Args:
            db_path_str (str): Absolute path to the ChromaDB persistence directory.
            embedding_config_dct (Dict): Config for embedding model (name, device).
        """
        self.db_path_str = db_path_str
        
        # Initialize Open Source Embeddings (HuggingFace)
        self.embedding_func_ins = HuggingFaceEmbeddings(
            model_name=embedding_config_dct.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            model_kwargs={'device': embedding_config_dct.get("device", "cpu")}
        )

    def store_chunks(self, chunks_lst: List[Any]) -> Dict[str, Any]:
        """
        Embeds and stores document chunks into ChromaDB.

        Args:
            chunks_lst (List[Any]): List of LangChain Document objects.

        Returns:
            Dict[str, Any]: Result dictionary with status and message.
        """
        try:
            if not chunks_lst:
                return {"status": "warning", "message": "No chunks provided to store.", "data": None}
            
            # Initialize Chroma with persistence
            vector_db_ins = Chroma(
                persist_directory=self.db_path_str,
                embedding_function=self.embedding_func_ins,
                collection_name="hr_policy_vectors"
            )
            
            # Add documents (Batched automatically by LangChain)
            vector_db_ins.add_documents(documents=chunks_lst)
            
            return {
                "status": "success", 
                "message": f"Successfully stored {len(chunks_lst)} chunks in ChromaDB.",
                "data": None
            }

        except Exception as e:
            return {"status": "error", "message": str(e), "data": None}
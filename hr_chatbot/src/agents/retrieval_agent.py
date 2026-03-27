#hr_chatbot/src/agents/retrieval_agent.py
import os
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase

class RetrievalAgent:
    """
    Fetches relevant data from ChromaDB (Vector) and Neo4j (Graph).
    Uses Open Source Embeddings for vector search.
    """
    def __init__(self, config_dct: Dict[str, Any]):
        """
        Initializes connectors for Vector and Graph databases.
        Args:
            config_dct (Dict): Configuration dictionary containing paths and model settings.
        """
        paths_dct = config_dct["processed_paths"]
        embed_cfg_dct = config_dct["embedding_config"]
        
        self.chroma_path_str = paths_dct["chroma_db_relAbs"]
        
        # Open Source Embeddings (Matching Phase 1)
        self.embedding_func_ins = HuggingFaceEmbeddings(
            model_name=embed_cfg_dct.get("model_name"),
            model_kwargs={'device': embed_cfg_dct.get("device")}
        )
        
        # Neo4j Config
        neo_conf_dct = config_dct["neo4j_config"]
        self.driver_ins = GraphDatabase.driver(
            neo_conf_dct["uri"], 
            auth=(neo_conf_dct["auth_user"], os.getenv("NEO4J_PASSWORD"))
        )

    def process(self, mcp_context_ins: Any) -> Dict[str, str]:
        """
        Performs hybrid retrieval (Vector + Graph) based on the user query.
        Args:
            mcp_context_ins (MCPContext): The shared context packet.
        Returns:
            Dict: Status result.
        """
        query_str = mcp_context_ins.user_query_str
        mcp_context_ins.add_log("Retriever", f"Initiating hybrid search for: {query_str}")
        
        try:
            # 1. Vector Search
            vector_db_ins = Chroma(
                persist_directory=self.chroma_path_str,
                embedding_function=self.embedding_func_ins,
                collection_name="hr_policy_vectors"
            )
            
            results_lst = vector_db_ins.similarity_search(query_str, k=4)
            
            for doc_ins in results_lst:
                mcp_context_ins.retrieved_chunks_lst.append({
                    "content": doc_ins.page_content,
                    "source": doc_ins.metadata.get("source", "unknown"),
                    "page": doc_ins.metadata.get("page", 0)
                })
            
            mcp_context_ins.add_log("Retriever", f"Found {len(results_lst)} vector chunks.")
            
            # 2. Graph Search (Placeholder for Phase 3 advanced traversal)
            mcp_context_ins.add_log("Retriever", "Graph entity lookup skipped (Planned for Phase 3).")
            
            return {"status": "success"}
        except Exception as e:
            mcp_context_ins.add_log("Retriever", f"Error: {str(e)}")
            return {"status": "error"}

    def close(self) -> None:
        """Closes the Neo4j driver connection."""
        self.driver_ins.close()
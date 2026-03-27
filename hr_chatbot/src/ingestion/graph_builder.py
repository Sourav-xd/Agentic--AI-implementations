#hr_chatbot/src/graph_builder.py
from neo4j import GraphDatabase
from typing import List, Tuple, Dict, Any

class Neo4jBuilder:
    """
    Manages interactions with Neo4j Graph Database.
    """

    def __init__(self, uri_str: str, auth_tpl: Tuple[str, str]):
        """
        Initialize Neo4j driver.

        Args:
            uri_str (str): Neo4j URI (e.g., bolt://localhost:7687).
            auth_tpl (Tuple[str, str]): Tuple of (username, password).
        """
        self.driver_ins = GraphDatabase.driver(uri_str, auth=auth_tpl)

    def close(self):
        """Closes the Neo4j driver connection."""
        self.driver_ins.close()

    def create_knowledge_graph(self, chunks_lst: List[Any]) -> Dict[str, Any]:
        """
        Creates a basic Document -> Chunk graph structure.
        
        Nodes:
            - Document (name)
            - Chunk (id, text, page)
        Relationships:
            - (Document)-[:HAS_CHUNK]->(Chunk)

        Args:
            chunks_lst (List[Any]): List of LangChain Document objects.

        Returns:
            Dict[str, Any]: Result dictionary.
        """
        # Cypher query to merge nodes and relationships
        query_str = """
        MERGE (d:Document {name: $doc_name})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text, c.page = $page_num
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        
        try:
            with self.driver_ins.session() as session_ins:
                for idx_int, doc_ins in enumerate(chunks_lst):
                    
                    # Prepare parameters
                    doc_name_str = doc_ins.metadata.get("source", "unknown_doc")
                    page_num_int = doc_ins.metadata.get("page", 0)
                    chunk_id_str = f"{doc_name_str}_chk_{idx_int}"
                    
                    # Execute write transaction
                    session_ins.run(
                        query_str,
                        doc_name=doc_name_str,
                        chunk_id=chunk_id_str,
                        text=doc_ins.page_content,
                        page_num=page_num_int
                    )
            
            return {
                "status": "success", 
                "message": f"Graph nodes created for {len(chunks_lst)} chunks.", 
                "data": None
            }

        except Exception as e:
            return {"status": "error", "message": f"Neo4j Error: {str(e)}", "data": None}
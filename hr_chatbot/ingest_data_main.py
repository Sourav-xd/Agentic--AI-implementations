#hr_chatbot/ingest_data_main.py
import os
import sys
from dotenv import load_dotenv

# Ensure 'src' is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_config, setup_logger
from src.ingestion.chunking import AdvancedPDFProcessor
from src.ingestion.vector_store import ChromaManager
from src.ingestion.graph_builder import Neo4jBuilder

# Load secrets (API Keys, DB Passwords)
load_dotenv("config/secrets.env")

def main():
    """
    Main Orchestrator for Phase 1: Data Ingestion.
    1. Load Config
    2. Chunk PDF (Text + Tables)
    3. Store in Vector DB (Chroma + HuggingFace)
    4. Store in Graph DB (Neo4j)
    """
    
    # --- 1. CONFIGURATION ---
    config_res_dct = load_config("config/app_config.json")
    if config_res_dct["status"] != "success":
        print(f"CRITICAL: {config_res_dct['message']}")
        return

    config_dct = config_res_dct["data"]
    paths_dct = config_dct["processed_paths"]
    
    # Setup Logger
    logger_ins = setup_logger(paths_dct["logs_relAbs"], "ingestion_process")
    logger_ins.info("--- Phase 1: Ingestion Process Started ---")

    # --- 2. CHUNKING ---
    logger_ins.info("Step 1: Starting Document Chunking...")
    processor_ins = AdvancedPDFProcessor(paths_dct["raw_docs_relAbs"])
    
    # NOTE: In production, loop through all files in directory. 
    # For now, targeting specific file as per requirement.
    target_file_str = "HR_Policy_2025.pdf"
    
    pdf_res_dct = processor_ins.load_and_chunk(target_file_str, config_dct["chunking"])
    
    if pdf_res_dct["status"] != "success":
        logger_ins.error(f"PDF Processing Failed: {pdf_res_dct['message']}")
        return
        
    chunks_lst = pdf_res_dct["data"]
    logger_ins.info(f"Step 1 Complete: {pdf_res_dct['message']}")

    # # --- 3. VECTOR STORE (CHROMA) ---
    # logger_ins.info("Step 2: Storing in ChromaDB...")
    # chroma_manager_ins = ChromaManager(
    #     db_path_str=paths_dct["chroma_db_relAbs"],
    #     embedding_config_dct=config_dct["embedding_config"]
    # )
    
    # chroma_res_dct = chroma_manager_ins.store_chunks(chunks_lst)
    
    # if chroma_res_dct["status"] == "success":
    #     logger_ins.info(f"Step 2 Complete: {chroma_res_dct['message']}")
    # else:
    #     logger_ins.error(f"Step 2 Failed: {chroma_res_dct['message']}")

    # # --- 4. KNOWLEDGE GRAPH (NEO4J) ---
    # logger_ins.info("Step 3: Building Knowledge Graph in Neo4j...")
    
    # neo4j_password_str = os.getenv("NEO4J_PASSWORD")
    # if not neo4j_password_str:
    #     logger_ins.warning("NEO4J_PASSWORD not found in env. Skipping Graph Build.")
    # else:
    #     neo_auth_tpl = (config_dct["neo4j_config"]["auth_user"], neo4j_password_str)
    #     graph_builder_ins = Neo4jBuilder(config_dct["neo4j_config"]["uri"], neo_auth_tpl)
        
    #     graph_res_dct = graph_builder_ins.create_knowledge_graph(chunks_lst)
    #     graph_builder_ins.close()
        
    #     if graph_res_dct["status"] == "success":
    #         logger_ins.info(f"Step 3 Complete: {graph_res_dct['message']}")
    #     else:
    #         logger_ins.error(f"Step 3 Failed: {graph_res_dct['message']}")

    # logger_ins.info("--- Phase 1: Ingestion Process Finished ---")

if __name__ == "__main__":
    main()
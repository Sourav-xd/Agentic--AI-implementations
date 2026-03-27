#hr_chatbot/src/ingestion/chunking.py
import os
import pdfplumber
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

class AdvancedPDFProcessor:
    """
    Extracts text AND tables from PDF files.
    Uses Semantic Chunking (Embeddings-based) to group text by meaning rather than character count.
    """

    def __init__(self, raw_docs_path_str: str, embedding_config_dct: Optional[Dict[str, str]] = None):
        """
        Initialize the processor with Embedding model for semantic analysis.

        Args:
            raw_docs_path_str (str): Absolute path to the raw documents directory.
            embedding_config_dct (Dict[str, str], optional): Configuration for the embedding model.
                                                             Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        """
        self.raw_docs_path_str = raw_docs_path_str
        
        # Default config if none provided
        if not embedding_config_dct:
            embedding_config_dct = {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu"
            }

        # Initialize Embedding Model (Required for Semantic Chunking)
        self.embedding_model_ins = HuggingFaceEmbeddings(
            model_name=embedding_config_dct.get("model_name"),
            model_kwargs={'device': embedding_config_dct.get("device")}
        )

    def _extract_page_content(self, page_ins: Any) -> str:
        """
        Helper method to extract text and tables from a single PDF page object.
        
        Args:
            page_ins (Any): pdfplumber page object.

        Returns:
            str: Combined text content (Markdown tables + Raw text).
        """
        # 1. Extract Tables
        tables_lst = page_ins.extract_tables()
        table_text_lst = []
        
        if tables_lst:
            for table_data_lst in tables_lst:
                if not table_data_lst:
                    continue
                try:
                    # Convert list-of-lists to DataFrame then Markdown
                    df_ins = pd.DataFrame(table_data_lst[1:], columns=table_data_lst[0]) 
                    markdown_table_str = df_ins.to_markdown(index=False)
                    table_text_lst.append(f"\n[TABLE DATA_START]\n{markdown_table_str}\n[TABLE DATA_END]\n")
                except Exception:
                    continue
        
        # 2. Extract Normal Text
        raw_text_str = page_ins.extract_text() or ""
        
        # Combine: Tables first, then text
        return "\n".join(table_text_lst) + "\n" + raw_text_str

    def load_and_chunk(self, file_name_str: str, chunk_config_dct: Dict[str, Any]) -> Dict[str, Any]:
        """
        Loads a PDF, extracts content, and splits it using Semantic Chunking.
        
        Logic:
        1. Embeds all sentences.
        2. Calculates distance between sentences.
        3. Splits when distance > threshold (breakpoint).

        Args:
            file_name_str (str): Name of the file to process.
            chunk_config_dct (Dict[str, Any]): Config for semantic splitting.
                - breakpoint_threshold_type (str): 'percentile', 'standard_deviation', 'interquartile'
                - breakpoint_threshold_amount (float): Value to control sensitivity (e.g., 95.0 for percentile).

        Returns:
            Dict[str, Any]: 
                - status: 'success' or 'error'
                - data: List[Document]
                - message: Status details
        """
        full_file_path_str = os.path.join(self.raw_docs_path_str, file_name_str)
        
        if not os.path.exists(full_file_path_str):
            return {"status": "error", "message": f"File not found: {full_file_path_str}", "data": []}

        documents_lst = []
        
        try:
            # 1. Open PDF & Extract Content
            with pdfplumber.open(full_file_path_str) as pdf_ins:
                for i_int, page_ins in enumerate(pdf_ins.pages):
                    page_content_str = self._extract_page_content(page_ins)
                    
                    doc_ins = Document(
                        page_content=page_content_str,
                        metadata={
                            "source": file_name_str, 
                            "page": i_int + 1
                        }
                    )
                    documents_lst.append(doc_ins)

            # 2. Semantic Chunking 
            # This uses the embedding model to calculate semantic distance between sentences.
            semantic_splitter_ins = SemanticChunker(
                embeddings=self.embedding_model_ins,
                breakpoint_threshold_type=chunk_config_dct.get("threshold_type", "percentile"),
                breakpoint_threshold_amount=chunk_config_dct.get("threshold_amount", 95)
            )
            
            # Note: split_documents works, but sometimes merging creates huge chunks.
            # SemanticChunker often works better on raw text, but split_documents preserves metadata better.
            chunked_docs_lst = semantic_splitter_ins.split_documents(documents_lst)
            
            return {
                "status": "success", 
                "data": chunked_docs_lst,
                "message": f"Semantic Chunking complete. Created {len(chunked_docs_lst)} chunks."
            }

        except Exception as e:
            return {"status": "error", "message": f"Semantic Chunking failed: {str(e)}", "data": []}
        

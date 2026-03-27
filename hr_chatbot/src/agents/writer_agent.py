from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class WriterAgent:
    """
    Synthesizes the retrieved context into a professional HR response.
    """
    def __init__(self, config_dct: Dict[str, Any]):
        llm_cfg_dct = config_dct["llm_config"]
        self.llm_ins = ChatGoogleGenerativeAI(
            model=llm_cfg_dct["model_name"], 
            temperature=0.3 # Slightly higher for natural flow
        )

    def process(self, mcp_context_ins: Any) -> Dict[str, str]:
        """
        Generates the final answer.
        Args:
            mcp_context_ins (MCPContext): The shared context packet.
        """
        chunks_lst = mcp_context_ins.retrieved_chunks_lst
        query_str = mcp_context_ins.user_query_str
        
        # Build context string
        context_block_str = "\n\n".join([
            f"--- Source: {c_dct['source']} (Page {c_dct['page']}) ---\n{c_dct['content']}" 
            for c_dct in chunks_lst
        ])
        
        prompt_str = """
        You are a helpful HR Assistant. Use the provided policy context to answer the question.
        If the information is missing, state that you cannot find it in the current policy.

        Context:
        {context}

        Question:
        {question}

        Answer (Professional, clear, and structured):
        """
        
        prompt_tpl_ins = ChatPromptTemplate.from_template(prompt_str)
        chain_ins = prompt_tpl_ins | self.llm_ins
        
        res_msg_ins = chain_ins.invoke({"context": context_block_str, "question": query_str})
        
        mcp_context_ins.final_response_str = res_msg_ins.content
        mcp_context_ins.add_log("Writer", "Final response generated.")
        
        return {"status": "success"}
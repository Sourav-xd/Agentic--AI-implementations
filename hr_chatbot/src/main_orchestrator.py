#hr_chatbot/src/main_orchestrator.py
import uuid
from typing import Dict, Any
from .mcp_schema import MCPContext
from .agents.planner_agent import PlannerAgent
from .agents.retrieval_agent import RetrievalAgent
from .agents.writer_agent import WriterAgent

class Orchestrator:
    """
    The central controller managing the Model Context Protocol (MCP) and agent handoffs.
    """
    def __init__(self, config_dct: Dict[str, Any]):
        self.config_dct = config_dct
        self.planner_ins = PlannerAgent(config_dct)
        self.retriever_ins = RetrievalAgent(config_dct)
        self.writer_ins = WriterAgent(config_dct)

    def run_query(self, user_query_str: str) -> Dict[str, Any]:
        """
        Orchestrates the full flow from Query -> Plan -> Retrieve -> Write.
        Args:
            user_query_str (str): Input from user.
        Returns:
            Dict: Result containing response and context logs.
        """
        session_id_str = str(uuid.uuid4())[:8]
        mcp_ins = MCPContext(session_id_str=session_id_str, user_query_str=user_query_str)
        
        # 1. Plan
        self.planner_ins.process(mcp_ins)
        
        # 2. Retrieve
        self.retriever_ins.process(mcp_ins)
        
        # 3. Write
        self.writer_ins.process(mcp_ins)
        
        # Cleanup
        self.retriever_ins.close()
        
        return {
            "status": "success",
            "answer_str": mcp_ins.final_response_str,
            "logs_lst": mcp_ins.logs_lst,
            "mcp_dct": mcp_ins.to_dict()
        }
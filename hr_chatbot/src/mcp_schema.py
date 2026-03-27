#hr_chatbot/src/mcp_schema.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class MCPContext:
    """
    Model Context Protocol (MCP) Packet.
    Holds the state of the conversation, plans, and retrieved data artifacts.
    """
    session_id_str: str
    user_query_str: str
    
    # Intelligent Planning
    plan_steps_lst: List[str] = field(default_factory=list)
    current_step_int: int = 0
    
    # Retrieved Knowledge (RAG)
    retrieved_chunks_lst: List[Dict[str, Any]] = field(default_factory=list)
    graph_entities_lst: List[Dict[str, Any]] = field(default_factory=list)
    
    # Outputs
    draft_answer_str: str = ""
    final_response_str: str = ""
    confidence_score_flt: float = 0.0
    
    # Metadata & Logs
    logs_lst: List[str] = field(default_factory=list)
    
    def add_log(self, agent_name_str: str, message_str: str) -> None:
        """Adds a timestamped-style log entry to the context."""
        self.logs_lst.append(f"[{agent_name_str}]: {message_str}")
        
    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass to a dictionary for logging/JSON export."""
        return self.__dict__
#hr_chatbot/src/agents/planner_agent.py
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class PlannerAgent:
    """
    Analyzes user query and creates a step-by-step execution plan.
    """
    def __init__(self, config_dct: Dict[str, Any]):
        """
        Args:
            config_dct (Dict): Config containing Gemini model settings.
        """
        llm_cfg_dct = config_dct["llm_config"]
        self.llm_ins = ChatGoogleGenerativeAI(
            model=llm_cfg_dct["model_name"], 
            temperature=llm_cfg_dct["temperature"]
        )

    def process(self, mcp_context_ins: Any) -> Dict[str, str]:
        """
        Generates a logical plan for the HR query.
        Args:
            mcp_context_ins (MCPContext): The shared context packet.
        """
        query_str = mcp_context_ins.user_query_str
        
        prompt_str = """
        You are a Professional HR Strategy Planner. 
        Analyze the query: "{query}"
        Break it down into 3 concise steps to find and explain the policy correctly.
        Format: Return only the steps separated by new lines.
        """
        
        prompt_tpl_ins = ChatPromptTemplate.from_template(prompt_str)
        chain_ins = prompt_tpl_ins | self.llm_ins
        
        res_msg_ins = chain_ins.invoke({"query": query_str})
        plan_str = res_msg_ins.content
        
        mcp_context_ins.plan_steps_lst = plan_str.strip().split("\n")
        mcp_context_ins.add_log("Planner", "Execution strategy defined.")
        
        return {"status": "success"}

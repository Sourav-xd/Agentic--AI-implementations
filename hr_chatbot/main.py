#hr_chatbot/main.py
import os
import sys
from dotenv import load_dotenv

# Path fixing for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_config
from src.main_orchestrator import Orchestrator

load_dotenv("config/secrets.env")

def main():
    """
    Entry point for the Phase 2 Agentic Chat System.
    """
    # 1. Load configuration
    config_res_dct = load_config("config/app_config.json")
    if config_res_dct["status"] != "success":
        print(f"Error loading config: {config_res_dct['message']}")
        return
    
    config_dct = config_res_dct["data"]
    
    # 2. Initialize the Orchestrator (The Brain)
    print("--- Initializing HR Agentic System (Gemini Core) ---")
    orchestrator_ins = Orchestrator(config_dct)
    print("✅ Ready! Type 'exit' to end the session.\n")

    # 3. Chat Loop
    while True:
        user_input_str = input("You: ")
        if user_input_str.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        try:
            result_dct = orchestrator_ins.run_query(user_input_str)
            
            print(f"\n🤖 AI Agent:\n{result_dct['answer_str']}")
            print("-" * 30)
            
        except Exception as e:
            print(f"❌ System Error: {str(e)}")

if __name__ == "__main__":
    main()
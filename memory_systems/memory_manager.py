#memory_manager.py

import json
import os

class PythonicMemory:
    def __init__(self):
        self.short_term_file = "session_buffer.json"
        self.long_term_file = "knowledge_archive.json"
        self._init_files()

    def _init_files(self):
        for f in [self.short_term_file, self.long_term_file]:
            if not os.path.exists(f):
                with open(f, 'w') as file:
                    json.dump([], file)

    def save_interaction(self, user_input, ai_output):
        # 1. Update Short Term (Rolling Buffer of 5)
        with open(self.short_term_file, 'r+') as f:
            data = json.load(f)
            data.append({"user": user_input, "ai": ai_output})
            if len(data) > 5:
                # Move oldest to Long Term
                oldest = data.pop(0)
                self._archive_to_long_term(oldest)
            
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    def _archive_to_long_term(self, entry):
        with open(self.long_term_file, 'r+') as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)

    def get_context(self, current_query):
        # Retrieve Short Term
        with open(self.short_term_file, 'r') as f:
            st_memory = json.load(f)
        
        # Search Long Term (Simple Keyword Retrieval simulation)
        lt_context = []
        with open(self.long_term_file, 'r') as f:
            lt_memory = json.load(f)
            # Logic: If query keywords match past interactions, bring them back
            keywords = current_query.lower().split()
            for entry in lt_memory:
                if any(k in entry['user'].lower() for k in keywords):
                    lt_context.append(entry)

        return st_memory, lt_context
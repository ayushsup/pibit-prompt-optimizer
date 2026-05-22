from src.agents.base_agent import BaseAgent

class Extractor(BaseAgent):
    def extract(self, document_text: str, current_prompt: str, schema: str) -> str:
        sys_prompt = f"{current_prompt}\n\nYou must output JSON conforming to this schema:\n{schema}"
        return self.call_llm(sys_prompt, document_text, role_name="Extractor")

class Critic(BaseAgent):
    def critique(self, document_text: str, predicted_json: str, gold_json: str) -> str:
        sys_prompt = "You are an expert data extraction critic. Identify exactly why the predicted JSON differs from the gold standard JSON. Be concise and specific."
        user_prompt = f"Document Snippet (First 1000 chars):\n{document_text[:1000]}...\n\nPredicted:\n{predicted_json}\n\nGold Standard:\n{gold_json}\n\nWhy did the extraction fail?"
        return self.call_llm(sys_prompt, user_prompt, role_name="Critic")

class Mutator(BaseAgent):
    def mutate(self, current_prompt: str, critiques: list[str], rejected_prompts: list[str] = None) -> str:
        sys_prompt = "You are a prompt engineer. Update the provided prompt to fix the listed extraction errors. Do not degrade general performance. Return ONLY the new prompt text."
        
        formatted_critiques = "\n".join([f"- {c}" for c in critiques])
        user_prompt = f"Current Prompt:\n{current_prompt}\n\nCritiques from failed extractions:\n{formatted_critiques}\n"
        
        # Add memory of what NOT to do
        if rejected_prompts:
            user_prompt += "\nWARNING - The following prompts were already tried and resulted in lower scores. DO NOT propose these:\n"
            for rp in rejected_prompts[-3:]: # Show the last 3 failures
                user_prompt += f"- {rp}\n"
                
        user_prompt += "\nWrite the updated prompt."
        
        return self.call_llm(sys_prompt, user_prompt, role_name="Mutator")
from src.agents.base_agent import BaseAgent

class Extractor(BaseAgent):
    def extract(self, document_text: str, current_prompt: str, schema: str) -> str:
        sys_prompt = f"{current_prompt}\n\nYou must output JSON conforming to this schema:\n{schema}"
        return self.call_llm(sys_prompt, document_text, role_name="Extractor")

class Critic(BaseAgent):
    def critique(self, document_text: str, predicted_json: str, gold_json: str) -> str:
        sys_prompt = "You are an expert data extraction critic. Identify exactly why the predicted JSON differs from the gold standard JSON. Be concise and specific."
        user_prompt = f"Document:\n{document_text}\n\nPredicted:\n{predicted_json}\n\nGold Standard:\n{gold_json}\n\nWhy did the extraction fail?"
        return self.call_llm(sys_prompt, user_prompt, role_name="Critic")

class Mutator(BaseAgent):
    def mutate(self, current_prompt: str, critiques: list[str]) -> str:
        sys_prompt = "You are a prompt engineer. Update the provided prompt to fix the listed extraction errors. Do not degrade general performance. Return ONLY the new prompt."
        formatted_critiques = "\n".join([f"- {c}" for c in critiques])
        user_prompt = f"Current Prompt:\n{current_prompt}\n\nCritiques from failed extractions:\n{formatted_critiques}\n\nWrite the updated prompt."
        return self.call_llm(sys_prompt, user_prompt, role_name="Mutator")
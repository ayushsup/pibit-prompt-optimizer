import os
import time
from openai import OpenAI
from src.core.state_manager import StateManager

class BaseAgent:
    def __init__(self, model_name: str, state_manager: StateManager):
        # Point to Gemini's OpenAI-compatible endpoint
        self.client = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"), 
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model_name = model_name
        self.state_manager = state_manager

    def call_llm(self, system_prompt: str, user_prompt: str, role_name: str) -> str:
        max_retries = 5 # Increased retries to handle unstable network traffic
        base_delay = 15
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2
                )
                
                content = response.choices[0].message.content
                
                # Log the call
                self.state_manager.log_llm_call(
                    role=role_name,
                    prompt=f"SYS: {system_prompt}\nUSER: {user_prompt}",
                    response=content,
                    model=self.model_name,
                    usage=response.usage.model_dump() if response.usage else {},
                    cost=0.0 
                )
                
                # Brief pause to respect standard RPM limits
                time.sleep(4) 
                
                return content
                
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    if "429" in error_msg:
                        # Exponential backoff for rate limits: 15s, 30s, 60s...
                        delay = base_delay * (2 ** attempt) 
                        print(f"⚠️ Rate limit (429) hit for {role_name}. Retrying in {delay}s...")
                        time.sleep(delay)
                    elif "503" in error_msg or "500" in error_msg:
                        # Standard wait for server-side hiccups
                        print(f"⚠️ Server overloaded (503). Retrying {role_name} in 20s...")
                        time.sleep(20)
                    else:
                        # If it's a different error (like a 404 Bad Request), fail immediately
                        raise e 
                else:
                    print(f"❌ Max retries reached for {role_name}. Failing gracefully.")
                    raise e
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
        # Simple retry mechanism to handle Free Tier rate limits gracefully
        max_retries = 3
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
                
                # Log the call (Cost is $0 on the free tier)
                self.state_manager.log_llm_call(
                    role=role_name,
                    prompt=f"SYS: {system_prompt}\nUSER: {user_prompt}",
                    response=content,
                    model=self.model_name,
                    usage=response.usage.model_dump() if response.usage else {},
                    cost=0.0 
                )
                
                # Brief pause to respect the 15 RPM rate limit
                time.sleep(4) 
                
                return content
                
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"Rate limit hit for {role_name}. Sleeping for 60 seconds to reset quota...")
                    time.sleep(60)
                else:
                    raise e
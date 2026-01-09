import os
from groq import Groq

AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        model: str, 
        temperature: float = 0.2, 
        top_p: float = 0.9
    ):
        """
        Generate response from Groq LLM with optional sampling parameters.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content
        except Exception:
            # fallback to smaller model
            fallback = "llama-3.1-8b-instant"
            response = self.client.chat.completions.create(
                model=fallback,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content

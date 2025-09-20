from openai import OpenAI
from config import API_KEY

models = [
    "z-ai/glm-4.5-air:free",
    "deepseek/deepseek-chat-v3.1:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-r1-0528:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "qwen/qwen3-14b:free",
    "qwen/qwen3-8b:free"
]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

def ask_with_fallback(messages):
    for model in models:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages
            )
            print(f"✅ Used model: {model}")
            return resp.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Model {model} failed: {e}")
    raise RuntimeError("All models failed")
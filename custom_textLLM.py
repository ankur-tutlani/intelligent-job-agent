import requests
from typing import List, Dict, Optional
from pydantic import PrivateAttr
from llama_index.core.llms import ChatMessage
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.llms import CustomLLM

MODEL_LIMITS: Dict[str, Dict[str, Optional[int]]] = {
    "z-ai/glm-4.5-air:free": {"context": 131072, "output": None},
    "meta-llama/llama-3.3-70b-instruct:free": {"context": 65536, "output": None},
    "meta-llama/llama-3.3-8b-instruct:free": {"context": 128000, "output": 4028},
    "mistralai/mistral-7b-instruct:free": {"context": 32768, "output": 16384},
    "meta-llama/llama-3.2-3b-instruct:free": {"context": 131072, "output": None},
    "meta-llama/llama-3.1-405b-instruct:free": {"context": 65536, "output": None},
}

DEFAULT_OUTPUT_TOKENS = 1024

class OpenRouterFallbackTextLLM(CustomLLM):
    _api_key: str = PrivateAttr()
    _models: List[str] = PrivateAttr()
    _active_model: str = PrivateAttr()

    def __init__(self, models: List[str], api_key: str, **kwargs):
        super().__init__(**kwargs)
        if not models:
            raise ValueError("Must provide at least one model")
        self._api_key = api_key
        self._models = models
        self._active_model = models[0]

    def _call_chat_api_with_fallback(self, messages: List[ChatMessage]) -> str:
        last_error = None
        for model in self._models:
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": m.role, "content": m.content} for m in messages]
                }
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json"
                }
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60
                )
                resp.raise_for_status()
                data = resp.json()
                self._active_model = model
                print(f"✅ Used chat model: {model}")
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"⚠️ Chat model {model} failed: {e}")
                last_error = e
        raise RuntimeError(f"All chat models failed. Last error: {last_error}")

    def _call_completion_api_with_fallback(self, prompt: str) -> str:
        last_error = None
        for model in self._models:
            try:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": MODEL_LIMITS.get(model, {}).get("output", DEFAULT_OUTPUT_TOKENS)
                }
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json"
                }
                resp = requests.post(
                    "https://openrouter.ai/api/v1/completions",
                    json=payload,
                    headers=headers,
                    timeout=60
                )
                resp.raise_for_status()
                data = resp.json()
                self._active_model = model
                print(f"✅ Used completion model: {model}")
                return data["choices"][0]["text"]
            except Exception as e:
                print(f"⚠️ Completion model {model} failed: {e}")
                last_error = e
        raise RuntimeError(f"All completion models failed. Last error: {last_error}")

    def chat(self, messages: List[ChatMessage], **kwargs) -> CompletionResponse:
        text_out = self._call_chat_api_with_fallback(messages)
        return CompletionResponse(text=text_out)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        text_out = self._call_completion_api_with_fallback(prompt)
        return CompletionResponse(text=text_out)

    async def achat(self, messages: List[ChatMessage], **kwargs) -> CompletionResponse:
        return self.chat(messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        return self.complete(prompt, **kwargs)
    
    def stream_complete(self, prompt: str, **kwargs):
        yield self.complete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs):
        yield await self.acomplete(prompt, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        limits = MODEL_LIMITS.get(self._active_model, {"context": 4096, "output": DEFAULT_OUTPUT_TOKENS})
        return LLMMetadata(
            context_window=limits["context"],
            num_output=limits["output"] or DEFAULT_OUTPUT_TOKENS,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self._active_model
        )
import requests
import mimetypes
from typing import Any, List
from pydantic import PrivateAttr
from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.core.llms import ChatMessage
from llama_index.core.base.llms.types import LLMMetadata, CompletionResponse
from llama_index.core.schema import ImageDocument


# --- Known limits for your 4 models ---
MODEL_LIMITS = {
    "meta-llama/llama-4-scout:free": {
        "context": 128000,
        "output": 4028
    },
    "meta-llama/llama-4-maverick:free": {
        "context": 128000,
        "output": 4028
    },
    "qwen/qwen2.5-vl-32b-instruct:free": {
        "context": 8192,
        "output": None  # No published limit
    },
    "qwen/qwen2.5-vl-72b-instruct:free": {
        "context": 32768,
        "output": None  # No published limit
    }
}

DEFAULT_OUTPUT_TOKENS = 1024  # Safe fallback if output=None


class OpenRouterFallbackMultiModal(MultiModalLLM):
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

    def _detect_mime_type(self, img_doc: ImageDocument) -> str:
        """Detect MIME type from ImageDocument or default to PNG."""
        if hasattr(img_doc, "mime_type") and img_doc.mime_type:
            return img_doc.mime_type
        if hasattr(img_doc, "file_path") and img_doc.file_path:
            mime_type, _ = mimetypes.guess_type(img_doc.file_path)
            if mime_type:
                return mime_type
        return "image/png"

    def _build_payload(self, prompt: str, image_documents: List[ImageDocument] = None, model: str = None):
        content_parts = [{"type": "text", "text": prompt}]
        if image_documents:
            for img_doc in image_documents:
                if getattr(img_doc, "image_url", None):
                    content_parts.append({"type": "image_url", "image_url": {"url": img_doc.image_url}})
                elif getattr(img_doc, "image_base64", None):
                    mime_type = self._detect_mime_type(img_doc)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{img_doc.image_base64}"}
                    })
        return {
            "model": model or self._active_model,
            "messages": [{"role": "user", "content": content_parts}]
        }

    def _call_api_with_fallback(self, payload_builder):
        last_error = None
        for model in self._models:
            try:
                payload = payload_builder(model)
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
                print(f"✅ Used model: {model}")
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"⚠️ Model {model} failed: {e}")
                last_error = e
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    # --- Required sync methods ---
    def complete(self, prompt: str, image_documents: List[ImageDocument] = None, **kwargs) -> CompletionResponse:
        text_out = self._call_api_with_fallback(lambda m: self._build_payload(prompt, image_documents, m))
        return CompletionResponse(text=text_out)

    def stream_complete(self, prompt: str, image_documents: List[ImageDocument] = None, **kwargs):
        yield self.complete(prompt, image_documents, **kwargs)

    def chat(self, messages: List[ChatMessage], **kwargs) -> CompletionResponse:
        def build_payload(model):
            content_parts = [{"type": "text", "text": m.content} for m in messages]
            return {"model": model, "messages": [{"role": "user", "content": content_parts}]}
        text_out = self._call_api_with_fallback(build_payload)
        return CompletionResponse(text=text_out)

    def stream_chat(self, messages: List[ChatMessage], **kwargs):
        yield self.chat(messages, **kwargs)

    # --- Required async methods ---
    async def acomplete(self, prompt: str, image_documents: List[ImageDocument] = None, **kwargs) -> CompletionResponse:
        return self.complete(prompt, image_documents, **kwargs)

    async def astream_complete(self, prompt: str, image_documents: List[ImageDocument] = None, **kwargs):
        yield await self.acomplete(prompt, image_documents, **kwargs)

    async def achat(self, messages: List[ChatMessage], **kwargs) -> CompletionResponse:
        return self.chat(messages, **kwargs)

    async def astream_chat(self, messages: List[ChatMessage], **kwargs):
        yield await self.achat(messages, **kwargs)

    # --- Dynamic metadata ---
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
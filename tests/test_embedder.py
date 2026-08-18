"""Tests for Embedder runtime-config integration."""

import pytest
from types import SimpleNamespace

from app.rag.embedder import Embedder
from app.services.config_service import RuntimeLLMConfig


@pytest.fixture(autouse=True)
def clear_runtime():
    from app.services.config_service import _config_path

    _config_path().unlink(missing_ok=True)
    yield
    _config_path().unlink(missing_ok=True)


class TestEmbedderRuntimeConfig:
    def test_uses_saved_runtime_config(self, monkeypatch):
        saved = RuntimeLLMConfig(
            provider="openai",
            api_key="sk-runtime",
            model="gpt-4o",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            embedding_model="custom-embed",
        )
        monkeypatch.setattr("app.services.config_service.load_runtime_config", lambda: saved)
        emb = Embedder()
        assert emb._mode == "openai"
        assert emb.model == "custom-embed"
        assert emb._cfg_signature == ("sk-runtime", saved.base_url, "custom-embed")

    def test_default_model_by_base_url(self, monkeypatch):
        saved = RuntimeLLMConfig(
            provider="openai",
            api_key="sk-openai",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            embedding_model="",
        )
        monkeypatch.setattr("app.services.config_service.load_runtime_config", lambda: saved)
        emb = Embedder()
        assert emb.model == "text-embedding-3-small"

    def test_rebuild_on_config_change(self, monkeypatch):
        saved = RuntimeLLMConfig(
            provider="openai",
            api_key="sk-one",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            embedding_model="text-embedding-3-small",
        )
        monkeypatch.setattr("app.services.config_service.load_runtime_config", lambda: saved)
        emb = Embedder()
        assert emb.model == "text-embedding-3-small"

        saved.embedding_model = "text-embedding-3-large"
        emb._ensure_config()
        assert emb.model == "text-embedding-3-large"

    def test_independent_embedding_key_priority(self, monkeypatch):
        saved = RuntimeLLMConfig(
            provider="openai",
            api_key="sk-llm",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            embedding_api_key="sk-embed-only",
            embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            embedding_model="text-embedding-v3",
        )
        monkeypatch.setattr("app.services.config_service.load_runtime_config", lambda: saved)
        emb = Embedder()
        assert emb._mode == "openai"
        assert emb.model == "text-embedding-v3"
        assert emb._cfg_signature == (
            "sk-embed-only",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "text-embedding-v3",
        )

    def test_falls_back_to_llm_key(self, monkeypatch):
        saved = RuntimeLLMConfig(
            provider="openai",
            api_key="sk-llm",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            embedding_api_key="",
            embedding_model="",
        )
        monkeypatch.setattr("app.services.config_service.load_runtime_config", lambda: saved)
        emb = Embedder()
        assert emb._cfg_signature == ("sk-llm", "https://api.openai.com/v1", "text-embedding-3-small")

    async def test_embed_openai_sends_float_encoding(self, monkeypatch):
        """Nvidia/OpenRouter embeddings reject base64; must send encoding_format=float."""
        saved = RuntimeLLMConfig(
            provider="openai",
            api_key="sk-llm",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            embedding_api_key="sk-embed",
            embedding_model="text-embedding-3-small",
        )
        monkeypatch.setattr("app.services.config_service.load_runtime_config", lambda: saved)
        emb = Embedder()
        captured = {}

        class FakeEmbeddings:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

        class FakeClient:
            embeddings = FakeEmbeddings()

        emb._openai_client = FakeClient()
        await emb.embed("hello")
        assert captured["encoding_format"] == "float"

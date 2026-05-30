"""
Tests for provider configuration in src/classifier.py.

These tests avoid live API calls and SDK imports by substituting small fake modules.
"""

import sys
from types import SimpleNamespace

from src.classifier import DEFAULT_MODELS, HarmClassifier


def test_groq_provider_is_registered():
    assert DEFAULT_MODELS["groq"] == "llama-3.1-8b-instant"


def test_groq_client_uses_openai_compatible_base_url(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    client = HarmClassifier._build_client("groq", api_key=None)

    assert isinstance(client, FakeOpenAI)
    assert captured["api_key"] == "gsk_test"
    assert str(captured["base_url"]) == "https://api.groq.com/openai/v1"

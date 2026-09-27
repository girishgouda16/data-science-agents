"""Proves _resolve_model produces a model string litellm can actually route,
for every provider — not just the two an allowlist used to name. Offline:
get_llm_provider() is pure string routing, no network, no API key.

Run: python test_model_resolution.py
"""

import os

import litellm

import agent as a

CASES = [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-sonnet-4-20250514"),
    ("openrouter", "deepseek/deepseek-chat"),
    ("ollama", "llama3.1"),
    ("ollama_chat", "llama3.1"),
    ("deepseek", "deepseek-chat"),
    ("groq", "llama-3.3-70b-versatile"),
    ("mistral", "mistral-large-latest"),
    ("gemini", "gemini-2.0-flash"),
    ("xai", "grok-2"),
]


def _with_provider(provider: str, bare: str) -> str:
    before = os.environ.get("LLM_PROVIDER")
    os.environ["LLM_PROVIDER"] = provider
    try:
        return a._resolve_model(bare)
    finally:
        if before is None:
            os.environ.pop("LLM_PROVIDER", None)
        else:
            os.environ["LLM_PROVIDER"] = before


def test_every_provider_routes():
    """The regression that started this: a bare name only auto-resolves for
    openai/anthropic, so anything else raised BadRequestError at call time."""
    for provider, bare in CASES:
        model = _with_provider(provider, bare)
        _, routed, _, _ = litellm.get_llm_provider(model=model)
        # litellm normalizes a few providers to a *_chat variant (e.g. cohere).
        assert routed.startswith(provider) or provider.startswith(routed), (
            f"{provider}/{bare} -> {model!r} routed to {routed!r}"
        )


def test_already_qualified_name_is_not_double_prefixed():
    """An .env that already spells the provider out must not become
    'openrouter/openrouter/...', which routes nowhere."""
    assert (
        _with_provider("openrouter", "openrouter/deepseek/deepseek-chat")
        == "openrouter/deepseek/deepseek-chat"
    )
    assert _with_provider("openai", "gpt-4o-mini") == "openai/gpt-4o-mini"


if __name__ == "__main__":
    test_every_provider_routes()
    test_already_qualified_name_is_not_double_prefixed()
    print("ok")

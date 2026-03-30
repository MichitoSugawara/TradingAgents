from typing import Any, Optional

from .base_client import BaseLLMClient
from .validators import validate_model


class CodexOAuthClient(BaseLLMClient):
    """Client for ChatGPT OAuth Codex models.

    Uses langchain-codex-oauth to authenticate via ChatGPT Plus/Pro OAuth,
    allowing API access without an OPENAI_API_KEY.
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatCodexOAuth instance."""
        try:
            from langchain_codex_oauth import ChatCodexOAuth
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "langchain-codex-oauth is required for llm_provider='codex_oauth'. "
                "Install it with: pip install langchain-codex-oauth\n"
                "Then run: tradingagents auth login"
            ) from exc

        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in (
            "timeout",
            "max_retries",
            "reasoning_effort",
            "max_tokens",
            "temperature",
            "callbacks",
        ):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return ChatCodexOAuth(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for codex_oauth."""
        return validate_model("codex_oauth", self.model)

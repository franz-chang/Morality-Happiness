from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ..settings import env_override, llm_provider_settings, normalize_provider


LLM_PROVIDER_OPTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "rule",
        "label": "规则默认",
        "model": "deterministic-wellbeing-engine",
        "base_url": "",
        "description": "直接根据本场景下的规则实现",
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "model": "gpt-4o-mini",
        "base_url": "",
        "description": "Camel-AI + OpenAI 官方接口，默认读取 OPENAI_API_KEY。",
    },
    {
        "id": "deepseek",
        "label": "DeepSeek-V4-flash",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com/v1",
        "description": "Camel-AI + DeepSeek-V4-flash 行为决策接口。",
    },
    {
        "id": "local-deepseek",
        "label": "本地DeepSeek",
        "model": "deepseek-v4-flash",
        "base_url": "https://uni-api.cstcloud.cn/v1",
        "description": "中国科技云 Uni-API OpenAI-Compatible 接口，调用 deepseek-v4-flash。",
    },
    {
        "id": "qwen3",
        "label": "Qwen3 (Ollama)",
        "model": "qwen3:8b",
        "base_url": "http://localhost:11434",
        "description": "Ollama 原生 /api/chat 接口，可指向远程服务器。",
    },
    {
        "id": "qwen-bailian",
        "label": "阿里云Qwen",
        "model": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "description": "阿里云百炼 OpenAI-Compatible Chat Completions 接口。",
    },
)


def _chat_template_kwargs(settings: dict[str, Any], *, default_thinking: bool | None = None) -> dict[str, Any]:
    raw_kwargs = settings.get("chat_template_kwargs", {})
    kwargs = dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}
    if "thinking" in settings and "thinking" not in kwargs:
        kwargs["thinking"] = bool(settings["thinking"])
    if default_thinking is not None and "thinking" not in kwargs:
        kwargs["thinking"] = default_thinking
    return kwargs


def _openai_extra_body(config: "LLMProviderConfig") -> dict[str, Any]:
    if not config.chat_template_kwargs:
        return {}
    chat_template_kwargs = dict(config.chat_template_kwargs)

    # CSTCloud docs say deepseek-v4-flash defaults to non-thinking when the
    # thinking flag is omitted. Passing chat_template_kwargs at the top level
    # breaks the OpenAI SDK; passing explicit False is unnecessary here.
    if config.provider == "local-deepseek" and chat_template_kwargs.get("thinking") is False:
        chat_template_kwargs.pop("thinking", None)

    if not chat_template_kwargs:
        return {}
    return {"chat_template_kwargs": chat_template_kwargs}


@dataclass
class LLMProviderConfig:
    provider: str = "rule"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.2
    timeout_seconds: int = 60
    max_tokens: int = 256
    response_mode: str = "auto"
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, provider: str = "rule") -> "LLMProviderConfig":
        return cls.from_config(provider)

    @classmethod
    def from_config(cls, provider: str = "rule") -> "LLMProviderConfig":
        provider = normalize_provider(provider)
        settings = llm_provider_settings(provider)
        if provider == "openai":
            return cls(
                provider=provider,
                model=str(env_override("OPENAI_MODEL", settings.get("model", "gpt-4o-mini"))),
                api_key=str(env_override("OPENAI_API_KEY", settings.get("api_key", "replace-me"))),
                base_url=str(env_override("OPENAI_BASE_URL", settings.get("base_url", ""))),
                temperature=float(settings.get("temperature", 0.2)),
                timeout_seconds=int(settings.get("timeout_seconds", 60)),
                max_tokens=int(settings.get("max_tokens", 256)),
                response_mode=str(settings.get("response_mode", "auto")),
                chat_template_kwargs=_chat_template_kwargs(settings),
            )
        if provider == "deepseek":
            return cls(
                provider=provider,
                model=str(env_override("DEEPSEEK_MODEL", settings.get("model", "deepseek-v4-flash"))),
                api_key=str(env_override("DEEPSEEK_API_KEY", settings.get("api_key", "replace-me"))),
                base_url=str(
                    env_override("DEEPSEEK_BASE_URL", settings.get("base_url", "https://api.deepseek.com/v1"))
                ),
                temperature=float(settings.get("temperature", 0.2)),
                timeout_seconds=int(settings.get("timeout_seconds", 60)),
                max_tokens=int(settings.get("max_tokens", 256)),
                response_mode=str(settings.get("response_mode", "auto")),
                chat_template_kwargs=_chat_template_kwargs(settings),
            )
        if provider == "local-deepseek":
            return cls(
                provider=provider,
                model=str(
                    env_override(
                        "LOCAL_DEEPSEEK_MODEL",
                        env_override("CSTCLOUD_MODEL", settings.get("model", "deepseek-v4-flash")),
                    )
                ),
                api_key=str(
                    env_override(
                        "LOCAL_DEEPSEEK_API_KEY",
                        env_override("CSTCLOUD_API_KEY", settings.get("api_key", "这里是API")),
                    )
                ),
                base_url=str(
                    env_override(
                        "LOCAL_DEEPSEEK_BASE_URL",
                        env_override("CSTCLOUD_BASE_URL", settings.get("base_url", "https://uni-api.cstcloud.cn/v1")),
                    )
                ),
                temperature=float(settings.get("temperature", 0.2)),
                timeout_seconds=int(settings.get("timeout_seconds", 60)),
                max_tokens=int(settings.get("max_tokens", 256)),
                response_mode=str(settings.get("response_mode", "instruct")),
                chat_template_kwargs=_chat_template_kwargs(settings, default_thinking=False),
            )
        if provider == "qwen3":
            return cls(
                provider="qwen3",
                model=str(env_override("QWEN3_MODEL", settings.get("model", "qwen3:8b"))),
                api_key=str(env_override("QWEN3_API_KEY", settings.get("api_key", "EMPTY"))),
                base_url=str(
                    env_override("QWEN3_BASE_URL", settings.get("base_url", "http://localhost:11434"))
                ),
                temperature=float(settings.get("temperature", 0.2)),
                timeout_seconds=int(settings.get("timeout_seconds", 60)),
                max_tokens=int(settings.get("max_tokens", 256)),
                response_mode=str(settings.get("response_mode", "auto")),
                chat_template_kwargs=_chat_template_kwargs(settings),
            )
        if provider == "qwen-bailian":
            return cls(
                provider=provider,
                model=str(
                    env_override(
                        "QWEN_BAILIAN_MODEL",
                        env_override("DASHSCOPE_MODEL", settings.get("model", "qwen-plus")),
                    )
                ),
                api_key=str(
                    env_override(
                        "QWEN_BAILIAN_API_KEY",
                        env_override("DASHSCOPE_API_KEY", settings.get("api_key", "这里是API")),
                    )
                ),
                base_url=str(
                    env_override(
                        "QWEN_BAILIAN_BASE_URL",
                        env_override(
                            "DASHSCOPE_BASE_URL",
                            settings.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                        ),
                    )
                ),
                temperature=float(settings.get("temperature", 0.2)),
                timeout_seconds=int(settings.get("timeout_seconds", 60)),
                max_tokens=int(settings.get("max_tokens", 256)),
                response_mode=str(settings.get("response_mode", "instruct")),
                chat_template_kwargs=_chat_template_kwargs(settings),
            )
        return cls(
            provider="rule",
            model=str(settings.get("model", "deterministic-wellbeing-engine")),
            api_key=str(settings.get("api_key", "")),
            base_url=str(settings.get("base_url", "")),
            temperature=float(settings.get("temperature", 0.2)),
            timeout_seconds=int(settings.get("timeout_seconds", 60)),
            max_tokens=int(settings.get("max_tokens", 256)),
            response_mode=str(settings.get("response_mode", "auto")),
            chat_template_kwargs=_chat_template_kwargs(settings),
        )


@dataclass
class LLMProviderStatus:
    framework: str
    provider: str
    model: str
    base_url: str
    active: bool
    using_fallback: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LLMProvider(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class RuleProvider:
    def complete(self, prompt: str) -> str:
        return json.dumps(
            {
                "score": None,
                "reason": "rule-provider: deterministic wellbeing engine is active",
            },
            ensure_ascii=False,
        )


class CamelProvider:
    """Thin CAMEL-AI adapter with graceful fallback when camel-ai is not installed."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        try:
            from camel.agents import ChatAgent
            from camel.models import ModelFactory
            from camel.types import ModelPlatformType
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "camel-ai is not installed. Run `./setup.sh` "
                "or keep llm_provider='rule'."
            ) from exc

        platform = self._platform(ModelPlatformType, config)
        model_config_dict: dict[str, Any] = {"temperature": config.temperature, "max_tokens": config.max_tokens}
        extra_body = _openai_extra_body(config)
        if extra_body:
            model_config_dict["extra_body"] = extra_body

        kwargs = {
            "model_platform": platform,
            "model_type": config.model,
            "model_config_dict": model_config_dict,
        }
        if config.base_url:
            kwargs["url"] = config.base_url
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.timeout_seconds:
            kwargs["timeout"] = float(config.timeout_seconds)
        model = ModelFactory.create(**kwargs)
        self.agent = ChatAgent(
            system_message=(
                "You are a wellbeing evaluator for a computational social science "
                "multi-agent simulation. Return compact JSON only."
            ),
            model=model,
        )

    @staticmethod
    def _platform(model_platform_type: object, config: LLMProviderConfig) -> object:
        provider = config.provider.lower()
        if provider == "openai" and not config.base_url:
            return getattr(model_platform_type, "OPENAI")
        if provider == "deepseek" and hasattr(model_platform_type, "DEEPSEEK"):
            return getattr(model_platform_type, "DEEPSEEK")
        if provider == "qwen3" and not config.base_url and hasattr(model_platform_type, "QWEN"):
            return getattr(model_platform_type, "QWEN")
        return getattr(
            model_platform_type,
            "OPENAI_COMPATIBLE_MODEL",
            getattr(model_platform_type, "OPENAI_COMPATIBILITY_MODEL", None),
        )

    def complete(self, prompt: str) -> str:  # pragma: no cover - requires external API
        response = self.agent.step(prompt)
        if hasattr(response, "msg") and hasattr(response.msg, "content"):
            return str(response.msg.content)
        if hasattr(response, "msgs") and response.msgs:
            return str(response.msgs[0].content)
        return str(response)


class OllamaProvider:
    """Native Ollama /api/chat adapter for remote or local Qwen models."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self.base_url = self._normalize_base_url(config.base_url or "http://localhost:11434")
        if not config.model:
            raise RuntimeError("Ollama provider requires a model name, e.g. qwen3:8b.")

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        cleaned = base_url.strip().rstrip("/")
        if cleaned.endswith("/api/chat"):
            cleaned = cleaned[:-9].rstrip("/")
        if cleaned.endswith("/v1"):
            cleaned = cleaned[:-3].rstrip("/")
        if cleaned.endswith("/api"):
            cleaned = cleaned[:-4].rstrip("/")
        return cleaned or "http://localhost:11434"

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.config.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a behavior decision engine for a multi-agent simulation. "
                        "Return compact JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": self.config.temperature},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key not in {"EMPTY", "replace-me"}:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - requires external service
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama API HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - requires external service
            raise RuntimeError(f"Ollama API unavailable at {url}: {exc}") from exc
        message = body.get("message", {}) if isinstance(body, dict) else {}
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
        if isinstance(body, dict) and body.get("response"):
            return str(body["response"])
        raise RuntimeError("Ollama API response did not include message.content.")


def build_llm_provider(config: LLMProviderConfig | None = None, *, allow_fallback: bool = True) -> LLMProvider:
    config = config or LLMProviderConfig()
    provider = config.provider.lower()
    if provider == "rule":
        return RuleProvider()
    if provider == "qwen3":
        try:
            return OllamaProvider(config)
        except Exception:
            if not allow_fallback:
                raise
            return RuleProvider()
    try:
        return CamelProvider(config)
    except Exception:
        if not allow_fallback:
            raise
        return RuleProvider()


def provider_options() -> list[dict[str, str]]:
    options = []
    for option in LLM_PROVIDER_OPTIONS:
        provider = option["id"]
        settings = llm_provider_settings(provider)
        merged = dict(option)
        for key in ("label", "model", "base_url", "description"):
            if key in settings:
                merged[key] = str(settings[key])
        options.append(merged)
    return options


def provider_status(provider: str) -> LLMProviderStatus:
    config = LLMProviderConfig.from_config(provider)
    if config.provider == "rule":
        return LLMProviderStatus(
            framework="camel-ai",
            provider="rule",
            model="deterministic-wellbeing-engine",
            base_url="",
            active=True,
            using_fallback=False,
            reason="本地规则评分引擎启用；不需要 API key。",
        )
    if config.provider == "qwen3":
        missing_config = not config.model or not config.base_url
        base_url = OllamaProvider._normalize_base_url(config.base_url or "http://localhost:11434")
        return LLMProviderStatus(
            framework="ollama",
            provider=config.provider,
            model=config.model,
            base_url=base_url,
            active=not missing_config,
            using_fallback=missing_config,
            reason=(
                "Ollama 原生 /api/chat 配置缺少 model 或 base_url；模型行为驱动不可用。"
                if missing_config
                else f"Ollama 原生 /api/chat 已配置，将调用 {base_url}/api/chat。"
            ),
        )
    try:
        import camel  # noqa: F401
    except Exception:
        return LLMProviderStatus(
            framework="camel-ai",
            provider=config.provider,
            model=config.model,
            base_url=config.base_url,
            active=False,
            using_fallback=True,
            reason=(
                "未检测到 camel-ai 包，模型行为驱动不可用；"
                "运行 `pip install -e .` 或使用项目 .venv 后可启用 Camel-AI provider。"
            ),
        )
    missing_key = config.api_key in {"", "replace-me", "这里是API"} or (
        config.api_key == "EMPTY" and config.provider != "qwen3"
    )
    mode_note = ""
    if config.provider == "local-deepseek":
        thinking = config.chat_template_kwargs.get("thinking")
        mode_note = f"；response_mode={config.response_mode}, thinking={thinking}"
    return LLMProviderStatus(
        framework="camel-ai",
        provider=config.provider,
        model=config.model,
        base_url=config.base_url,
        active=not missing_key,
        using_fallback=missing_key,
        reason=(
            "Camel-AI provider 已配置，但当前使用缺省 API key；模型行为驱动不可用。"
            if missing_key
            else f"Camel-AI provider 已就绪{mode_note}。"
        ),
    )

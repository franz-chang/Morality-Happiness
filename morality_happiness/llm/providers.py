from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ..settings import env_override, llm_provider_settings, normalize_provider
from .traces import append_llm_trace


_PREFERRED_BASE_URLS: dict[tuple[str, str], str] = {}


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
        "description": "OpenAI-Compatible direct-json + DeepSeek-V4-flash 行为决策接口。",
    },
    {
        "id": "local-deepseek",
        "label": "本地DeepSeek",
        "model": "deepseek-v4-flash",
        "base_url": "https://uni-api.cstcloud.cn/v1",
        "description": "中国科技云 Uni-API OpenAI-Compatible direct-json 接口，优先国内入口并自动备用 Global。",
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


def _normalize_chat_base_url(base_url: str) -> str:
    cleaned = str(base_url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
        return f"{cleaned}/v1"
    return cleaned


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip().rstrip("/") for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip().rstrip("/") for item in value.split(",") if item.strip()]
    return []


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
    max_retries: int = 3
    route_timeout_seconds: int = 0
    fallback_base_urls: list[str] = field(default_factory=list)
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
                base_url=_normalize_chat_base_url(str(env_override("OPENAI_BASE_URL", settings.get("base_url", "")))),
                temperature=float(settings.get("temperature", 0.2)),
                timeout_seconds=int(settings.get("timeout_seconds", 60)),
                max_tokens=int(settings.get("max_tokens", 256)),
                max_retries=int(settings.get("max_retries", 3)),
                route_timeout_seconds=int(settings.get("route_timeout_seconds", 0)),
                fallback_base_urls=_string_list(settings.get("fallback_base_urls", [])),
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
                max_retries=int(settings.get("max_retries", 3)),
                route_timeout_seconds=int(settings.get("route_timeout_seconds", 0)),
                fallback_base_urls=_string_list(settings.get("fallback_base_urls", [])),
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
                        env_override(
                            "CSTCLOUD_BASE_URL",
                            settings.get("base_url", "https://uni-api.cstcloud.cn/v1"),
                        ),
                    )
                ),
                temperature=float(settings.get("temperature", 0.2)),
                timeout_seconds=int(settings.get("timeout_seconds", 60)),
                max_tokens=int(settings.get("max_tokens", 256)),
                max_retries=int(settings.get("max_retries", 0)),
                route_timeout_seconds=int(settings.get("route_timeout_seconds", 12)),
                fallback_base_urls=_string_list(
                    settings.get("fallback_base_urls", ["https://uni-api-global.cstcloud.cn/v1"])
                ),
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
                max_retries=int(settings.get("max_retries", 3)),
                route_timeout_seconds=int(settings.get("route_timeout_seconds", 0)),
                fallback_base_urls=_string_list(settings.get("fallback_base_urls", [])),
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
                max_retries=int(settings.get("max_retries", 3)),
                route_timeout_seconds=int(settings.get("route_timeout_seconds", 0)),
                fallback_base_urls=_string_list(settings.get("fallback_base_urls", [])),
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
            max_retries=int(settings.get("max_retries", 3)),
            route_timeout_seconds=int(settings.get("route_timeout_seconds", 0)),
            fallback_base_urls=_string_list(settings.get("fallback_base_urls", [])),
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
        output = json.dumps(
            {
                "score": None,
                "reason": "rule-provider: deterministic wellbeing engine is active",
            },
            ensure_ascii=False,
        )
        append_llm_trace(
            provider="rule",
            model="deterministic-wellbeing-engine",
            stage="rule",
            prompt=prompt,
            response=output,
            ok=True,
        )
        return output


class DirectJSONProvider:
    """OpenAI-compatible JSON-only adapter used when Camel overhead is undesirable."""

    def __init__(self, config: LLMProviderConfig) -> None:
        if not config.api_key or not config.base_url:
            raise RuntimeError(f"{config.provider} direct JSON provider requires api_key and base_url.")
        self.config = config
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("openai package is required for direct JSON provider.") from exc

        timeout = self._client_timeout()
        urls = self._candidate_base_urls()
        self.clients = [
            (
                base_url,
                OpenAI(
                    api_key=self.config.api_key,
                    base_url=base_url,
                    timeout=timeout,
                    max_retries=max(0, int(self.config.max_retries)),
                ),
            )
            for base_url in urls
        ]
        self.system_message = (
            "You are a behavior decision engine for a multi-agent simulation. "
            "Return exactly one compact JSON object. Do not use markdown."
        )

    def _candidate_base_urls(self) -> list[str]:
        seen = set()
        urls: list[str] = []
        for raw_url in [self.config.base_url, *self.config.fallback_base_urls]:
            base_url = str(raw_url or "").strip().rstrip("/")
            if base_url and base_url not in seen:
                seen.add(base_url)
                urls.append(base_url)
        preferred = _PREFERRED_BASE_URLS.get((self.config.provider, self.config.model))
        if preferred in urls:
            urls.insert(0, urls.pop(urls.index(preferred)))
        return urls

    def _client_timeout(self) -> float:
        timeout = float(self.config.timeout_seconds or 30)
        if self.config.provider == "local-deepseek" and self.config.route_timeout_seconds:
            timeout = min(timeout, float(self.config.route_timeout_seconds))
        return timeout

    def complete(self, prompt: str) -> str:
        min_tokens = 80 if self.config.provider == "local-deepseek" else 128
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": max(int(self.config.max_tokens or 256), min_tokens),
            "response_format": {"type": "json_object"},
        }
        if self.config.provider == "deepseek":
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        last_error: Exception | None = None
        for index, (base_url, client) in enumerate(list(self.clients)):
            stage = f"direct-json:{urllib.parse.urlparse(base_url).netloc or base_url}"
            try:
                response = client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                last_error = exc
                append_llm_trace(
                    provider=self.config.provider,
                    model=self.config.model,
                    stage=stage,
                    prompt=prompt,
                    response="",
                    ok=False,
                    error=str(exc),
                )
                if self.config.provider == "local-deepseek" and index < len(self.clients) - 1:
                    continue
                raise
            message = response.choices[0].message
            content = str(message.content or "").strip()
            if content:
                if index > 0:
                    self.clients.insert(0, self.clients.pop(index))
                _PREFERRED_BASE_URLS[(self.config.provider, self.config.model)] = base_url
                append_llm_trace(
                    provider=self.config.provider,
                    model=self.config.model,
                    stage=stage,
                    prompt=prompt,
                    response=content,
                    ok=True,
                )
                return content
            reasoning = getattr(message, "reasoning_content", None)
            last_error = RuntimeError(f"模型未返回最终JSON内容。reasoning={str(reasoning)[:240] if reasoning else ''}")
            append_llm_trace(
                provider=self.config.provider,
                model=self.config.model,
                stage=stage,
                prompt=prompt,
                response=str(reasoning or ""),
                ok=False,
                error=str(last_error),
            )
            if self.config.provider == "local-deepseek" and index < len(self.clients) - 1:
                continue
            raise last_error
        raise RuntimeError(str(last_error or "No OpenAI-compatible route configured."))


class CamelProvider:
    """Thin CAMEL-AI adapter with graceful fallback when camel-ai is not installed."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self.system_message = (
            "You are a wellbeing evaluator for a computational social science "
            "multi-agent simulation. Return compact JSON only."
        )
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
        kwargs["max_retries"] = max(0, int(config.max_retries))
        model = ModelFactory.create(**kwargs)
        self.agent = ChatAgent(
            system_message=self.system_message,
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
        response = None
        camel_error: Exception | None = None
        try:
            response = self.agent.step(prompt)
            if hasattr(response, "msg") and hasattr(response.msg, "content"):
                content = str(response.msg.content or "").strip()
                if content:
                    append_llm_trace(
                        provider=self.config.provider,
                        model=self.config.model,
                        stage="camel",
                        prompt=prompt,
                        response=content,
                        ok=True,
                    )
                    return content
            if hasattr(response, "msgs") and response.msgs:
                content = str(response.msgs[0].content or "").strip()
                if content:
                    append_llm_trace(
                        provider=self.config.provider,
                        model=self.config.model,
                        stage="camel",
                        prompt=prompt,
                        response=content,
                        ok=True,
                    )
                    return content
        except Exception as exc:
            camel_error = exc

        if self._supports_direct_json_fallback():
            return self._direct_json_complete(prompt)

        if camel_error is not None:
            append_llm_trace(
                provider=self.config.provider,
                model=self.config.model,
                stage="camel",
                prompt=prompt,
                response="",
                ok=False,
                error=str(camel_error),
            )
            raise camel_error
        output = str(response)
        append_llm_trace(
            provider=self.config.provider,
            model=self.config.model,
            stage="camel",
            prompt=prompt,
            response=output,
            ok=bool(output.strip()),
            error="" if output.strip() else "empty response",
        )
        return output

    def _supports_direct_json_fallback(self) -> bool:
        return self.config.provider in {"openai", "qwen-bailian"} and bool(
            self.config.api_key and self.config.base_url
        )

    def _direct_json_complete(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("openai package is required for direct JSON fallback.") from exc

        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=float(self.config.timeout_seconds or 30),
            max_retries=max(0, int(self.config.max_retries)),
        )
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.system_message + " Return exactly one JSON object."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": max(int(self.config.max_tokens or 256), 1024),
            "response_format": {"type": "json_object"},
        }
        if self.config.provider == "deepseek":
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        content = str(message.content or "").strip()
        if content:
            append_llm_trace(
                provider=self.config.provider,
                model=self.config.model,
                stage="direct-json-fallback",
                prompt=prompt,
                response=content,
                ok=True,
            )
            return content
        reasoning = getattr(message, "reasoning_content", None)
        error = f"模型未返回最终JSON内容。reasoning={str(reasoning)[:240] if reasoning else ''}"
        append_llm_trace(
            provider=self.config.provider,
            model=self.config.model,
            stage="direct-json-fallback",
            prompt=prompt,
            response=str(reasoning or ""),
            ok=False,
            error=error,
        )
        raise RuntimeError(error)


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
            append_llm_trace(
                provider=self.config.provider,
                model=self.config.model,
                stage="ollama",
                prompt=prompt,
                response=detail[:500],
                ok=False,
                error=f"Ollama API HTTP {exc.code}",
            )
            raise RuntimeError(f"Ollama API HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - requires external service
            append_llm_trace(
                provider=self.config.provider,
                model=self.config.model,
                stage="ollama",
                prompt=prompt,
                response="",
                ok=False,
                error=f"Ollama API unavailable at {url}: {exc}",
            )
            raise RuntimeError(f"Ollama API unavailable at {url}: {exc}") from exc
        message = body.get("message", {}) if isinstance(body, dict) else {}
        if isinstance(message, dict) and message.get("content"):
            content = str(message["content"])
            append_llm_trace(
                provider=self.config.provider,
                model=self.config.model,
                stage="ollama",
                prompt=prompt,
                response=content,
                ok=True,
            )
            return content
        if isinstance(body, dict) and body.get("response"):
            content = str(body["response"])
            append_llm_trace(
                provider=self.config.provider,
                model=self.config.model,
                stage="ollama",
                prompt=prompt,
                response=content,
                ok=True,
            )
            return content
        append_llm_trace(
            provider=self.config.provider,
            model=self.config.model,
            stage="ollama",
            prompt=prompt,
            response=json.dumps(body, ensure_ascii=False)[:500],
            ok=False,
            error="Ollama API response did not include message.content.",
        )
        raise RuntimeError("Ollama API response did not include message.content.")


def build_llm_provider(config: LLMProviderConfig | None = None, *, allow_fallback: bool = True) -> LLMProvider:
    config = config or LLMProviderConfig()
    provider = config.provider.lower()
    if provider == "rule":
        return RuleProvider()
    if provider in {"deepseek", "local-deepseek"}:
        try:
            return DirectJSONProvider(config)
        except Exception:
            if not allow_fallback:
                raise
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
    if config.provider in {"deepseek", "local-deepseek"}:
        missing_key = config.api_key in {"", "replace-me", "这里是API"}
        provider_label = "DeepSeek" if config.provider == "deepseek" else "本地DeepSeek"
        mode_note = ""
        if config.provider == "local-deepseek":
            thinking = config.chat_template_kwargs.get("thinking")
            fallback_note = f", fallback={len(config.fallback_base_urls)}" if config.fallback_base_urls else ""
            route_timeout = config.route_timeout_seconds or config.timeout_seconds
            mode_note = (
                f"；response_mode={config.response_mode}, thinking={thinking}, "
                f"route_timeout={route_timeout}s{fallback_note}"
            )
        return LLMProviderStatus(
            framework="openai-compatible",
            provider=config.provider,
            model=config.model,
            base_url=config.base_url,
            active=not missing_key,
            using_fallback=missing_key,
            reason=(
                f"{provider_label} direct-json provider 已配置，但当前使用缺省 API key；模型行为驱动不可用。"
                if missing_key
                else f"{provider_label} direct-json provider 已就绪；不经过 Camel-AI{mode_note}。"
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

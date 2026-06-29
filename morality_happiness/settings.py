from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.getenv("MORALITY_HAPPINESS_CONFIG", PROJECT_ROOT / "config.yaml"))


def load_app_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    text = CONFIG_PATH.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]
    except Exception:
        return _parse_simple_yaml(text)
    loaded = yaml.safe_load(text)
    return loaded if isinstance(loaded, dict) else {}


def simulation_defaults() -> dict[str, Any]:
    config = load_app_config()
    defaults = dict(_dict_at(config, "simulation"))
    llm_config = _dict_at(config, "llm")
    if "default_provider" in llm_config:
        defaults["llm_provider"] = llm_config["default_provider"]
    if "sample_rate" in llm_config:
        defaults["llm_sample_rate"] = llm_config["sample_rate"]
    return defaults


def server_defaults() -> dict[str, Any]:
    return dict(_dict_at(load_app_config(), "server"))


def llm_defaults() -> dict[str, Any]:
    return dict(_dict_at(load_app_config(), "llm"))


def llm_provider_settings(provider: str) -> dict[str, Any]:
    provider = normalize_provider(provider)
    llm_config = llm_defaults()
    provider_configs = _dict_at(llm_config, "providers")
    settings = dict(_dict_at(provider_configs, provider))
    settings.setdefault("provider", provider)
    settings.setdefault("temperature", llm_config.get("temperature", 0.2))
    settings.setdefault("timeout_seconds", llm_config.get("timeout_seconds", 60))
    settings.setdefault("max_tokens", llm_config.get("max_tokens", 256))
    if "max_retries" in llm_config:
        settings.setdefault("max_retries", llm_config["max_retries"])
    return settings


def normalize_provider(provider: str) -> str:
    normalized = str(provider or "rule").lower().replace("_", "-")
    aliases = {
        "default": "rule",
        "local": "qwen3",
        "qwen-local": "qwen3",
        "local-qwen3": "qwen3",
        "local-deepseek": "local-deepseek",
        "deepseek-local": "local-deepseek",
        "cstcloud": "local-deepseek",
        "cstcloud-deepseek": "local-deepseek",
        "dashscope": "qwen-bailian",
        "bailian": "qwen-bailian",
        "aliyun-qwen": "qwen-bailian",
        "qwen-aliyun": "qwen-bailian",
        "bailian-qwen": "qwen-bailian",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"rule", "openai", "deepseek", "local-deepseek", "qwen3", "qwen-bailian"} else "rule"


def env_override(name: str, current: Any) -> Any:
    value = os.getenv(name)
    return current if value is None else value


def _dict_at(values: dict[str, Any], key: str) -> dict[str, Any]:
    item = values.get(key, {})
    return item if isinstance(item, dict) else {}


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = _strip_inline_comment(raw_line.rstrip())
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("- "):
            raise ValueError("config.yaml fallback parser supports mappings, not lists")
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or line[index - 1].isspace():
                return line[:index].rstrip()
    return line


def _parse_scalar(value: str) -> Any:
    if value in {'""', "''"}:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value

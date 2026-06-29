from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import SCENARIOS, SimulationParameters
from .llm import (
    LLMProviderConfig,
    build_llm_provider,
    clear_llm_traces,
    list_llm_traces,
    provider_options,
    provider_status,
)
from .settings import server_defaults
from .simulation import run_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOP_EVENT = threading.Event()


class APIHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/scenarios":
            self._json({"scenarios": [scenario.to_dict() for scenario in SCENARIOS.values()]})
            return
        if self.path == "/api/defaults":
            self._json(
                {
                    "parameters": SimulationParameters.from_dict({}).to_dict(),
                    "model_options": provider_options(),
                }
            )
            return
        if self.path.startswith("/api/llm-traces"):
            self._json({"items": list_llm_traces(self._trace_limit())})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/stop":
            STOP_EVENT.set()
            self._json({"stopped": True})
            return
        if self.path == "/api/llm-traces/clear":
            clear_llm_traces()
            self._json({"ok": True, "cleared": True})
            return
        if self.path == "/api/model-test":
            try:
                payload = self._read_json()
                self._json(self._test_model_route(payload))
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, status=400)
            return
        if self.path not in {"/api/run", "/api/run-stream"}:
            self.send_error(404)
            return
        try:
            payload = self._read_json()
            scenario_ids = payload.get("scenario_ids") or list(SCENARIOS)
            scenario_ids = [str(item).upper() for item in scenario_ids]
            params = SimulationParameters.from_dict(payload.get("parameters", {}))
            STOP_EVENT.clear()
            if self.path == "/api/run-stream":
                self._run_stream(scenario_ids, params)
                return
            result = run_batch(scenario_ids, params, stop_checker=STOP_EVENT.is_set)
        except Exception as exc:
            self._json({"error": str(exc)}, status=400)
            return
        self._json(result)

    def _run_stream(self, scenario_ids: list[str], params: SimulationParameters) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        current_scenario_id = scenario_ids[0] if scenario_ids else "S1"
        current_tick = 0
        current_sequence = 0

        def write_event(event: dict[str, Any]) -> None:
            nonlocal current_scenario_id, current_tick, current_sequence
            if event.get("type") == "scenario_start" and isinstance(event.get("scenario"), dict):
                current_scenario_id = str(event["scenario"].get("id", current_scenario_id))
                current_tick = 0
                current_sequence = 0
            elif event.get("type") == "log" and isinstance(event.get("log"), dict):
                current_scenario_id = str(event.get("scenario_id", event["log"].get("scenario_id", current_scenario_id)))
                current_tick = int(event["log"].get("tick", current_tick) or 0)
                current_sequence = int(event["log"].get("sequence", current_sequence) or 0)
            raw = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            self.wfile.write(raw)
            self.wfile.flush()

        try:
            run_batch(scenario_ids, params, event_callback=write_event, stop_checker=STOP_EVENT.is_set)
        except Exception as exc:
            try:
                error_text = str(exc)
                write_event(
                    {
                        "type": "log",
                        "target": "system_logs",
                        "scenario_id": current_scenario_id,
                        "log": {
                            "tick": current_tick,
                            "scenario_id": current_scenario_id,
                            "sequence": current_sequence + 1,
                            "level": "error",
                            "event": "simulation_error",
                            "message": f"运行错误: {error_text}",
                        },
                    }
                )
                write_event({"type": "error", "scenario_id": current_scenario_id, "error": error_text, "logged": True})
            except (BrokenPipeError, ConnectionResetError):
                return

    def _test_model_route(self, payload: dict[str, Any]) -> dict[str, Any]:
        params = SimulationParameters.from_dict(payload.get("parameters", {}))
        provider = str(payload.get("provider") or params.llm_provider or "rule")
        status = provider_status(provider)
        config = LLMProviderConfig.from_config(provider)
        started = time.perf_counter()
        diagnostics: dict[str, Any] = {}

        if config.provider == "rule":
            return {
                "ok": True,
                "provider": status.provider,
                "model": status.model,
                "base_url": status.base_url,
                "latency_ms": 0,
                "message": "本地规则引擎可用；无需外部 API。",
                "status": status.to_dict(),
            }

        if status.using_fallback or not status.active:
            return {
                "ok": False,
                "provider": status.provider,
                "model": status.model,
                "base_url": status.base_url,
                "latency_ms": 0,
                "message": status.reason,
                "status": status.to_dict(),
            }

        if config.provider == "local-deepseek":
            diagnostics = self._probe_openai_compatible_route(config)

        config.timeout_seconds = min(config.timeout_seconds, 20)
        config.max_tokens = min(config.max_tokens, 64)
        config.temperature = 0.0
        provider_impl = build_llm_provider(config, allow_fallback=False)
        try:
            raw = provider_impl.complete('Return compact JSON only: {"ok": true, "route": "test"}')
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            message = str(exc)
            if diagnostics.get("models_ok"):
                model_list = diagnostics.get("models_sample", [])
                message = (
                    f"模型列表探测正常，但 chat/completions 调用失败: {exc}"
                    + (f"；已探测到模型: {', '.join(model_list[:5])}" if model_list else "")
                )
            return {
                "ok": False,
                "provider": status.provider,
                "model": config.model,
                "base_url": config.base_url,
                "latency_ms": latency_ms,
                "message": message,
                "error": str(exc),
                "diagnostics": diagnostics,
                "status": status.to_dict(),
            }
        latency_ms = int((time.perf_counter() - started) * 1000)
        is_valid = bool(str(raw).strip())
        return {
            "ok": is_valid,
            "provider": status.provider,
            "model": config.model,
            "base_url": config.base_url,
            "latency_ms": latency_ms,
            "message": "模型通路测试成功。" if is_valid else "模型返回为空。",
            "sample": str(raw).strip()[:240],
            "diagnostics": diagnostics,
            "status": status.to_dict(),
        }

    def _probe_openai_compatible_route(self, config: LLMProviderConfig) -> dict[str, Any]:
        result: dict[str, Any] = {
            "models_ok": False,
            "model_present": False,
            "models_sample": [],
        }
        if not config.base_url or not config.api_key:
            return result
        url = config.base_url.rstrip("/") + "/models"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {config.api_key}"}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=min(config.timeout_seconds, 8)) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            result["models_error"] = f"HTTP {exc.code}: {detail[:300]}"
            return result
        except Exception as exc:
            result["models_error"] = str(exc)
            return result

        data = body.get("data", []) if isinstance(body, dict) else []
        model_ids = [
            str(item.get("id"))
            for item in data
            if isinstance(item, dict) and item.get("id")
        ]
        result["models_ok"] = True
        result["models_sample"] = model_ids[:8]
        result["model_present"] = config.model in model_ids
        return result

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _trace_limit(self) -> int:
        try:
            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(self.path).query)
            raw = query.get("limit", ["80"])[0]
            return max(1, min(200, int(raw)))
        except Exception:
            return 80


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the morality-happiness API and frontend.")
    defaults = server_defaults()
    parser.add_argument("--host", default=str(defaults.get("host", "127.0.0.1")))
    parser.add_argument("--port", type=int, default=int(defaults.get("port", 8000)))
    args = parser.parse_args(argv)
    static_dir = PROJECT_ROOT / "frontend" / "dist"
    if not static_dir.exists():
        static_dir = PROJECT_ROOT / "frontend"
    handler = partial(APIHandler, directory=str(static_dir))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

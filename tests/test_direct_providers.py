import unittest
from types import SimpleNamespace
from unittest.mock import patch

from morality_happiness.llm.providers import (
    DirectJSONProvider, LLMProviderConfig, build_llm_provider, provider_status,
)


class DirectProviderTest(unittest.TestCase):
    def config(self, provider="openai", **kwargs):
        return LLMProviderConfig(
            provider=provider, model="test-model", api_key="test-key",
            base_url="https://example.invalid/v1", **kwargs,
        )

    @patch("morality_happiness.llm.providers.append_llm_trace")
    @patch("openai.OpenAI")
    def test_all_compatible_providers_are_stateless_and_reuse_client(self, factory, trace):
        factory.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
        )
        for name in ("openai", "deepseek", "local-deepseek", "qwen-bailian"):
            factory.reset_mock()
            with patch.dict("sys.modules", {"camel": None}):
                provider = build_llm_provider(self.config(name), allow_fallback=False)
                self.assertIsInstance(provider, DirectJSONProvider)
                self.assertEqual(provider.complete("first"), '{"ok":true}')
                provider.complete("second")
            factory.assert_called_once()
            calls = factory.return_value.chat.completions.create.call_args_list
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(calls[-1].kwargs["messages"]), 2)
            self.assertEqual(calls[-1].kwargs["messages"][-1]["content"], "second")
            self.assertEqual(calls[-1].kwargs["model"], "test-model")

    @patch("morality_happiness.llm.providers.append_llm_trace")
    @patch("openai.OpenAI")
    def test_errors_do_not_trigger_second_adapter(self, factory, trace):
        factory.return_value.chat.completions.create.side_effect = RuntimeError("unavailable")
        provider = build_llm_provider(self.config(), allow_fallback=False)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            provider.complete("test")
        factory.return_value.chat.completions.create.assert_called_once()

    @patch("morality_happiness.llm.providers.append_llm_trace")
    @patch("openai.OpenAI")
    def test_instruct_and_extra_body(self, factory, trace):
        factory.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
        )
        provider = DirectJSONProvider(self.config(
            response_mode="instruct", chat_template_kwargs={"enable_thinking": False},
            timeout_seconds=17, max_retries=0,
        ))
        provider.complete("test")
        kwargs = factory.return_value.chat.completions.create.call_args.kwargs
        self.assertNotIn("response_format", kwargs)
        self.assertEqual(kwargs["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(factory.call_args.kwargs["timeout"], 17)
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)

    @patch("openai.OpenAI")
    def test_official_url_default(self, factory):
        DirectJSONProvider(LLMProviderConfig(provider="openai", api_key="test-key"))
        self.assertEqual(factory.call_args.kwargs["base_url"], "https://api.openai.com/v1")

    def test_rule_status_is_native(self):
        self.assertEqual(provider_status("rule").framework, "native")

    def test_missing_key_fails_without_fallback(self):
        with self.assertRaises(RuntimeError):
            build_llm_provider(LLMProviderConfig(provider="openai"), allow_fallback=False)

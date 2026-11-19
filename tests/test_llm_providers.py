"""Offline unit tests for llm_providers (no network): python -m unittest discover -s tests -v"""
import base64
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from llm_providers import message_media, usage
from llm_providers.llm_cache import LLMCache


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class DispatchTest(unittest.TestCase):
    def test_prefixed_models_use_shared_provider_routing(self):
        from llm_providers.dispatch import call_llm, call_llm_provider

        messages = [{"role": "user", "content": "hi"}]
        cases = [
            ("aws-eu.model", "aws_bedrock", "call_llm", "bedrock_model_id", "eu.model", True),
            ("gemini-2.5-flash", "gemini", "call_gemini", "model", "gemini-2.5-flash", True),
            ("or-provider/model", "openrouter", "call_openrouter", "model", "provider/model", True),
            ("azure-deployment", "azure_openai", "call_llm", "azure_deployment_model", "deployment", True),
            ("copilot-model", "copilot_cli", "call_copilot", "model", "model", False),
        ]
        for model, module_name, function_name, model_arg, provider_model, has_temperature in cases:
            with self.subTest(model=model):
                provider = types.ModuleType("llm_providers." + module_name)
                provider_call = mock.Mock(return_value="answer")
                setattr(provider, function_name, provider_call)
                with mock.patch.dict(sys.modules, {provider.__name__: provider}):
                    self.assertEqual(call_llm(messages, model, 123, 0, "high",
                                              conversation_key="planner"), "answer")
                    self.assertEqual(call_llm_provider(None, model, messages, 123, "high",
                                                       conversation_key="planner"), "answer")
                kwargs = {model_arg: provider_model, "max_tokens": 123, "reasoning_effort": "high"}
                if has_temperature:
                    kwargs["temperature"] = 0
                else:
                    kwargs["conversation_key"] = "planner"
                self.assertEqual(provider_call.call_count, 2)
                provider_call.assert_called_with(messages, **kwargs)

    def test_unprefixed_client_stream_and_unknown_model(self):
        from llm_providers.dispatch import call_llm, call_llm_provider

        messages = [{"role": "user", "content": "hi"}]
        client = mock.Mock()
        client.chat.completions.create.return_value = [
            types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=content))])
            for content in ("hello", None, " world")
        ]
        self.assertEqual(call_llm_provider(client, "plain-model", messages, 123, "low"), "hello world")
        client.chat.completions.create.assert_called_once_with(
            model="plain-model", temperature=0, messages=messages, stream=True
        )
        with self.assertRaisesRegex(ValueError, "Unknown provider prefix"):
            call_llm(messages, "plain-model")

    def test_response_format_is_forwarded_or_rejected(self):
        from llm_providers.dispatch import call_llm

        provider = types.ModuleType("llm_providers.openrouter")
        provider.call_openrouter = mock.Mock(return_value="{}")
        messages = [{"role": "user", "content": "hi"}]
        response_format = {"type": "json_object"}
        with mock.patch.dict(sys.modules, {provider.__name__: provider}):
            self.assertEqual(call_llm(messages, "or-provider/model", response_format=response_format), "{}")
        self.assertEqual(provider.call_openrouter.call_args.kwargs["response_format"], response_format)
        with self.assertRaisesRegex(ValueError, "only for OpenRouter"):
            call_llm(messages, "gemini-2.5-flash", response_format=response_format)


class OpenRouterTest(unittest.TestCase):
    def setUp(self):
        usage.reset()

    def test_payload_usage_and_response_format(self):
        from llm_providers import openrouter

        seen = {}

        def fake_post(url, headers, payload):
            seen.clear()
            seen.update(payload)
            return FakeResponse({"choices": [{"message": {"content": "{\"ok\": 1}"}}],
                                 "usage": {"prompt_tokens": 11, "completion_tokens": 5, "cost": 0.002}})

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), \
                mock.patch.object(openrouter, "post_with_retries", fake_post):
            out = openrouter.call_openrouter([{"role": "user", "content": "hi"}], model="m/x",
                                             response_format={"type": "json_object"})
            out2 = openrouter.call_openrouter([{"role": "user", "content": "hi"}], model="m/x")
        self.assertEqual(out, "{\"ok\": 1}")
        self.assertEqual(out2, out)
        self.assertNotIn("response_format", seen)  # last call: default payload unchanged
        t = usage.totals()["openrouter:m/x"]
        self.assertEqual((t["calls"], t["input_tokens"], t["output_tokens"]), (2, 22, 10))
        self.assertAlmostEqual(t["cost"], 0.004)

    def test_requires_key(self):
        from llm_providers import openrouter

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                openrouter.call_openrouter([{"role": "user", "content": "hi"}], model="m/x")


class BedrockTest(unittest.TestCase):
    def setUp(self):
        usage.reset()

    def test_content_conversion_and_usage(self):
        try:
            from llm_providers import aws_bedrock
        except ImportError:
            self.skipTest("boto3 not installed")
        img = "data:image/png;base64," + base64.b64encode(b"png").decode()
        parts = aws_bedrock._oai_content_to_bedrock([{"type": "text", "text": "t"},
                                                     {"type": "image_url", "image_url": {"url": img}}])
        self.assertEqual(parts[1]["source"]["media_type"], "image/png")

        body = {"content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "answer"}],
                "usage": {"input_tokens": 7, "output_tokens": 3}}
        client = mock.Mock()
        client.invoke_model.return_value = {"body": io.BytesIO(json.dumps(body).encode())}
        with mock.patch.object(aws_bedrock, "_build_client", return_value=client):
            out = aws_bedrock.call_llm([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                                       bedrock_model_id="eu.anthropic.claude-x")
        self.assertEqual(out, "answer")
        sent = json.loads(client.invoke_model.call_args.kwargs["body"])
        self.assertEqual(sent["system"], "s")
        t = usage.totals()["bedrock:eu.anthropic.claude-x"]
        self.assertEqual((t["input_tokens"], t["output_tokens"]), (7, 3))


class UsageLogTest(unittest.TestCase):
    def test_jsonl_log(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "u", "usage.jsonl")
            usage.set_log_path(p)
            try:
                usage.record("openrouter", "m", 1, 2, None, 0.1)
            finally:
                usage.set_log_path(None)
            rows = [json.loads(x) for x in open(p, encoding="utf-8")]
        self.assertEqual(rows[0]["output_tokens"], 2)


class MediaAndCacheTest(unittest.TestCase):
    def test_append_to_messages_with_image(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.png")
            with open(p, "wb") as f:
                f.write(b"\x89PNG")
            msgs = message_media.append_to_messages("look", [p])
        self.assertEqual(msgs[0]["content"][1]["type"], "image_url")
        self.assertTrue(msgs[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_cache_hit_ignores_image_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            c = LLMCache(d, logger=mock.Mock())
            m1 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
            m2 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,BB"}}]}]
            self.assertEqual(c.get("m", m1, {}, None, lambda: "first"), "first")
            self.assertEqual(c.get("m", m2, {}, None, lambda: "second"), "first")


if __name__ == "__main__":
    unittest.main()

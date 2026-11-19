# llm_providers

Shared LLM/VLM provider clients, extracted from `language-models-trajectory-generators/providers/llms`
so several repos (trajectory generator, Excel table parsing fine-tuning, ...) use one implementation.

| Module | Entry point | Notes |
|---|---|---|
| `openrouter.py` | `call_openrouter(messages, model, max_tokens, temperature, reasoning_effort, response_format=None)` | `OPENROUTER_API_KEY`; optional JSON `response_format` |
| `dispatch.py` | `call_llm(messages, model, max_tokens, temperature, reasoning_effort, conversation_key=None, client=None, response_format=None)` | provider by model prefix: `or-`, `aws-`, `azure-`, `gemini-`, `copilot-`; unprefixed models use an OpenAI-compatible client |
| `batch.py` | `wrap_sync_call_llm_many_async(prompts, contexts, model_name, max_new_tokens, base_url, apiKey)` | concurrent batch over `dispatch.call_llm` (threads + semaphore, `max_concurrency=8`); sends `prompt + '\n' + context`; returns `[{'answer', 'model', 'prompt', 'context'}]`; `base_url`/`apiKey` ignored |
| `aws_bedrock.py` | `call_llm(messages, bedrock_model_id=None, ...)` | boto3 (`[bedrock]` extra); AWS profile/SSO, optional guardrail env vars |
| `azure_openai.py` | `call_llm(messages, azure_deployment_model=None, ...)` | `AZURE_OPENAI_*` |
| `gemini.py` | `call_gemini(messages, model, ...)` | `GEMINI_API_KEY` |
| `copilot_cli.py` | `call_copilot(messages, model, ...)` | shells out to the `copilot` CLI (`COPILOT_CLI_PATH` optional) |
| `llm_cache.py` | response cache | |
| `message_media.py` | `append_to_messages`, `append_images`, `append_videos`, `encode_media` | OpenAI-style multimodal messages |
| `model_registry.py` | `get_model_info(model)` | |
| `usage.py` | `record`, `totals`, `reset`, `set_log_path` | token/cost/latency tracking (OpenRouter + Bedrock record automatically); JSONL log via `LLM_PROVIDERS_USAGE_LOG` |

Provider selection by model-name prefix (`or-`, `aws-`, `azure-`, `gemini-`, `copilot-`): `dispatch.call_llm`.
AWS, Azure, OpenRouter and Copilot prefixes are stripped; Gemini receives the full model name.
`conversation_key` selects a Copilot conversation. For existing client-first callers,
`dispatch.call_llm_provider(client, model, messages, max_tokens, reasoning_effort, conversation_key=None)`
delegates to the same dispatcher. `response_format` is supported only for OpenRouter; other
providers raise `ValueError` rather than silently ignoring it.
Bedrock Claude Sonnet 5.5 maps `reasoning_effort` to adaptive thinking `output_config.effort`
(`low`, `medium`, `high`, `xhigh`, `max`); temperature is omitted because it is incompatible
with adaptive thinking. Other Bedrock models currently ignore `reasoning_effort`.

## Install

Either put the repo root on `PYTHONPATH` (what the consumers' `setup_env\setup_everytime.bat` do):

```bat
set PYTHONPATH=%PYTHONPATH%;D:\NLP\llm_providers
```

or install editable: `pip install -e .` (add `[bedrock]` for boto3). Python >= 3.9.

## Configuration

Copy `.env.example` to `.env` (gitignored) or export the variables. **Never commit `.env` or real keys.**
Modules call `load_dotenv()` on import.

## Usage

```python
from llm_providers.dispatch import call_llm

messages = [{"role": "user", "content": "hi"}]
text = call_llm(messages, "gemini-2.5-flash", max_tokens=4096)
# Use "aws-<bedrock-model-id>", "or-<model>", "azure-<deployment>", or
# "copilot-<model>" to select another provider.
```

`examples/aws_bedrock_post.py` is a standalone Bedrock example (model/guardrail from env vars).

## Tests (offline, no API calls)

```bat
python -m unittest discover -s tests -v
```

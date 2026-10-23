# llm_providers

Shared LLM/VLM provider clients, extracted from `language-models-trajectory-generators/providers/llms`
so several repos (trajectory generator, Excel table parsing fine-tuning, ...) use one implementation.

| Module | Entry point | Notes |
|---|---|---|
| `openrouter.py` | `call_openrouter(messages, model, max_tokens, temperature, reasoning_effort, response_format=None)` | `OPENROUTER_API_KEY`; optional JSON `response_format` |
| `aws_bedrock.py` | `call_llm(messages, bedrock_model_id=None, ...)` | boto3 (`[bedrock]` extra); AWS profile/SSO, optional guardrail env vars |
| `azure_openai.py` | `call_llm(messages, azure_deployment_model=None, ...)` | `AZURE_OPENAI_*` |
| `gemini.py` | `call_gemini(messages, model, ...)` | `GEMINI_API_KEY` |
| `copilot_cli.py` | `call_copilot(messages, model, ...)` | shells out to the `copilot` CLI (`COPILOT_CLI_PATH` optional) |
| `llm_cache.py` | response cache | |
| `message_media.py` | `append_to_messages`, `append_images`, `append_videos`, `encode_media` | OpenAI-style multimodal messages |
| `model_registry.py` | `get_model_info(model)` | |
| `usage.py` | `record`, `totals`, `reset`, `set_log_path` | token/cost/latency tracking (OpenRouter + Bedrock record automatically); JSONL log via `LLM_PROVIDERS_USAGE_LOG` |

Provider selection by model-name prefix (`or-`, `aws-`, `azure-`, `gemini-`, `copilot-`) lives in the consumer repo
(e.g. `models.py` in the trajectory repo), not here.

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
from llm_providers.openrouter import call_openrouter
from llm_providers import usage

text = call_openrouter([{"role": "user", "content": "hi"}], "google/gemini-3.7-flash",
                       response_format={"type": "json_object"})
print(usage.totals())
```

`examples/aws_bedrock_post.py` is a standalone Bedrock example (model/guardrail from env vars).

## Tests (offline, no API calls)

```bat
python -m unittest discover -s tests -v
```

"""Provider-agnostic sync call: pick provider by model-name prefix (or-, aws-, azure-, gemini-, copilot-)."""

PROVIDER_PREFIXES = ("or-", "aws-", "azure-", "gemini-", "copilot-")


def call_llm(messages, model, max_tokens=4096, temperature=0, reasoning_effort=None):
    """Dispatch to the provider selected by `model` prefix; returns the assistant response text.
    e.g. 'or-meta-llama/llama-3.3-70b-instruct', 'aws-<bedrock id>', 'azure-<deployment>', 'gemini-2.5-flash'."""
    if model.startswith("or-"):
        from llm_providers.openrouter import call_openrouter
        return call_openrouter(messages, model=model[len("or-"):], max_tokens=max_tokens, temperature=temperature,
                               reasoning_effort=reasoning_effort)
    if model.startswith("aws-"):
        from llm_providers.aws_bedrock import call_llm as call_bedrock
        return call_bedrock(messages, bedrock_model_id=model[len("aws-"):], max_tokens=max_tokens,
                            temperature=temperature, reasoning_effort=reasoning_effort)
    if model.startswith("azure-"):
        from llm_providers.azure_openai import call_llm as call_azure
        return call_azure(messages, azure_deployment_model=model[len("azure-"):], max_tokens=max_tokens,
                          temperature=temperature, reasoning_effort=reasoning_effort)
    if model.startswith("gemini-"):
        from llm_providers.gemini import call_gemini
        return call_gemini(messages, model=model, max_tokens=max_tokens, temperature=temperature,
                           reasoning_effort=reasoning_effort)
    if model.startswith("copilot-"):
        from llm_providers.copilot_cli import call_copilot
        return call_copilot(messages, model=model[len("copilot-"):], max_tokens=max_tokens,
                            reasoning_effort=reasoning_effort)
    raise ValueError(f"Unknown provider prefix in model {model!r}; expected one of {PROVIDER_PREFIXES}")

"""Provider-agnostic sync call: pick provider by model-name prefix (or-, aws-, azure-, gemini-, copilot-)."""

PROVIDER_PREFIXES = ("or-", "aws-", "azure-", "gemini-", "copilot-")


def call_llm(messages, model, max_tokens=4096, temperature=0, reasoning_effort=None,
             conversation_key=None, client=None, response_format=None):
    """Dispatch to the provider selected by `model` prefix; returns the assistant response text.
    e.g. 'or-meta-llama/llama-3.3-70b-instruct', 'aws-<bedrock id>', 'azure-<deployment>', 'gemini-2.5-flash'.
    An unprefixed model uses the optional OpenAI-compatible client.
    response_format is supported only by OpenRouter."""
    if response_format is not None and not model.startswith("or-"):
        raise ValueError("response_format is supported only for OpenRouter models")
    if model.startswith("or-"):
        from llm_providers.openrouter import call_openrouter
        kwargs = {}
        if response_format is not None:
            kwargs["response_format"] = response_format
        return call_openrouter(messages, model=model[len("or-"):], max_tokens=max_tokens, temperature=temperature,
                               reasoning_effort=reasoning_effort, **kwargs)
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
        kwargs = {}
        if conversation_key is not None:
            kwargs["conversation_key"] = conversation_key
        return call_copilot(messages, model=model[len("copilot-"):], max_tokens=max_tokens,
                            reasoning_effort=reasoning_effort, **kwargs)
    if client is not None:
        completion = client.chat.completions.create(
            model=model, temperature=temperature, messages=messages, stream=True
        )
        return "".join(
            chunk.choices[0].delta.content
            for chunk in completion
            if chunk.choices[0].delta.content is not None
        )
    raise ValueError(f"Unknown provider prefix in model {model!r}; expected one of {PROVIDER_PREFIXES}")


def call_llm_provider(client, model, messages, max_tokens, reasoning_effort, conversation_key=None):
    """Compatibility entry point for callers using the original client-first signature."""
    return call_llm(messages, model, max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                    conversation_key=conversation_key, client=client)

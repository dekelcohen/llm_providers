"""Concurrent batch of LLM calls over any provider (sync providers run in threads)."""
import asyncio

from llm_providers.dispatch import call_llm


async def call_llm_many_async(prompts, contexts=None, model_name="or-meta-llama/llama-3.3-70b-instruct",
                              max_new_tokens=4096, temperature=0, max_concurrency=8):
    """Call call_llm for each prompts[i] + '\\n' + contexts[i] (one user message) concurrently.
    Returns list of dicts (prompt order): {'answer': response text, 'model': model_name, 'prompt', 'context'}."""
    contexts = contexts if contexts is not None else [""] * len(prompts)
    if len(contexts) != len(prompts):
        raise ValueError(f"len(contexts)={len(contexts)} != len(prompts)={len(prompts)}")
    sem = asyncio.Semaphore(max_concurrency)

    async def one(prompt, context):
        messages = [{"role": "user", "content": f"{prompt}\n{context}"}]
        async with sem:
            answer = await asyncio.to_thread(call_llm, messages, model_name, max_new_tokens, temperature)
        return {"answer": answer, "model": model_name, "prompt": prompt, "context": context}

    return await asyncio.gather(*(one(p, c) for p, c in zip(prompts, contexts)))


def wrap_sync_call_llm_many_async(prompts, contexts=None, model_name="or-meta-llama/llama-3.3-70b-instruct",
                                  max_new_tokens=4096, base_url=None, apiKey=None, **kwargs):
    """Sync wrapper over call_llm_many_async. model_name has a provider prefix (see dispatch.call_llm).
    base_url / apiKey are accepted for signature compatibility and ignored (providers read keys from env).
    Returns list[dict] with 'answer', 'model', 'prompt', 'context'. Not callable from a running event loop (asyncio.run)."""
    return asyncio.run(call_llm_many_async(prompts, contexts, model_name=model_name,
                                           max_new_tokens=max_new_tokens, **kwargs))

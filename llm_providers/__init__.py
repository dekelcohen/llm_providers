"""Shared LLM/VLM provider clients (OpenRouter, AWS Bedrock, Azure OpenAI, Gemini, Copilot CLI).

All providers take OpenAI-style `messages` (text + `image_url` data-URL parts, see `message_media`).
Submodules are imported by the caller, so optional deps (boto3 for Bedrock) are only needed when used.
Keys come from the environment or a local `.env` (never committed).
"""

__version__ = "0.1.0"
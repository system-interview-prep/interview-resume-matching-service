"""LLM Service integration for question generation, evaluation, follow-up, and feedback.

Supports OpenAI, AWS Bedrock, and Mock providers.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


class LLMService:
    """Wrapper service for executing prompts on LLM APIs."""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        self.model = os.getenv("LLM_MODEL", "").strip()
        self._client = None

        if self.provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for LLM_PROVIDER=openai.")
            self._client = OpenAI(api_key=api_key)
            if not self.model:
                self.model = "gpt-4o"

        elif self.provider in {"bedrock", "aws_bedrock", "aws"}:
            import boto3
            region = os.getenv("AWS_REGION", "us-east-1")
            self._client = boto3.client("bedrock-runtime", region_name=region)
            if not self.model:
                self.model = "anthropic.claude-3-sonnet-20240229-v1:0"
        else:
            raise ValueError(
                f"Invalid or missing LLM_PROVIDER: '{self.provider}'. "
                "Must configure LLM_PROVIDER=openai|bedrock in environment variables."
            )

    def generate_text(self, prompt: str) -> str:
        """Execute prompt against target LLM provider and return response text.

        Args:
            prompt: Structured prompt string.

        Returns:
            The raw text response from the model.
        """
        if self.provider == "openai" and self._client:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return resp.choices[0].message.content or ""

        elif self.provider in {"bedrock", "aws_bedrock", "aws"} and self._client:
            # Standard invoke model body payload for Claude 3 / standard models
            if "anthropic.claude" in self.model:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                })
                resp = self._client.invoke_model(modelId=self.model, body=body)
                payload = json.loads(resp["body"].read())
                return payload["content"][0]["text"]
            elif "nova" in self.model.lower():
                body = json.dumps({
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    "inferenceConfig": {
                        "temperature": 0.7
                    }
                })
                resp = self._client.invoke_model(modelId=self.model, body=body)
                payload = json.loads(resp["body"].read())
                content = payload.get("output", {}).get("message", {}).get("content", [])
                if content and isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                    return content[0]["text"]
                return str(payload)
            else:
                # Generic model invoke fallback
                body = json.dumps({"prompt": prompt, "max_tokens": 1024})
                resp = self._client.invoke_model(modelId=self.model, body=body)
                payload = json.loads(resp["body"].read())
                return payload.get("completion") or payload.get("text") or str(payload)

        raise ValueError(f"LLM provider '{self.provider}' is not configured or client is unavailable.")


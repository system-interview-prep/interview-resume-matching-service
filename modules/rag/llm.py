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
    """Wrapper service for executing prompts on LLM APIs or local mock fallback."""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
        self.model = os.getenv("LLM_MODEL", "").strip()
        self._client = None

        if self.provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set. Falling back to Mock LLM provider.")
                self.provider = "mock"
            else:
                self._client = OpenAI(api_key=api_key)
                if not self.model:
                    self.model = "gpt-4o"

        elif self.provider in {"bedrock", "aws_bedrock", "aws"}:
            import boto3
            region = os.getenv("AWS_REGION", "us-east-1")
            self._client = boto3.client("bedrock-runtime", region_name=region)
            if not self.model:
                self.model = "anthropic.claude-3-sonnet-20240229-v1:0"

        if self.provider == "mock":
            logger.info("Initializing Mock LLM Service (no external API calls)")

    def generate_text(self, prompt: str) -> str:
        """Execute prompt against target LLM provider and return response text.

        Args:
            prompt: Structured prompt string.

        Returns:
            The raw text response from the model.
        """
        if self.provider == "openai" and self._client:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.error("OpenAI LLM generation failed: %s. Falling back to Mock.", e)

        elif self.provider in {"bedrock", "aws_bedrock", "aws"} and self._client:
            try:
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
                else:
                    # Generic model invoke fallback
                    body = json.dumps({"prompt": prompt, "max_tokens": 1024})
                    resp = self._client.invoke_model(modelId=self.model, body=body)
                    payload = json.loads(resp["body"].read())
                    return payload.get("completion") or payload.get("text") or str(payload)
            except Exception as e:
                logger.error("AWS Bedrock LLM generation failed: %s. Falling back to Mock.", e)

        # Mock Fallback logic
        prompt_lower = prompt.lower()

        # 1. Check if it's the Feedback prompt
        if "technical coach" in prompt_lower:
            return """### Strengths
- Clear and accurate explanation of the CPython GIL.
- Correctly identified multiprocessing as the way to utilize multiple CPU cores in Python.

### Areas for Improvement
- Deepen understanding of memory allocation: explain why processes do not share memory by default unlike threads.

### Actionable Next Steps (Deliverables)
- Implement a task queue manager using the `multiprocessing` library to split computing tasks across a process pool.
"""

        # 2. Check if it's the Evaluation prompt (JSON schema expected)
        elif "technical assessor" in prompt_lower:
            return json.dumps({
                "must_have_covered": [
                    "Explained the Global Interpreter Lock (GIL) limitations",
                    "Suggested multiprocessing as a solution for CPU-bound tasks"
                ],
                "must_have_missed": [
                    "Did not clarify the difference between memory sharing in threads vs processes"
                ],
                "mistakes_identified": [
                    "Confused threading with parallel CPU execution in standard CPython"
                ],
                "evaluation_score": 0.82,
                "justification": "The candidate has a solid high-level understanding of GIL and CPU-bound constraints but lacks precision regarding memory sharing details."
            }, indent=2)

        # 3. Check if it's the Follow-up question generation prompt
        elif "conversational session" in prompt_lower:
            return "That makes sense. As a follow-up: How do you pass data or share state safely between multiple processes in Python if they do not share memory space? Have you used Queue or Value/Array?"

        # 4. Question generation prompt (Default mock question)
        else:
            return "Could you explain what the CPython Global Interpreter Lock (GIL) is, how it affects threading in CPU-bound vs I/O-bound programs, and what approaches you would take to achieve true parallelism?"

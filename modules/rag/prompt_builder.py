"""Prompt Builder for LLM interview preparation activities.

Constructs structured prompts for question generation, evaluation, follow-ups, and feedback.
Enforces the strict rule of not copying the reference Knowledge Base verbatim.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class PromptBuilder:
    """Helper class to construct prompts for the RAG LLM pipeline.

    Ensures reference documents are integrated cleanly without being copied verbatim.
    """

    def __init__(self):
        pass

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Helper to format retrieval chunks into a readable context block."""
        if not chunks:
            return "No reference documents available."

        formatted_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            chunk_type = chunk.get("metadata", {}).get("chunk_type", "general")
            text = chunk.get("text") or ""
            formatted_blocks.append(
                f"[Document {idx} | Type: {chunk_type}]\n{text.strip()}"
            )
        return "\n\n".join(formatted_blocks)

    def build_question_generation_prompt(
        self,
        context_chunks: List[Dict[str, Any]],
        *,
        topic: str,
        difficulty: str,
        history: Optional[List[str]] = None,
    ) -> str:
        """Construct the prompt to generate a new interview question.

        Args:
            context_chunks: Clean retrieval chunks containing knowledge and expected points.
            topic: Target interview topic.
            difficulty: Target difficulty level.
            history: List of questions already asked in this session.

        Returns:
            The formatted prompt string for LLM question generation.
        """
        context_str = self._format_context(context_chunks)
        history_list = history or []
        history_str = (
            "\n".join(f"- {q}" for q in history_list)
            if history_list
            else "None (this is the first question)."
        )

        return f"""You are an expert technical interviewer. Your goal is to generate a natural, situational, and technical interview question.

[Topic]
{topic}

[Difficulty Level]
{difficulty}

[Previously Asked Questions]
{history_str}

[Reference Knowledge Base Context]
{context_str}

[Instructions]
1. Formulate ONE clear, technical interview question suitable for the target difficulty.
2. The question must test the concepts detailed in the Reference Knowledge Base.
3. You may frame the question as a scenario, a problem-solving task, or a deep-dive conceptual question.
4. CRITICAL RULE: Do NOT copy the Reference Knowledge Base text verbatim. You must synthesize the technical details and express the question in your own natural, conversational words. Verbatim copying of phrases from the context is strictly prohibited.
5. Provide the output in a clean, interview-ready format. Do not include meta-commentary or introductory pleasantries.
"""

    def build_evaluation_prompt(
        self,
        *,
        context_chunks: List[Dict[str, Any]],
        question: str,
        candidate_answer: str,
    ) -> str:
        """Construct the prompt to evaluate the candidate's answer.

        Args:
            context_chunks: Clean retrieval chunks containing expected points and common mistakes.
            question: The question asked.
            candidate_answer: The candidate's response.

        Returns:
            The formatted prompt string for LLM evaluation.
        """
        context_str = self._format_context(context_chunks)

        return f"""You are a senior technical assessor. Your task is to evaluate the candidate's response to the interview question based on expected points and common mistakes.

[Interview Question]
{question}

[Candidate's Answer]
{candidate_answer}

[Reference Rubric & Expected Criteria Context]
{context_str}

[Instructions]
Evaluate the candidate's answer and produce a detailed assessment. 
You must analyze:
1. **Must-Have Coverage**: Identify which expected/critical points from the reference criteria were covered, and which were missed.
2. **Common Mistakes**: Check if the candidate fell into any of the common mistakes mentioned in the reference criteria.
3. **Pillar Score**: Assign a quantitative score from 0.0 (completely incorrect/blank) to 1.0 (flawless response) based on completeness.

CRITICAL RULE:
- Do NOT copy the rubric or expected points verbatim in your assessment report.
- Analyze the candidate's response in your own words. When discussing missing points, describe the concepts rather than repeating the reference phrases word-for-word.

Format the output as a valid JSON object matching this schema:
{{
  "must_have_covered": ["list of covered points, summarized in your own words"],
  "must_have_missed": ["list of missed points, summarized in your own words"],
  "mistakes_identified": ["list of mistakes made, if any"],
  "evaluation_score": 0.85,
  "justification": "Detailed explanation of the score in your own words"
}}
"""

    def build_follow_up_prompt(
        self,
        context_chunks: List[Dict[str, Any]],
        *,
        history: List[Dict[str, str]],
        last_question: str,
        candidate_answer: str,
    ) -> str:
        """Construct the prompt to generate a follow-up question.

        Args:
            context_chunks: Clean retrieval chunks of type 'follow_up'.
            history: List of conversation turns e.g. [{"role": "interviewer", "text": "..."}, {"role": "candidate", "text": "..."}]
            last_question: The last question asked.
            candidate_answer: The candidate's answer to the last question.

        Returns:
            The formatted prompt string for follow-up generation.
        """
        context_str = self._format_context(context_chunks)

        formatted_history = []
        for turn in history:
            role = turn.get("role", "user").capitalize()
            text = turn.get("text", "")
            formatted_history.append(f"{role}: {text}")
        history_str = "\n".join(formatted_history)

        return f"""You are a technical interviewer conducting a conversational session. You need to ask a follow-up question based on the candidate's last answer, the overall conversation history, and reference follow-up topics.

[Conversation History]
{history_str}
Interviewer (Last Question): {last_question}
Candidate (Last Answer): {candidate_answer}

[Reference Follow-Up Topics]
{context_str}

[Instructions]
1. Generate ONE natural, concise follow-up question.
2. The follow-up should bridge the candidate's last response with the concepts or questions suggested in the Reference Follow-Up Topics.
3. The follow-up can clarify a point, challenge a potential issue in the candidate's answer, or probe deeper into the next level of detail.
4. CRITICAL RULE: Do NOT copy the Reference Follow-Up Topics verbatim. Adapt the suggestion dynamically based on what the candidate actually said in their response. Avoid verbatim repetition.
5. Provide the output in a clean, direct format.
"""

    def build_feedback_prompt(
        self,
        *,
        evaluation_details: Dict[str, Any] | str,
        deliverables: List[Dict[str, Any]],
    ) -> str:
        """Construct the prompt to generate overall constructive feedback.

        Args:
            evaluation_details: The assessments of the candidate's answers.
            deliverables: Clean retrieval chunks of type 'deliverables'.

        Returns:
            The formatted prompt string for feedback generation.
        """
        deliverables_str = self._format_context(deliverables)
        
        eval_str = (
            json.dumps(evaluation_details, indent=2)
            if isinstance(evaluation_details, dict)
            else str(evaluation_details)
        )

        return f"""You are a technical coach. Your task is to provide constructive feedback to a candidate after their interview session.

[Evaluation Data]
{eval_str}

[Reference Deliverables & Learning Outcomes]
{deliverables_str}

[Instructions]
Create structured feedback including:
1. **Strengths**: Highlight what the candidate did well based on the evaluation data.
2. **Key Areas for Improvement**: Pinpoint specific technical concepts they missed or made mistakes on.
3. **Actionable Next Steps (Deliverables)**: Provide clear, tailored learning recommendations or tasks aligned with the Reference Deliverables to help the candidate improve.

CRITICAL RULE:
- Do NOT copy the Reference Deliverables or reference answer guides verbatim.
- Customize the recommendations so they are direct, practical, and written in your own words.

Provide the feedback in a encouraging, professional tone.
"""

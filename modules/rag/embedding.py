"""Embedding utilities for Interview Documents (env-based, provider-agnostic).

Responsibilities:
- Read Interview Document from file path / raw text / dict
- Normalize + chunk content
- Generate embedding vectors via adapter (OpenAI/AWS Bedrock/Mock)
- Return payloads ready for vector insertion
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join([_to_text(x) for x in value if _to_text(x)])
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            rendered = _to_text(v)
            if rendered:
                lines.append(f"{k}: {rendered}")
        return "\n".join(lines)
    return str(value)


class EmbeddingAdapter:
    """Base adapter interface for embedding providers."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError



class OpenAIEmbeddingAdapter(EmbeddingAdapter):
    """OpenAI embeddings adapter.
    Env:
    - OPENAI_API_KEY
    - EMBEDDING_MODEL (default: text-embedding-3-small)
    """

    def __init__(self, model_name: str):
        from openai import OpenAI  # pip install openai>=1.x
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = model_name

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            resp = self._client.embeddings.create(model=self._model, input=text)
            vectors.append(list(resp.data[0].embedding))
        return vectors


class BedrockEmbeddingAdapter(EmbeddingAdapter):
    """AWS Bedrock embeddings adapter.
    Env:
    - AWS_REGION
    - EMBEDDING_MODEL (e.g., cohere.embed-multilingual-v3)
    """

    def __init__(self, model_id: str):
        import boto3
        region = os.getenv("AWS_REGION", "us-east-1")
        self._model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        # Batch support for Cohere
        if "cohere" in self._model_id.lower():
            body = json.dumps({
                "texts": texts,
                "input_type": "search_document"
            })
            resp = self._client.invoke_model(modelId=self._model_id, body=body)
            payload = json.loads(resp["body"].read())
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list):
                return [[float(x) for x in emb] for emb in embeddings]
            raise ValueError("Cohere response does not contain 'embeddings' list")

        # Fallback loop for Titan or other models
        vectors: List[List[float]] = []
        for text in texts:
            body = json.dumps({"inputText": text})
            resp = self._client.invoke_model(modelId=self._model_id, body=body)
            payload = json.loads(resp["body"].read())
            emb = payload.get("embedding") or payload.get("embeddings")
            if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                vectors.append([float(x) for x in emb])
            elif isinstance(emb, list) and emb and isinstance(emb[0], list):
                vectors.append([float(x) for x in emb[0]])
            else:
                raise ValueError("Invalid Bedrock embedding response format")
        return vectors


def build_embedding_adapter_from_env() -> EmbeddingAdapter:
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()

    if provider == "openai":
        return OpenAIEmbeddingAdapter(model_name=model)
    if provider in {"bedrock", "aws_bedrock", "aws"}:
        return BedrockEmbeddingAdapter(model_id=model)

    raise ValueError(
        f"Invalid or missing EMBEDDING_PROVIDER: '{provider}'. "
        "Must configure EMBEDDING_PROVIDER=openai|bedrock in environment variables."
    )


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


def _load_document_from_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    import yaml
    obj = yaml.safe_load(raw)
    if isinstance(obj, dict):
        return obj
    raise ValueError("Unsupported document format. Provide valid JSON/YAML content.")


def _build_chunks_from_interview_document(doc: Dict[str, Any]) -> List[ChunkRecord]:
    document_id = _safe_get(doc, ["document", "document_id"], "unknown_document")
    version = _safe_get(doc, ["document", "version"], "1.0.0")

    topic_name = _safe_get(doc, ["topic", "topic_name"], "unknown-topic")
    domain = _safe_get(doc, ["topic", "domain"], "general")
    difficulty = _safe_get(doc, ["difficulty", "level"], "intermediate")
    roles = _ensure_list(_safe_get(doc, ["topic", "role_targets"], []))
    job_levels = _ensure_list(_safe_get(doc, ["topic", "job_levels"], []))
    language = _safe_get(doc, ["document", "language"], "vi")

    common_meta = {
        "knowledge_unit_id": document_id,
        "version": version,
        "topic": topic_name,
        "domain": domain,
        "difficulty": difficulty,
        "roles": roles,
        "job_levels": job_levels,
        "language": language,
        "is_active": _safe_get(doc, ["metadata", "retrieval", "is_active"], True),
        "quality_score": _safe_get(doc, ["metadata", "retrieval", "quality_score"], 0.8),
        "updated_at": _safe_get(doc, ["document", "updated_at"], _utc_now_iso()),
    }

    chunks: List[ChunkRecord] = []
    chunk_specs = [
        ("knowledge", _safe_get(doc, ["knowledge"], {})),
        ("expected_points", _safe_get(doc, ["expected_points"], {})),
        ("common_mistakes", _safe_get(doc, ["common_mistakes"], {})),
        ("follow_up", _safe_get(doc, ["follow_up"], {})),
        ("reference_source", _safe_get(doc, ["reference_source"], {})),
        ("deliverables", _safe_get(doc, ["deliverables"], {})),
    ]

    idx = 1
    for chunk_type, payload in chunk_specs:
        text = _to_text(payload)
        if not text:
            continue
        chunk_id = f"{document_id}_chunk_{idx:02d}"
        meta = dict(common_meta)
        meta.update(
            {
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "source": "interview_document",
            }
        )
        chunks.append(ChunkRecord(chunk_id=chunk_id, text=text, metadata=meta))
        idx += 1

    if not chunks:
        raise ValueError("No chunkable content found in Interview Document.")
    return chunks


def read_interview_document(
    *,
    file_path: Optional[str] = None,
    raw_text: Optional[str] = None,
    document_obj: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    provided = [x is not None for x in (file_path, raw_text, document_obj)]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of: file_path, raw_text, document_obj")

    if file_path:
        return _load_document_from_file(file_path)

    if raw_text is not None:
        try:
            obj = json.loads(raw_text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        import yaml
        obj = yaml.safe_load(raw_text)
        if isinstance(obj, dict):
            return obj
        raise ValueError("raw_text is not valid JSON/YAML object")

    return document_obj or {}


def generate_embedding_payloads(
    *,
    file_path: Optional[str] = None,
    raw_text: Optional[str] = None,
    document_obj: Optional[Dict[str, Any]] = None,
    adapter: Optional[EmbeddingAdapter] = None,
) -> List[Dict[str, Any]]:
    doc = read_interview_document(file_path=file_path, raw_text=raw_text, document_obj=document_obj)
    chunks = _build_chunks_from_interview_document(doc)

    embedder = adapter or build_embedding_adapter_from_env()
    vectors = embedder.embed_texts([c.text for c in chunks])

    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding output size mismatch")

    records: List[Dict[str, Any]] = []
    for chunk, vec in zip(chunks, vectors):
        records.append(
            {
                "id": chunk.chunk_id,
                "vector": [float(x) for x in vec],
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
        )
    return records

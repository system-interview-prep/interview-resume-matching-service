"""Insert/update/search Interview Document embeddings into vector store (PyVector adapter style).

This module provides:
- insert_vectors(records)
- upsert_vectors(records)
- upsert_interview_document(...)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from modules.rag.embedding import generate_embedding_payloads

logger = logging.getLogger(__name__)


class VectorStoreAdapter:
    """Safe env-based adapter for vector insertion/upsert and search."""

    def __init__(self):
        self.provider = os.getenv("VECTOR_DB_PROVIDER", "mock").strip().lower()
        self.collection = os.getenv("PYVECTOR_COLLECTION", "interview_knowledge_vectors")
        self.mock_db_file = os.getenv("MOCK_VECTOR_STORE_FILE", "mock_vector_store.json")

        self._client = None
        if self.provider == "pyvector":
            # NOTE: adjust import according to your internal SDK if needed.
            # Example expected env:
            # - PYVECTOR_URL
            # - PYVECTOR_API_KEY
            try:
                from pyvector import Client  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "VECTOR_DB_PROVIDER=pyvector but pyvector package/client is unavailable."
                ) from exc

            self._client = Client(
                base_url=os.getenv("PYVECTOR_URL"),
                api_key=os.getenv("PYVECTOR_API_KEY"),
            )

    def _load_mock_db(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.mock_db_file):
            return {}
        try:
            with open(self.mock_db_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_mock_db(self, db: Dict[str, Dict[str, Any]]):
        try:
            with open(self.mock_db_file, "w", encoding="utf-8") as f:
                json.dump(db, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to write to mock vector store: %s", e)

    def insert(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.provider == "mock":
            logger.info("[MOCK] insert %d vectors into %s", len(records), self.collection)
            db = self._load_mock_db()
            for r in records:
                db[r["id"]] = r
            self._save_mock_db(db)
            return {"success": True, "inserted": len(records), "provider": "mock"}

        # PyVector expected API style (adapt if SDK differs)
        self._client.insert(
            collection=self.collection,
            records=records,
        )
        return {"success": True, "inserted": len(records), "provider": "pyvector"}

    def upsert(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.provider == "mock":
            logger.info("[MOCK] upsert %d vectors into %s", len(records), self.collection)
            db = self._load_mock_db()
            for r in records:
                db[r["id"]] = r
            self._save_mock_db(db)
            return {"success": True, "upserted": len(records), "provider": "mock"}

        # PyVector expected API style (adapt if SDK differs)
        self._client.upsert(
            collection=self.collection,
            records=records,
        )
        return {"success": True, "upserted": len(records), "provider": "pyvector"}

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search similar vectors with metadata filtering.

        Returns a list of dicts: [{"id": ..., "vector": ..., "text": ..., "metadata": ..., "score": ...}]
        """
        if self.provider == "mock":
            logger.info("[MOCK] search top %d in %s", limit, self.collection)
            db = self._load_mock_db()
            candidates = list(db.values())

            # Apply filters
            if filters:
                filtered = []
                for cand in candidates:
                    meta = cand.get("metadata", {})
                    match = True
                    for k, v in filters.items():
                        if k not in meta:
                            match = False
                            break
                        # Handle list vs scalar match for attributes like roles or job_levels
                        meta_val = meta[k]
                        if isinstance(meta_val, list):
                            if v not in meta_val:
                                match = False
                                break
                        else:
                            if meta_val != v:
                                match = False
                                break
                    if match:
                        filtered.append(cand)
                candidates = filtered

            # Calculate cosine similarities
            results = []
            for cand in candidates:
                cand_vec = cand.get("vector")
                if not cand_vec or len(cand_vec) != len(query_vector):
                    continue
                # Cosine similarity
                dot_prod = sum(x * y for x, y in zip(cand_vec, query_vector))
                norm_cand = sum(x * x for x in cand_vec) ** 0.5
                norm_q = sum(x * x for x in query_vector) ** 0.5
                score = dot_prod / (norm_cand * norm_q) if (norm_cand > 0 and norm_q > 0) else 0.0
                
                results.append({
                    "id": cand["id"],
                    "vector": cand["vector"],
                    "text": cand["text"],
                    "metadata": cand["metadata"],
                    "score": float(score)
                })

            # Sort by similarity score descending
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]

        # PyVector API search
        try:
            return self._client.search(
                collection=self.collection,
                vector=query_vector,
                filter=filters,
                limit=limit,
            )
        except AttributeError:
            # Fallback if SDK style differs (e.g. query)
            return self._client.query(
                collection=self.collection,
                vector=query_vector,
                filter=filters,
                limit=limit,
            )


def insert_vectors(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    store = VectorStoreAdapter()
    return store.insert(records)


def upsert_vectors(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    store = VectorStoreAdapter()
    return store.upsert(records)


def upsert_interview_document(
    *,
    file_path: Optional[str] = None,
    raw_text: Optional[str] = None,
    document_obj: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read doc -> embed -> upsert vectors."""
    records = generate_embedding_payloads(
        file_path=file_path,
        raw_text=raw_text,
        document_obj=document_obj,
    )
    result = upsert_vectors(records)
    result["records"] = len(records)
    return result

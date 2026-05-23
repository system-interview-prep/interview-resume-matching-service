"""PostgreSQL + pgvector database module for managing JD vector embeddings."""

import logging
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)


class VectorDB:
    """Manages PostgreSQL connection and pgvector operations for JD embeddings."""

    def __init__(self, database_url: str, vector_dimension: int = 384):
        self.database_url = database_url
        self.vector_dimension = vector_dimension
        self._conn = None

    def _get_connection(self):
        """Get or create a database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.database_url)
            # Must create extension before registering vector type!
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self._conn.commit()
            register_vector(self._conn)
        return self._conn

    def init_tables(self):
        """Create the job_profiles_vector table."""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS job_profiles_vector (
                    job_id VARCHAR(255) PRIMARY KEY,
                    job_vector vector({self.vector_dimension}),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.commit()
            logger.info("job_profiles_vector table initialized successfully")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize vector table: {e}")
            raise
        finally:
            cur.close()

    def upsert_jd_vector(self, job_id: str, vector: list):
        """Insert or update a JD vector embedding."""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO job_profiles_vector (job_id, job_vector)
                VALUES (%s, %s)
                ON CONFLICT (job_id)
                DO UPDATE SET job_vector = EXCLUDED.job_vector, created_at = NOW();
            """, (job_id, vector))
            conn.commit()
            logger.info(f"Upserted vector for job_id={job_id}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to upsert vector for job_id={job_id}: {e}")
            raise
        finally:
            cur.close()

    def get_jd_vector(self, job_id: str):
        """Retrieve a JD vector embedding by job_id."""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT job_vector FROM job_profiles_vector WHERE job_id = %s;",
                (job_id,)
            )
            result = cur.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to get vector for job_id={job_id}: {e}")
            raise
        finally:
            cur.close()

    def cosine_similarity_score(self, job_id: str, cv_vector: list) -> float | None:
        """Calculate cosine similarity between a CV vector and a stored JD vector.
        
        Uses pgvector's <=> (cosine distance) operator.
        Returns similarity score (1 - distance), or None if job_id not found.
        """
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT 1 - (job_vector <=> %s::vector) AS similarity_score
                FROM job_profiles_vector
                WHERE job_id = %s;
            """, (cv_vector, job_id))
            result = cur.fetchone()
            return float(result[0]) if result else None
        except Exception as e:
            logger.error(f"Failed to compute similarity for job_id={job_id}: {e}")
            raise
        finally:
            cur.close()

    def delete_jd_vector(self, job_id: str):
        """Delete a JD vector embedding."""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "DELETE FROM job_profiles_vector WHERE job_id = %s;",
                (job_id,)
            )
            conn.commit()
            logger.info(f"Deleted vector for job_id={job_id}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete vector for job_id={job_id}: {e}")
            raise
        finally:
            cur.close()

    def close(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("Database connection closed")

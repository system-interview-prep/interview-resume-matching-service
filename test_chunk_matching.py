import requests
import json
import time

job_id = "test_job_scanjob_123"
job_text = "We are looking for a Senior React Developer with Python experience who knows Kubernetes and Docker."

# Step 1: Pre-vectorize the JD to populate database cache
vectorize_url = "http://127.0.0.1:5001/api/v1/jd/vectorize"
v_payload = {
    "job_id": job_id,
    "job_text": job_text
}

print("Step 1: Pre-vectorizing JD...")
v_resp = requests.post(vectorize_url, json=v_payload)
print(f"Vectorize Status: {v_resp.status_code}")
print(v_resp.json())

# Wait for Celery worker to finish embedding and cache save
print("Waiting 5 seconds for Celery task to complete...")
time.sleep(5)

# Step 2: Query matching service process-resumes endpoint
match_url = "http://127.0.0.1:5001/api/process-resumes"
m_payload = {
    "job_id": job_id,
    "resumes": [
        "John Doe is a frontend specialist. He is a Senior React Developer with extensive experience in React, JavaScript, and Tailwind. In his spare time, he wrote Python automation scripts."
    ],
    "methods": ["sbert", "cross_encoder"],
    "options": {
        "chunk_level": True
    }
}

print("\nStep 2: Sending matching request to FastAPI...")
try:
    response = requests.post(match_url, json=m_payload, timeout=30)
    print(f"Match Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Failed to query endpoint: {e}")


for QDRANT 
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
RUNNING CODE 
uvicorn main:app --reload --host 0.0.0.0 --port 8000
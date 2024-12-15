from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from patent_retrieval.engine import PatentRetrievalService
import uvicorn

app = FastAPI(title="Patent Retrieval API")


service = PatentRetrievalService(
        dataset_path="/app/src/patent_retrieval/data.txt",
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

class SearchRequest(BaseModel):
    keywords: List[str]
    precision_recall_balance: Optional[float] = 0.5

@app.post("/search")
def search_patents(request: SearchRequest):
    try:
        results, metadata = service.retrieve_patents(
            keywords=request.keywords,
            precision_recall_balance=request.precision_recall_balance
        )
        return {
            "results": results,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# uvicorn main:app --reload --host 0.0.0.0 --port 8000


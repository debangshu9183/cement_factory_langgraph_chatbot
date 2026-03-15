from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import app as graph_app

api = FastAPI(title="LangGraph Chatbot API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    confidence: float

@api.post("/chat", response_model=AnswerResponse)
async def chat(request: QuestionRequest):
    result = await graph_app.ainvoke({"question": request.question})
    return AnswerResponse(
        answer=result["answer"],
        confidence=result.get("confidence", 1.0)
    )

@api.get("/health")
def health():
    return {"status": "ok"}

# Run with: uvicorn api:api --host 0.0.0.0 --port 8000 --reload

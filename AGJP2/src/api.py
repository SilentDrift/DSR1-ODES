import sys
import pathlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to path for imports
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.chatbot import DeepSeekODETutor

app = FastAPI(title="DeepSeek‑ODE‑Tutor API", description="AI-powered differential equations tutor")

# Global engine instance
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    try:
        print("🔄 Initializing DeepSeek ODE Tutor...")
        engine = DeepSeekODETutor()
        print("✅ API ready!")
    except Exception as e:
        print(f"❌ Failed to initialize tutor: {e}")
        raise

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@app.get("/")
def read_root():
    return {"message": "DeepSeek ODE Tutor API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": engine is not None}

@app.post("/generate", response_model=Answer)
def generate(query: Query):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        answer = engine.generate(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}") 
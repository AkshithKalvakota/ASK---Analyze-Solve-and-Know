from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.projects import router as projects_router
from app.api.datasets import router as datasets_router

app = FastAPI(title="ASK - Analyze, Solve and Know")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://ask-analyze-solve-and-know.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.models import router as models_router

app.include_router(models_router)
app.include_router(projects_router)
app.include_router(datasets_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .schemas import ApplicantData, CreditScoreResponse
from .predictor import load_model, score_applicant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run at startup — preload model so first request is fast"""
    logger.info("🚀 Starting Softint AI Credit Scoring API...")
    try:
        load_model()
    except FileNotFoundError as e:
        logger.warning(f"⚠️  {e}")
        logger.warning("⚠️  Train the model first: python scripts/train_credit_model.py")
    yield
    logger.info("👋 API shutting down...")

# Initialize API
app = FastAPI(
    title="Softint AI — Credit Scoring API",
    description="AI-powered credit scoring using alternative data",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Endpoints ──

@app.get("/")
async def root():
    """API Health Check"""
    return {
        "service": "Softint AI Credit Scoring API",
        "version": "1.0.0",
        "status": "✅ Online",
        "endpoints": {
            "POST /api/v1/score": "Calculate credit score",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/api/v1/score", response_model=CreditScoreResponse)
async def calculate_credit_score(applicant: ApplicantData):
    """
    Calculate Credit Score
    
    Submit applicant data → Receive credit score, risk grade, and key factors.
    Score Range: **300 – 850**
    Grades: **A** (Excellent) → **E** (High Risk)
    """
    try:
        logger.info(f"📊 Scoring applicant: {applicant.occupation}, £{applicant.annual_salary_gbp:,}/yr")
        
        result = score_applicant(applicant.model_dump())
        
        logger.info(f"✅ Score: {result['credit_score_300_850']} | Grade: {result['score_grade']} | Risk: {result['default_risk_percent']}%")
        return result

    except Exception as e:
        logger.error(f"❌ Scoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
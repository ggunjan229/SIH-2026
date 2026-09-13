from __future__ import annotations
 
from pathlib import Path
from typing import List, Optional
 
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
 
import my_ml_core
 
MODEL_PATH = Path(__file__).parent / "model.joblib"
 
app = FastAPI(title="Smart Worker Allocation - Ranking Service")
 
# Allow the demo UI (index.html, opened as a local file or via a simple
# static server) to call this API directly from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_home():
    return FileResponse(Path(__file__).parent / "index.html")

# 2. Mount your directory so any CSS, JS, or images load automatically
app.mount("/static", StaticFiles(directory=Path(__file__).parent), name="static")

_artifact: Optional[dict] = None
 
 
@app.on_event("startup")
def load_model() -> None:
    global _artifact
    if not MODEL_PATH.exists():
        raise RuntimeError(f"'{MODEL_PATH}' not found - run `python train.py` first.")
    _artifact = joblib.load(MODEL_PATH)
 
 
class Candidate(BaseModel):
    worker_id: str
    experience_years: float = Field(ge=0)
    certifications_count: int = Field(ge=0)
    has_certifications: int = Field(ge=0, le=1)
    distance_km: float = Field(ge=0)
    worker_skill_price: float = Field(ge=0)
    worker_skill_price_type: str
    platform_tenure_days: int = Field(ge=0)
 
 
class RankRequest(BaseModel):
    booking_id: str
    quantity: int = Field(ge=1)
    scheduled_hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    candidates: List[Candidate]
 
 
@app.get("/health")
def health():
    return {
        "status": "ok" if _artifact else "model_not_loaded",
        "model_name": _artifact.get("model_name") if _artifact else None,
    }
 
 
@app.post("/rank-candidates")
def rank_candidates(request: RankRequest):
    if _artifact is None:
        raise HTTPException(503, "Model not loaded")
    if not request.candidates:
        raise HTTPException(400, "candidates list must not be empty")
 
    rows = [
        {
            my_ml_core.GROUP_COLUMN: request.booking_id,
            "worker_id": c.worker_id,
            "quantity": request.quantity,
            "scheduled_hour": request.scheduled_hour,
            "day_of_week": request.day_of_week,
            **c.model_dump(exclude={"worker_id"}),
        }
        for c in request.candidates
    ]
    df = pd.DataFrame(rows)
 
    try:
        my_ml_core.validate_schema(df, require_target=False)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
 
    df = my_ml_core.add_relative_features(df, group_col=my_ml_core.GROUP_COLUMN)
 
    pipeline = _artifact["pipeline"]
    scores = pipeline.predict_proba(df[_artifact["feature_columns"]])[:, 1]
    df = df.assign(score=scores).sort_values("score", ascending=False).reset_index(drop=True)
 
    return {
        "bookingContext": {"bookingId": request.booking_id, "quantity": request.quantity,
                            "scheduledHour": request.scheduled_hour, "dayOfWeek": request.day_of_week},
        "rankedWorkers": [
            {"worker_id": str(r.worker_id), "score": round(float(r.score), 4), "rank": i + 1}
            for i, r in df.iterrows()
        ],
        "modelVersion": str(_artifact.get("model_name", "unknown")),
    }
 
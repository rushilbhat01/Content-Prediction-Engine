"""
api.py — FastAPI backend for the Shorts Predictor.
Wraps predict.py so the beautiful HTML frontend can call it.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
import subprocess, sys, json, re, tempfile, os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

app = FastAPI(title="Shorts Predictor API")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("DEBUG validation error details:")
    print(exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(__file__).parent / "models"
BASE_DIR   = Path(__file__).parent

NICHES = {
    "fitness":   {"label": "Fitness & Gym",     "emoji": "💪", "color": "#818cf8"},
    "food":      {"label": "Food & Recipes",    "emoji": "🍳", "color": "#fb923c"},
    "finance":   {"label": "Finance & Money",   "emoji": "📈", "color": "#34d399"},
    "motivation":{"label": "Motivation",        "emoji": "🔥", "color": "#f87171"},
    "education": {"label": "Education",         "emoji": "📚", "color": "#60a5fa"},
    "beauty":    {"label": "Beauty & Makeup",   "emoji": "💄", "color": "#f472b6"},
}


@app.get("/api/niches")
def get_niches():
    """Return available niches + which have trained models."""
    result = []
    for key, cfg in NICHES.items():
        result.append({
            "id":    key,
            "label": cfg["label"],
            "emoji": cfg["emoji"],
            "color": cfg["color"],
            "ready": (MODELS_DIR / f"xgb_{key}.pkl").exists(),
        })
    return result


@app.post("/api/predict")
async def predict(
    video:      UploadFile = File(None),
    niche:      str        = Form("fitness"),
    title:      str        = Form(""),
    tags:       str        = Form(""),
    desc:       str        = Form(""),
    hour:       int        = Form(18),
    weekday:    int        = Form(1),
    channel:    str        = Form(""),
):
    print(f"DEBUG /api/predict: video={video}, niche={niche}, title={title}, tags={tags}, desc={desc}, hour={hour}, weekday={weekday}, channel={channel}")
    if video is not None:
        print(f"DEBUG /api/predict: video.filename={video.filename}, video.content_type={video.content_type}")
    else:
        print("DEBUG /api/predict: video is None!")
        raise HTTPException(400, "DEBUG: video parameter is missing from request body.")

    if not (MODELS_DIR / f"xgb_{niche}.pkl").exists():
        raise HTTPException(400, f"No trained model for niche '{niche}'")

    # Save uploaded video to temp file
    suffix   = Path(video.filename).suffix or ".mp4"
    tmp_vid  = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    contents = await video.read()
    tmp_vid.write(contents)
    tmp_vid.close()
    tmp_path = tmp_vid.name

    try:
        cmd = [
            sys.executable, str(BASE_DIR / "predict.py"),
            "--video",   tmp_path,
            "--title",   title,
            "--tags",    tags,
            "--desc",    desc,
            "--hour",    str(hour),
            "--weekday", str(weekday),
            "--niche",   niche,
        ]
        if channel.strip():
            cmd += ["--channel", channel.strip()]

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(BASE_DIR),
        )
        raw = result.stdout + result.stderr

        # Parse structured outputs
        pct_match         = re.search(r"Percentile:\s+(\d+)",   raw)
        hook_frame_match  = re.search(r"HOOK_FRAME:(.+)",       raw)
        hook_checks_match = re.search(r"HOOK_CHECKS:(.+)",      raw)
        recs_match        = re.search(r"RECOMMENDATIONS:(.+)",  raw)

        percentile      = int(pct_match.group(1))               if pct_match         else None
        hook_frame_path = hook_frame_match.group(1).strip()     if hook_frame_match  else None
        hook_checks     = json.loads(hook_checks_match.group(1)) if hook_checks_match else []
        recommendations = json.loads(recs_match.group(1))        if recs_match        else []

        # Read hook frame and convert to base64 for JSON response
        hook_frame_b64 = None
        if hook_frame_path and Path(hook_frame_path).exists():
            import base64
            with open(hook_frame_path, "rb") as f:
                hook_frame_b64 = base64.b64encode(f.read()).decode()
            try:
                os.unlink(hook_frame_path)
            except Exception:
                pass

        if percentile is None and result.returncode != 0:
            return JSONResponse(status_code=500, content={
                "error": "Prediction failed",
                "details": raw[-2000:],
            })

        return {
            "percentile":     percentile,
            "hook_frame":     hook_frame_b64,
            "hook_checks":    hook_checks,
            "recommendations": recommendations,
            "niche":          niche,
            "niche_label":    NICHES.get(niche, {}).get("label", niche),
            "niche_emoji":    NICHES.get(niche, {}).get("emoji", ""),
            "niche_color":    NICHES.get(niche, {}).get("color", "#8b5cf6"),
        }

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# Serve the frontend HTML at root
@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "index.html"))


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

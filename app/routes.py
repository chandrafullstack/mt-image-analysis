"""API routes for the mitochondria dashboard."""
import json
import pandas as pd
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pathlib import Path

from src.incoming_feedback import ingest_incoming_feedback

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Resolve outputs paths against project root so the API works regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "outputs" / "metrics" / "features_with_gratio.csv"
NEURON_PATH = PROJECT_ROOT / "outputs" / "metrics" / "neuron_gratios.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "per_image_report.json"
CROPS_DIR = PROJECT_ROOT / "outputs" / "crops"


@router.get("/")
async def dashboard(request: Request):
    """Render the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/api/gratio-data")
async def get_gratio_data():
    """
    Return mitochondria data as JSON for Plotly.
    Includes health, shape, fission/fusion, and myelin context.
    """
    # Pull in any newly dropped expert-labeled files before serving chart data.
    ingest_incoming_feedback(quiet=True)

    if not DATA_PATH.exists():
        return JSONResponse(content=[])

    df = pd.read_csv(DATA_PATH)
    records = []
    for _, row in df.iterrows():
        instance_id = int(row["label"])
        crop_filename = f"mito_{instance_id:04d}.png"
        crop_path = CROPS_DIR / crop_filename
        records.append({
            "instance_id": instance_id,
            "g_ratio": round(float(row.get("g_ratio", 0)), 4),
            "aspect_ratio": round(float(row.get("aspect_ratio", 0)), 2),
            "area_um2": round(float(row.get("area_um2", 0)), 4),
            "form_factor": round(float(row.get("form_factor", 0)), 3),
            "classification": row.get("label_final", "UNKNOWN"),
            "shape_category": row.get("shape_category", "OTHER"),
            "fission_fusion": row.get("fission_fusion_state", "NORMAL"),
            "myelin_context": row.get("myelin_context", "UNASSIGNED"),
            "crop_image": f"/crops/{crop_filename}" if crop_path.exists() else None,
        })
    return JSONResponse(content=records)


@router.get("/api/neuron-gratio")
async def get_neuron_gratio():
    """Return per-neuron G-ratio data (axon/fibre diameter ratio)."""
    if not NEURON_PATH.exists():
        return JSONResponse(content=[])
    df = pd.read_csv(NEURON_PATH)
    return JSONResponse(content=df.to_dict("records"))


@router.get("/api/summary")
async def get_summary():
    """Return aggregate statistics for the dashboard header."""
    # Keep summary in sync with incoming/healthy and incoming/unhealthy folders.
    ingest_incoming_feedback(quiet=True)

    if not DATA_PATH.exists():
        return JSONResponse(content={
            "total_mitochondria": 0,
            "healthy_count": 0,
            "unhealthy_count": 0,
            "mean_gratio": None,
            "std_gratio": None,
            "neuron_count": 0,
            "mean_neuron_gratio": None,
            "fission_count": 0,
            "fusion_count": 0,
        })

    df = pd.read_csv(DATA_PATH)
    total = len(df)
    healthy = int((df.get("label_final", pd.Series(dtype=str)) == "HEALTHY").sum())
    unhealthy = int((df.get("label_final", pd.Series(dtype=str)) == "UNHEALTHY").sum())

    # Fission/fusion counts
    fission = int((df.get("fission_fusion_state", pd.Series(dtype=str)) == "FISSION").sum())
    fusion = int((df.get("fission_fusion_state", pd.Series(dtype=str)) == "FUSION").sum())

    # Neuron data
    neuron_count = 0
    mean_neuron_gratio = None
    if NEURON_PATH.exists():
        ndf = pd.read_csv(NEURON_PATH)
        neuron_count = len(ndf)
        mean_neuron_gratio = round(float(ndf["g_ratio"].mean()), 4) if len(ndf) > 0 else None

    return JSONResponse(content={
        "total_mitochondria": total,
        "healthy_count": healthy,
        "unhealthy_count": unhealthy,
        "mean_gratio": round(float(df["g_ratio"].mean()), 4) if "g_ratio" in df else None,
        "std_gratio": round(float(df["g_ratio"].std()), 4) if "g_ratio" in df else None,
        "neuron_count": neuron_count,
        "mean_neuron_gratio": mean_neuron_gratio,
        "fission_count": fission,
        "fusion_count": fusion,
    })


@router.get("/api/report")
async def get_report():
    """Return the full per-image report (all 13 metrics)."""
    if not REPORT_PATH.exists():
        return JSONResponse(content=[])
    with open(REPORT_PATH) as f:
        return JSONResponse(content=json.load(f))

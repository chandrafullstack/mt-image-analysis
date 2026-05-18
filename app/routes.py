"""API routes for the mitochondria dashboard."""
import io
import json
import pandas as pd
from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, Response
from pathlib import Path
from PIL import Image, ImageDraw

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
    # Starlette >=0.29 requires `request` as the first positional argument.
    return templates.TemplateResponse(request, "index.html")


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
            "source_file": row.get("source_file", "unknown"),
            "resolution_group": row.get("resolution_group", "unknown"),
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


@router.get("/api/instance/{instance_id}/context")
async def get_instance_context(instance_id: int, max_size: int = 720):
    """
    Return the full source EM image with a red bbox drawn around the
    requested mitochondrion. Falls back to 404 if bbox metadata is
    missing (legacy CSVs) or the source image is no longer on disk.
    """
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="No metrics CSV.")

    df = pd.read_csv(DATA_PATH)
    required = {"source_path", "bbox_y1", "bbox_x1", "bbox_y2", "bbox_x2"}
    if not required.issubset(df.columns):
        raise HTTPException(
            status_code=404,
            detail="Bbox/source_path columns missing — re-run full_image_inference to enable full-image view.",
        )

    row = df.loc[df["label"] == instance_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found.")
    row = row.iloc[0]

    src = Path(str(row["source_path"]))
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"Source image missing: {src.name}")

    # Load + draw bbox
    try:
        if src.suffix.lower() in {".tif", ".tiff"}:
            import tifffile
            arr = tifffile.imread(str(src))
            if arr.ndim == 3:
                arr = arr[0]
            # Normalise to 0..255 uint8
            a = arr.astype("float32")
            a = (a - a.min()) / (a.max() - a.min() + 1e-8) * 255.0
            img = Image.fromarray(a.astype("uint8")).convert("RGB")
        else:
            img = Image.open(src).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load image: {exc}")

    y1, x1, y2, x2 = (int(row[k]) for k in ("bbox_y1", "bbox_x1", "bbox_y2", "bbox_x2"))
    draw = ImageDraw.Draw(img)
    # Thicker outline + corner ticks so the bbox is visible even after downscale
    w = max(6, min(img.size) // 120)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=w)
    # Corner brackets that extend outside the box for visibility
    bw, bh = x2 - x1, y2 - y1
    tick = max(20, min(bw, bh) // 2)
    for (cx, cy, dx, dy) in [
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
    ]:
        draw.line([(cx, cy), (cx + dx * tick, cy)], fill=(255, 220, 0), width=w)
        draw.line([(cx, cy), (cx, cy + dy * tick)], fill=(255, 220, 0), width=w)

    # Downscale for transport
    img.thumbnail((max_size, max_size), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return Response(content=buf.getvalue(), media_type="image/png")


@router.get("/api/image-preview")
async def get_image_preview(source_file: str, max_size: int = 720):
    """
    Return a downscaled preview of a source image (no bbox overlay).
    Used by the image-overview chart hover handler.
    """
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="No metrics CSV.")
    df = pd.read_csv(DATA_PATH)
    if "source_path" not in df.columns:
        raise HTTPException(status_code=404, detail="source_path column missing.")
    match = df[df["source_file"] == source_file]
    if match.empty:
        # try substring fallback
        match = df[df["source_file"].str.contains(source_file, case=False, na=False)]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"No image matches '{source_file}'.")
    src = Path(str(match.iloc[0]["source_path"]))
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"Source image not on disk: {src.name}")
    try:
        if src.suffix.lower() in {".tif", ".tiff"}:
            import tifffile
            arr = tifffile.imread(str(src))
            if arr.ndim == 3:
                arr = arr[0]
            a = arr.astype("float32")
            a = (a - a.min()) / (a.max() - a.min() + 1e-8) * 255.0
            img = Image.fromarray(a.astype("uint8")).convert("RGB")
        else:
            img = Image.open(src).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load image: {exc}")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return Response(content=buf.getvalue(), media_type="image/png")


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


@router.post("/api/chat")
async def chat(request: Request):
    """Natural-language research assistant with RAG over markdown docs + CSV tools."""
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse(content={"answer": "Please ask a question.", "tool_calls": []})
    try:
        from src.research_agent import answer_question
        result = answer_question(question)
        return JSONResponse(content=result)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"answer": f"Agent error: {exc}", "tool_calls": []},
        )


@router.get("/api/chat/key-status")
async def chat_key_status():
    """Tell the UI whether an ANTHROPIC_API_KEY is configured (no value leak)."""
    import os
    # Hydrate from .env if present, same logic as the agent
    if not os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:
            pass
    key = os.environ.get("ANTHROPIC_API_KEY") or ""
    return JSONResponse(content={
        "configured": bool(key and key.startswith("sk-")),
        "source": ".env or environment" if key else "none",
    })


@router.post("/api/chat/set-key")
async def chat_set_key(request: Request):
    """
    Accept an API key from the UI and set it in the current process env.
    NOT persisted to disk. Use a .env file for persistence across restarts.
    """
    import os
    body = await request.json()
    key = (body.get("key") or "").strip()
    if not key.startswith("sk-"):
        return JSONResponse(status_code=400, content={"ok": False, "error": "Key must start with 'sk-'."})
    os.environ["ANTHROPIC_API_KEY"] = key
    return JSONResponse(content={"ok": True, "configured": True})

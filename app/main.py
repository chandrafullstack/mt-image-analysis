"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.routes import router

app = FastAPI(title="Mitochondria Health Dashboard")

# Serve static files (JS, CSS)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Serve crop images
crops_dir = Path("outputs/crops")
crops_dir.mkdir(parents=True, exist_ok=True)
app.mount("/crops", StaticFiles(directory=str(crops_dir)), name="crops")

app.include_router(router)

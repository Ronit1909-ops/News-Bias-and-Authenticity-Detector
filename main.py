from pathlib import Path
from typing import Optional
import joblib
import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sklearn  

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

app = FastAPI(title="News Bias Detector")

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_model: Optional[object] = None
_vectorizer: Optional[object] = None


def load_artifacts() -> tuple[Optional[object], Optional[object], list[str]]:
    errors: list[str] = []
    global _model, _vectorizer

    if _vectorizer is None:
        try:
            _vectorizer = joblib.load(VECTORIZER_PATH)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Failed to load vectorizer: {exc}")

    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Failed to load model: {exc}")

    return _model, _vectorizer, errors


@app.get("/")
async def index(request: Request):
    _, _, errors = load_artifacts()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "errors": errors,
            "prediction": None,
            "input_text": "",
        },
    )


@app.post("/predict")
async def predict(request: Request, text: str = Form(...)):
    model, vectorizer, errors = load_artifacts()

    prediction_text: Optional[str] = None
    if model is not None and vectorizer is not None:
        try:
            features = vectorizer.transform([text])
            pred = model.predict(features)
            # Handle various model outputs
            label = pred[0] if hasattr(pred, "__getitem__") else pred
            prediction_text = str(label)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Prediction failed: {exc}")
    else:
        if model is None:
            errors.append("Model not loaded")
        if vectorizer is None:
            errors.append("Vectorizer not loaded")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "errors": errors,
            "prediction": prediction_text,
            "input_text": text,
        },
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

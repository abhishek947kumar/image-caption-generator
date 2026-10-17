"""
FastAPI Backend Application for the Image Caption Generator.
Exposes RESTful endpoints for image captioning, beam search exploration, BLEU evaluation, and model inspection.
"""
import os
import io
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

from backend.generator import CaptionGenerator
from backend.sample_data import SAMPLE_IMAGES, get_sample_by_id
from backend.metrics import calculate_bleu

app = FastAPI(
    title="Neural Image Caption Generator API",
    description="Deep Learning CNN-LSTM Image Captioning service with Greedy & Beam Search",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize caption generator engine
generator = CaptionGenerator()

class EvaluateRequest(BaseModel):
    candidate: str
    references: List[str]

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "device": generator.device,
        "backbone": generator.feature_extractor.backbone_name,
        "has_trained_weights": generator.has_trained_weights,
        "vocab_size": generator.tokenizer.vocab_size
    }

@app.get("/api/samples")
async def get_samples():
    """Return preset demo images with references and metadata."""
    samples_list = []
    for s in SAMPLE_IMAGES:
        samples_list.append({
            **s,
            "image_url": f"/data/samples/{s['filename']}"
        })
    return {"samples": samples_list}

@app.get("/api/model-info")
async def get_model_info():
    """Return model architectural specifications and parameter statistics."""
    params = generator.model.count_parameters()
    return {
        "architecture": "CNN Encoder + LSTM Decoder (Merge & Recurrent)",
        "backbone": generator.feature_extractor.backbone_name,
        "feature_dim": generator.feature_extractor.feature_dim,
        "embed_dim": generator.model.embed_dim,
        "hidden_dim": generator.model.hidden_dim,
        "vocab_size": generator.tokenizer.vocab_size,
        "total_parameters": params["total_parameters"],
        "trainable_parameters": params["trainable_parameters"],
        "device": generator.device,
        "has_custom_weights": generator.has_trained_weights
    }

@app.post("/api/caption")
async def generate_caption(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    strategy: str = Form("beam"),
    beam_width: int = Form(3),
    temperature: float = Form(1.0),
    max_length: int = Form(25)
):
    """Generate caption for an uploaded image or preset sample."""
    pil_image = None
    target_sample_id = None

    if file and file.filename:
        try:
            contents = await file.read()
            pil_image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    elif sample_id:
        sample = get_sample_by_id(sample_id)
        if not sample:
            raise HTTPException(status_code=404, detail="Sample image not found")
        sample_path = os.path.join("data", "samples", sample["filename"])
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail="Sample file missing from disk")
        pil_image = Image.open(sample_path)
        target_sample_id = sample_id
    else:
        raise HTTPException(status_code=400, detail="Either an image file or sample_id must be provided")

    # Run inference
    try:
        if strategy.lower() == "greedy":
            result = generator.generate_caption_greedy(
                image=pil_image,
                max_length=max_length,
                temperature=temperature,
                sample_id=target_sample_id
            )
        else: # beam
            result = generator.generate_caption_beam(
                image=pil_image,
                beam_width=beam_width,
                max_length=max_length,
                temperature=temperature,
                sample_id=target_sample_id
            )

        # If sample has reference captions, calculate automatic BLEU scores!
        sample_meta = get_sample_by_id(target_sample_id) if target_sample_id else None
        if sample_meta and sample_meta.get("references"):
            bleu_scores = calculate_bleu(result["caption"], sample_meta["references"])
            result["bleu_evaluation"] = bleu_scores
            result["reference_captions"] = sample_meta["references"]

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/api/evaluate")
async def evaluate_caption(payload: EvaluateRequest):
    """Compute BLEU-1 through BLEU-4 metrics for candidate against references."""
    if not payload.candidate.strip():
        raise HTTPException(status_code=400, detail="Candidate caption cannot be empty")
    if not payload.references:
        raise HTTPException(status_code=400, detail="At least one reference caption is required")
        
    scores = calculate_bleu(payload.candidate, payload.references)
    return scores

# Mount sample images
os.makedirs("data/samples", exist_ok=True)
app.mount("/data/samples", StaticFiles(directory="data/samples"), name="samples")

# Serve frontend
os.makedirs("frontend", exist_ok=True)
app.mount("/static", StaticFiles(directory="frontend"), name="frontend_static")

@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")

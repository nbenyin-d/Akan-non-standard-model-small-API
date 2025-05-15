# API_render.py - Enhanced API with proper error handling and wake-up endpoint
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
import uvicorn
import torch
import os
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Akan-non-standard-model-small-API", description="Transcribes non-standard Akan audio to text")

# Add CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (customize for production)
    allow_methods=["GET", "POST"],  # Added GET for health check and wake-up
    allow_headers=["*"],
)

# Initialize the ASR pipeline with error handling
try:
    logger.info("Loading ASR model...")
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="Tree-Diagram/whisper-small_Akan_non_standardspeech",
        device=device
    )
    logger.info("ASR model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load ASR model: {str(e)}")
    asr_pipeline = None

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Wake-up endpoint for the API"""
    return {"message": "Akan ASR API is running", "status": "ok"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if asr_pipeline is None:
        raise HTTPException(status_code=500, detail="ASR model not loaded")
    return {"status": "healthy", "model": "whisper-small_Akan_non_standardspeech"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe Akan audio to text"""
    if asr_pipeline is None:
        raise HTTPException(status_code=500, detail="ASR model not loaded")
    
    try:
        logger.info(f"Received file: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
        
        # Check if file is valid
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        logger.info(f"Processing audio file of size {len(audio_bytes)} bytes")
        
        # Apply the ASR pipeline
        result = asr_pipeline(audio_bytes)
        transcript = result["text"]
        
        logger.info(f"Transcription completed successfully. Length: {len(transcript)} characters")
        
        return {"transcript": transcript}
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Use Render's PORT or default to 10000
    
    # Log startup information
    logger.info(f"Starting server on port {port}")
    
    uvicorn.run("API_render:app", host="0.0.0.0", port=port)

from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import uvicorn
import torch  # Add this import
import os
from fastapi.middleware.cors import CORSMiddleware  # Add this line

app = FastAPI(title= "Akan ASR_nonstanderd_small_API", description = "Transcribes non-standard Akan audio to text")

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="Tree-Diagram/whisper-small_Akan_non_standardspeech"
)

# Add CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (customize for production)
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    result = asr_pipeline(audio_bytes)
    transcript = result["text"]
    return {"transcript": transcript}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Use Render's PORT or default to 10000
    uvicorn.run(app, host="0.0.0.0", port=port)
     

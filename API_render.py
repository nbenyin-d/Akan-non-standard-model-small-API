from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import uvicorn
import torch  # Add this import

app = FastAPI(title= "Akan ASR_nonstanderd_small_API", description = "Transcribes non-standard Akan audio to text")

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="Tree-Diagram/whisper-small_Akan_non_standardspeech"
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    result = asr_pipeline(audio_bytes)
    transript = result["text"]
    return {"transcript": transcript}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Use Render's PORT or default to 10000
    uvicorn.run(app, host="0.0.0.0", port=port)
     

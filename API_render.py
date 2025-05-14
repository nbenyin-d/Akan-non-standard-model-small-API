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

import nest_asyncio
nest_asyncio.apply()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)  
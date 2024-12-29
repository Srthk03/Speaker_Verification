from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import torchaudio
import torchaudio.transforms as T
import tempfile
import os
import uvicorn
from fastapi.responses import JSONResponse

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the Siamese model
class L1Distance(layers.Layer):
    def call(self, vectors):
        return tf.reduce_sum(tf.abs(vectors[0] - vectors[1]), axis=1)

model = models.load_model("siamese_model.keras", custom_objects={'L1Distance': L1Distance})

def extract_mfcc(audio_file, n_mfcc=13):
    try:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_file.read())
            temp_path = temp_file.name

        # Load and process the audio
        waveform, sample_rate = torchaudio.load(temp_path)
        
        transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
        )
        mfcc = transform(waveform)
        mfcc_mean = mfcc.mean(dim=-1).detach().numpy()

        # Clean up temporary file
        os.unlink(temp_path)
        
        return mfcc_mean
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def predict_pair(reference_mfcc, incoming_mfcc):
    reference_mfcc = np.expand_dims(reference_mfcc, axis=0)
    incoming_mfcc = np.expand_dims(incoming_mfcc, axis=0)
    
    prediction = model.predict([reference_mfcc, incoming_mfcc])
    
    is_same_person = prediction[0] > 0.4
    probability = float(prediction[0])
    
    return is_same_person, probability

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/compare")
async def compare_audio(reference: UploadFile = File(...), sample: UploadFile = File(...)):
    try:
        # Extract MFCC features
        reference_mfcc = extract_mfcc(reference.file)
        sample_mfcc = extract_mfcc(sample.file)
        
        if reference_mfcc is None or sample_mfcc is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not process audio files"}
            )
        
        # Make prediction
        is_same_person, probability = predict_pair(reference_mfcc, sample_mfcc)
        
        return {
            "result": "Same person" if is_same_person else "Different person",
            "probability": f"{probability:.2%}"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
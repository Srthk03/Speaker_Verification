# Speaker Verification System

This repository contains a Speaker Verification System that classifies real-time incoming audio as either the target speaker or a non-target speaker. The system is built using a Siamese Neural Network architecture, leveraging MFCC feature extraction for audio preprocessing.

## Problem Statement
Develop a system that takes a reference audio recording of a target speaker and classifies real-time incoming audio as either the target speaker or a non-target speaker.

### Requirements:
1. Preprocess audio and extract features (e.g., MFCCs).
2. Build and train a model for speaker verification.
3. Evaluate the system using relevant metrics (e.g., accuracy, F1-score).
4. Submit code and briefly describe the approach.

---

## Approach

### 1. Audio Preprocessing
- **MFCC Extraction**: Audio files are preprocessed to extract Mel Frequency Cepstral Coefficients (MFCCs) using `torchaudio`. 
- **Caching**: The extracted MFCC features are cached in HDF5 format for efficient processing.
- **Dataset**: The dataset comprises 50 speakers, each with 30-50 audio samples. 30,000 target-target and 30,000 target-non-target pairs were generated for model training and evaluation.

### 2. Siamese Neural Network
- **Architecture**: The Siamese network uses two identical subnetworks to process pairs of MFCC features. A custom L1 distance layer computes the similarity between embeddings.
- **Training**: The network is trained using binary cross-entropy loss to classify whether a pair of inputs belongs to the same speaker (target-target) or different speakers (target-non-target).

### 3. Evaluation
- The model's performance is evaluated using metrics such as:
  - **Accuracy**
  - **F1-Score**
  - **ROC-AUC**

### 4. Real-Time Prediction
- The system allows real-time verification of incoming audio by comparing it with a reference audio sample.
- The classification output specifies whether the incoming audio is from the target speaker or a non-target speaker, along with a confidence score.

---

## Features
- **Scalable MFCC Caching**: Efficient HDF5 storage for pre-extracted features.
- **Balanced Pair Generation**: Generates equal numbers of target-target and target-non-target pairs.
- **Real-Time Speaker Verification**: Processes incoming audio in real-time for speaker classification.
- **Web Application**: A user-friendly web interface to test and visualize the speaker verification system.

---

## How to Run

### Prerequisites
- Python 3.9 or above
- TensorFlow
- PyTorch and Torchaudio
- NumPy, h5py, scikit-learn
- FastAPI, Uvicorn

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/speaker-verification.git
   cd speaker-verification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
   - Place the dataset of audio files in the specified folder structure.
   - Run the preprocessing script to extract and cache MFCC features.

4. Train the model:
   Run the train notebook file in a notebook environment
   
5. Run the web application:
   ```bash
   streamlit run app.py
   ```

---

## Web Application
The system includes a web application where users can:
- Upload a reference audio file of the target speaker.
- Provide incoming audio for real-time verification.
- View the classification results (Target-Target or Target-Non-Target) and confidence score.

### Screenshots
(Placeholder for screenshots)
1. **Home Page**:
   - <img width="1440" alt="Screenshot 2024-12-29 at 6 33 13 PM" src="https://github.com/user-attachments/assets/cd00cebf-d354-464e-8fa7-8ab6f5033213" />


2. **Prediction Page**:
   - <img width="1440" alt="Screenshot 2024-12-29 at 6 33 38 PM" src="https://github.com/user-attachments/assets/e68f577f-e77c-458c-a85d-7e88c6063d75" />

---

## Evaluation Metrics
The model was evaluated on a balanced dataset of 30,000 target-target and 30,000 target-non-target pairs using the following metrics:
- **Accuracy**: Measures overall classification performance.
- **F1-Score**: Balances precision and recall.
- **ROC-AUC**: Evaluates model discrimination ability.

---

## Contact
This code can encounter dependency issues.
For any issues or questions, feel free to contact me at [ag.sarthak03@gmail.com].



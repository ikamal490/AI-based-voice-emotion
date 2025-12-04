"""
Improved Voice Emotion Recorder + Trainer + Predictor
Steps:
1) Press 1 to record training samples for emotions (recommended 5 samples each).
2) Press 2 to train model (uses augmentation to grow dataset).
3) Press 3 to test prediction (model must be trained or loaded).
4) Press 0 to exit.

Made for beginners — Kamal
"""

import os
import sys
import time
import sounddevice as sd
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ====== Config ======
EMOTIONS = ["happy", "sad", "angry", "calm"]
SAMPLES_PER_EMOTION = 6     # record 5-8 samples each for better accuracy
RECORD_SECONDS = 3          # 3 seconds per sample (you can increase to 4-5)
SAMPLE_RATE = 22050
DATA_DIR = "voice_samples"
MODEL_FILE = "emotion_model.joblib"
SCALER_FILE = "scaler.joblib"

os.makedirs(DATA_DIR, exist_ok=True)
for e in EMOTIONS:
    os.makedirs(os.path.join(DATA_DIR, e), exist_ok=True)

# ====== Audio recording helper ======
def record_audio(duration=RECORD_SECONDS, sr=SAMPLE_RATE):
    print(f"  → Recording for {duration}s. Speak now...")
    rec = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = rec.flatten()
    # normalize to -1..1 range just in case
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio

# ====== Save sample to file (wav) ======
def save_wav(filename, audio, sr=SAMPLE_RATE):
    # librosa.output.write_wav is deprecated; use soundfile via librosa
    import soundfile as sf
    sf.write(filename, audio, sr)
    print("   saved:", filename)

# ====== Feature extraction ======
def extract_features(y, sr=SAMPLE_RATE):
    # ensure float32
    y = np.asarray(y, dtype=np.float32)
    # if too short, pad
    if len(y) < 1:
        return None
    # mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = None
    try:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    except Exception:
        tonnetz = np.zeros((6, max(1, mfcc.shape[1])))
    feat = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spec_contrast, axis=1),
        np.mean(tonnetz, axis=1)
    ])
    return feat

# ====== Simple augmentations ======
def augment_time_stretch(y, rate=1.1):
    return librosa.effects.time_stretch(y, rate)

def augment_pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr, n_steps)

# ====== Collect training samples ======
def collect_samples():
    print("\n--- RECORDING TRAINING SAMPLES ---")
    for emo in EMOTIONS:
        print(f"\nRecord samples for emotion: {emo.upper()}")
        existing = len(os.listdir(os.path.join(DATA_DIR, emo)))
        start_index = existing
        for i in range(SAMPLES_PER_EMOTION - existing):
            input(f"Press Enter and speak a {emo} phrase ({i+1 + existing}/{SAMPLES_PER_EMOTION})...")
            audio = record_audio()
            filename = os.path.join(DATA_DIR, emo, f"{int(time.time())}_{i}.wav")
            save_wav(filename, audio)
    print("\nAll samples recorded. You can press 2 to train the model.")

# ====== Build dataset (load files + augment) ======
def build_dataset(augment=True):
    X = []
    y = []
    for emo in EMOTIONS:
        folder = os.path.join(DATA_DIR, emo)
        files = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".wav")]
        for fp in files:
            audio, sr = librosa.load(fp, sr=SAMPLE_RATE, mono=True)
            feat = extract_features(audio, sr)
            if feat is not None:
                X.append(feat); y.append(emo)
            if augment:
                # time stretch
                try:
                    ts = augment_time_stretch(audio, rate=0.9)
                    feat_ts = extract_features(ts, sr)
                    if feat_ts is not None:
                        X.append(feat_ts); y.append(emo)
                    ps = augment_pitch_shift(audio, sr, n_steps=2)
                    feat_ps = extract_features(ps, sr)
                    if feat_ps is not None:
                        X.append(feat_ps); y.append(emo)
                except Exception:
                    pass
    return np.array(X), np.array(y)

# ====== Train model ======
def train_and_save():
    print("\nBuilding dataset...")
    X, y = build_dataset(augment=True)
    if len(X) < 8:
        print("Not enough data. Record more samples (recommended 5-8 per class).")
        return
    print("Dataset shape:", X.shape, y.shape)
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    # model (SVM)
    print("Training classifier...")
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    # evaluate
    y_pred = clf.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    # save
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\nModel saved to {MODEL_FILE} and scaler saved to {SCALER_FILE}")

# ====== Load model ======
def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print("Model or scaler not found. Train first (press 2).")
        return None, None
    clf = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("Loaded model and scaler.")
    return clf, scaler

# ====== Predict on live audio ======
def live_predict(clf, scaler):
    print("\n--- LIVE PREDICTION ---")
    input("Press Enter and speak now...")
    audio = record_audio()
    feat = extract_features(audio, SAMPLE_RATE)
    if feat is None:
        print("Could not extract features.")
        return
    Xs = scaler.transform([feat])
    pred = clf.predict(Xs)[0]
    probs = clf.predict_proba(Xs)[0]
    # pretty print
    print("\n🔮 Prediction:", pred.upper())
    for emo, p in zip(clf.classes_, probs):
        print(f"   {emo:6s} : {p:.2f}")

# ====== Main menu ======
def main_menu():
    print("Voice Emotion Trainer & Predictor")
    print("1 - Record training samples (recommended first)")
    print("2 - Train model (after recording)")
    print("3 - Live test prediction (after training)")
    print("0 - Exit")
    while True:
        choice = input("\nEnter choice: ").strip()
        if choice == "1":
            collect_samples()
        elif choice == "2":
            train_and_save()
        elif choice == "3":
            clf, scaler = load_model()
            if clf is not None:
                live_predict(clf, scaler)
        elif choice == "0":
            print("Exiting.")
            break
        else:
            print("Invalid. Choose 1,2,3 or 0.")

if __name__ == "__main__":
    main_menu()

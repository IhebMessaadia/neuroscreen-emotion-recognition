import pickle
import numpy as np
import sys
import os

# ── 1. LOAD EEG, ECG, GSR MODELS ─────────────────────────
print("Loading models...")

with open('eeg_model.pkl', 'rb') as f:
    eeg_model = pickle.load(f)

with open('ecg_model.pkl', 'rb') as f:
    ecg_model = pickle.load(f)

with open('gsr_model.pkl', 'rb') as f:
    gsr_model = pickle.load(f)

# ── 2. LOAD FACE MODEL ────────────────────────────────────
sys.path.insert(0, 'deployment_video_branch_v8d')
from video_branch_v8d_inference import predict_video_emotion

print("All models loaded!")

EMOTIONS = ['happy', 'sad', 'stressed', 'neutral']

# ── 3. GET PROBABILITIES FROM SKLearn MODELS ──────────────
def get_probs(model, features):
    classes = list(model.classes_)
    probs_raw = model.predict_proba([features])[0]
    probs = {'happy': 0.0, 'sad': 0.0, 'stressed': 0.0, 'neutral': 0.0}
    for i, c in enumerate(classes):
        if c in probs:
            probs[c] = float(probs_raw[i])
    # normalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: v/total for k, v in probs.items()}
    return probs

# ── 4. FUSION FUNCTION ────────────────────────────────────
def predict_emotion(eeg_features, ecg_features, gsr_features, video_path=None):

    # Get probabilities from each model
    eeg_probs = get_probs(eeg_model, eeg_features)
    ecg_probs = get_probs(ecg_model, ecg_features)
    gsr_probs = get_probs(gsr_model, gsr_features)

    print(f"EEG  probs: {eeg_probs}")
    print(f"ECG  probs: {ecg_probs}")
    print(f"GSR  probs: {gsr_probs}")

    # Face model
    if video_path and os.path.exists(video_path):
        face_result = predict_video_emotion(video_path)
        face_probs = face_result['probabilities']
        print(f"Face probs: {face_probs}")
        # Weighted average: face gets more weight (73% accuracy)
        weights = {'eeg': 0.20, 'ecg': 0.20, 'gsr': 0.25, 'face': 0.35}
        final_probs = {}
        for e in EMOTIONS:
            final_probs[e] = (
                weights['eeg']  * eeg_probs.get(e, 0) +
                weights['ecg']  * ecg_probs.get(e, 0) +
                weights['gsr']  * gsr_probs.get(e, 0) +
                weights['face'] * face_probs.get(e, 0)
            )
    else:
        print("No video provided — using EEG + ECG + GSR only")
        weights = {'eeg': 0.33, 'ecg': 0.33, 'gsr': 0.34}
        final_probs = {}
        for e in EMOTIONS:
            final_probs[e] = (
                weights['eeg'] * eeg_probs.get(e, 0) +
                weights['ecg'] * ecg_probs.get(e, 0) +
                weights['gsr'] * gsr_probs.get(e, 0)
            )

    # Final decision
    final = max(final_probs, key=final_probs.get)
    confidence = final_probs[final] * 100

    print(f"\nFinal probs: {final_probs}")
    print(f"\n✅ Final emotion: {final} ({confidence:.1f}% confidence)")

    # Screen action
    if final == 'stressed':
        print("🔴 ACTION: Block screen — stress detected!")
    elif final == 'sad':
        print("🔴 ACTION: Block screen — sadness detected!")
    elif final == 'neutral':
        print("🟡 ACTION: Warning shown to user")
    else:
        print("🟢 ACTION: Screen normal — user is happy!")

    return final, final_probs

# ── 5. TEST ───────────────────────────────────────────────
print("\n--- Testing fusion ---")

# eeg: [eeg_mean, eeg_std, eeg_max, eeg_min, valence, arousal]
# ecg: [ecg_mean, ecg_std, ecg_max, ecg_min, ecg_range, ecg_median]
# gsr: [gsr_mean, gsr_std, gsr_max, gsr_min, gsr_range, gsr_median]

predict_emotion(
    eeg_features=[-18.8, 0.15, -17.5, -19.0, 0, 2],
    ecg_features=[-18.8, 0.15, -17.5, -19.0, 1.5, -18.8],
    gsr_features=[250.0, 5.0, 260.0, 240.0, 20.0, 250.0],
    video_path=None
)
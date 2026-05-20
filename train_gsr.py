import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ── 1. LOAD DATA ──────────────────────────────────────────
print("Loading GSR dataset...")
df = pd.read_csv('ECG + GSR/GSR_Dataset_Labeled.csv')
print(f"Dataset shape: {df.shape}")

# ── 2. REMAP LABELS ───────────────────────────────────────
label_map = {
    'Excited': 'happy',
    'Relaxed': 'neutral',
    'Stress':  'stressed',
    'Neutral': 'neutral'
}
df['emotion'] = df['emotion_label'].map(label_map)
print("\nEmotion distribution:")
print(df['emotion'].value_counts())

# ── 3. FEATURES & LABELS ──────────────────────────────────
features = ['gsr_mean', 'gsr_std', 'gsr_max', 'gsr_min', 'gsr_range', 'gsr_median']
X = df[features]
y = df['emotion']

# ── 4. SPLIT & TRAIN ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print("\nTraining GSR model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── 5. EVALUATE ───────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# ── 6. SAVE ───────────────────────────────────────────────
with open('gsr_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as gsr_model.pkl")
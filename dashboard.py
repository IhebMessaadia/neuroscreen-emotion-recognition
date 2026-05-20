import tkinter as tk
import pickle
import numpy as np
import sys
import os
import cv2
import threading
import tempfile
import time
import pandas as pd
from PIL import Image, ImageTk

# ── LOAD MODELS ───────────────────────────────────────────
print("Loading models...")
with open('eeg_model.pkl', 'rb') as f:
    eeg_model = pickle.load(f)
with open('ecg_model.pkl', 'rb') as f:
    ecg_model = pickle.load(f)
with open('gsr_model.pkl', 'rb') as f:
    gsr_model = pickle.load(f)

sys.path.insert(0, 'deployment_video_branch_v8d')
from video_branch_v8d_inference import predict_video_emotion

# ── LOAD REAL DATASET ROWS ────────────────────────────────
eeg_df = pd.read_csv('EEG Emotion & Stress Recognition/EEG_Dataset_Labeled.csv')
ecg_df = pd.read_csv('ECG + GSR/ECG_Dataset_Labeled.csv')
gsr_df = pd.read_csv('ECG + GSR/GSR_Dataset_Labeled.csv')

EEG_FEATURES = ['eeg_mean', 'eeg_std', 'eeg_max', 'eeg_min', 'valence', 'arousal']
ECG_FEATURES = ['ecg_mean', 'ecg_std', 'ecg_max', 'ecg_min', 'ecg_range', 'ecg_median']
GSR_FEATURES = ['gsr_mean', 'gsr_std', 'gsr_max', 'gsr_min', 'gsr_range', 'gsr_median']

EMOTIONS = ['happy', 'sad', 'stressed', 'neutral']
EMOTION_COLORS = {
    'stressed': '#FF4444',
    'sad':      '#FF4444',
    'neutral':  '#FFA500',
    'happy':    '#44BB44',
}
EMOTION_ICONS = {
    'stressed': '🔴',
    'sad':      '🔴',
    'neutral':  '🟡',
    'happy':    '🟢',
}

print("All models loaded!")

# ── SHARED CAMERA ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
cam_lock = threading.Lock()
latest_frame = [None]

def camera_thread():
    while True:
        with cam_lock:
            ret, frame = cap.read()
            if ret:
                latest_frame[0] = frame.copy()
        time.sleep(0.03)

t_cam = threading.Thread(target=camera_thread, daemon=True)
t_cam.start()

# ── HELPER FUNCTIONS ──────────────────────────────────────
def get_probs(model, features):
    classes = list(model.classes_)
    probs_raw = model.predict_proba([features])[0]
    probs = {'happy': 0.0, 'sad': 0.0, 'stressed': 0.0, 'neutral': 0.0}
    for i, c in enumerate(classes):
        if c in probs:
            probs[c] = float(probs_raw[i])
    total = sum(probs.values())
    if total > 0:
        probs = {k: v/total for k, v in probs.items()}
    return probs

def sample_real_row(df, features):
    row = df.sample(1).iloc[0]
    return [row[f] for f in features]

def record_clip_from_shared(duration=5, fps=20):
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp.close()
    frame = latest_frame[0]
    if frame is None:
        return None
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))
    start = time.time()
    while time.time() - start < duration:
        with cam_lock:
            f = latest_frame[0]
        if f is not None:
            out.write(f)
        time.sleep(1/fps)
    out.release()
    return tmp.name

# ── GUI ───────────────────────────────────────────────────
root = tk.Tk()
root.title("NeuroScreen — Live Emotion AI Dashboard")
root.geometry("1100x720")
root.configure(bg='#1e1e2e')
root.resizable(False, False)

tk.Label(root, text="🧠 NeuroScreen: AI-Based Emotion Detection System",
         font=('Helvetica', 17, 'bold'), bg='#1e1e2e', fg='white').pack(pady=10)

status_var = tk.StringVar(value='Press Start to begin live detection')
tk.Label(root, textvariable=status_var, font=('Helvetica', 10),
         bg='#1e1e2e', fg='#aaaacc').pack()

main = tk.Frame(root, bg='#1e1e2e')
main.pack(fill='both', expand=True, padx=20, pady=5)

left = tk.Frame(main, bg='#1e1e2e')
left.pack(side='left', fill='both', expand=True)

def make_panel(parent, title, color):
    frame = tk.Frame(parent, bg='#2a2a3e', relief='flat')
    frame.pack(fill='x', pady=4, ipady=6, ipadx=8)
    tk.Label(frame, text=title, font=('Helvetica', 11, 'bold'),
             bg='#2a2a3e', fg=color).pack(anchor='w', padx=10, pady=(6,2))
    pred_var = tk.StringVar(value='—')
    pred_label = tk.Label(frame, textvariable=pred_var,
                          font=('Helvetica', 18, 'bold'), bg='#2a2a3e', fg='white')
    pred_label.pack(anchor='w', padx=10)
    prob_var = tk.StringVar(value='')
    tk.Label(frame, textvariable=prob_var, font=('Helvetica', 9),
             bg='#2a2a3e', fg='#aaaacc').pack(anchor='w', padx=10)
    return pred_var, pred_label, prob_var

eeg_var,  eeg_lbl,  eeg_prob  = make_panel(left, "🧠  EEG  (Brainwaves)",        '#7eb8f7')
ecg_var,  ecg_lbl,  ecg_prob  = make_panel(left, "❤️   ECG  (Heart Rate)",        '#f7a07e')
gsr_var,  gsr_lbl,  gsr_prob  = make_panel(left, "💧  GSR  (Skin Conductance)",   '#f7e07e')
face_var, face_lbl, face_prob = make_panel(left, "😐  Face  (Live Webcam)",       '#a0f7b8')

# ── RIGHT PANEL ───────────────────────────────────────────
right = tk.Frame(main, bg='#2a2a3e', width=280)
right.pack(side='right', fill='both', padx=(20,0), pady=4)
right.pack_propagate(False)

tk.Label(right, text="📷 Live Camera", font=('Helvetica', 10, 'bold'),
         bg='#2a2a3e', fg='#aaaacc').pack(pady=(10,2))

cam_display = tk.Label(right, bg='#1a1a2e')
cam_display.pack(padx=10, pady=5)

tk.Label(right, text="Final Decision", font=('Helvetica', 13, 'bold'),
         bg='#2a2a3e', fg='#aaaacc').pack(pady=(5,2))

final_var = tk.StringVar(value='—')
final_label = tk.Label(right, textvariable=final_var,
                       font=('Helvetica', 24, 'bold'), bg='#2a2a3e', fg='white')
final_label.pack(pady=3)

conf_var = tk.StringVar(value='')
tk.Label(right, textvariable=conf_var, font=('Helvetica', 10),
         bg='#2a2a3e', fg='#aaaacc').pack()

action_var = tk.StringVar(value='')
action_label = tk.Label(right, textvariable=action_var,
                        font=('Helvetica', 10), bg='#2a2a3e', fg='white',
                        wraplength=240, justify='center')
action_label.pack(pady=8)

screen_frame = tk.Frame(right, bg='#FF4444', height=60)
screen_label = tk.Label(screen_frame, text="🛑  SCREEN BLOCKED",
                        font=('Helvetica', 11, 'bold'), bg='#FF4444', fg='white')

# ── WEBCAM PREVIEW UPDATE ─────────────────────────────────
def update_preview():
    frame = latest_frame[0]
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (240, 180))
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        cam_display.imgtk = imgtk
        cam_display.config(image=imgtk)
    root.after(33, update_preview)

update_preview()

# ── ANALYSIS ──────────────────────────────────────────────
running = False

def update_panel(var, lbl, prob_var, probs):
    pred = max(probs, key=probs.get)
    var.set(f"{EMOTION_ICONS.get(pred,'')} {pred.capitalize()}")
    lbl.config(fg=EMOTION_COLORS.get(pred, 'white'))
    prob_var.set(f"happy:{probs['happy']:.2f}  sad:{probs['sad']:.2f}  stressed:{probs['stressed']:.2f}  neutral:{probs['neutral']:.2f}")

def run_analysis():
    global running
    while running:
        status_var.set("🎥 Recording 3 seconds from webcam...")
        root.update()

        video_path = record_clip_from_shared(duration=3, fps=10)

        status_var.set("🔍 Analyzing emotion...")
        root.update()

        eeg_feats = sample_real_row(eeg_df, EEG_FEATURES)
        ecg_feats = sample_real_row(ecg_df, ECG_FEATURES)
        gsr_feats = sample_real_row(gsr_df, GSR_FEATURES)

        eeg_probs = get_probs(eeg_model, eeg_feats)
        ecg_probs = get_probs(ecg_model, ecg_feats)
        gsr_probs = get_probs(gsr_model, gsr_feats)

        try:
            face_result = predict_video_emotion(video_path)
            face_probs = face_result['probabilities']
        except Exception as e:
            print(f"Face model error: {e}")
            face_probs = {'happy': 0.25, 'sad': 0.25, 'stressed': 0.25, 'neutral': 0.25}

        try:
            os.remove(video_path)
        except:
            pass

        update_panel(eeg_var,  eeg_lbl,  eeg_prob,  eeg_probs)
        update_panel(ecg_var,  ecg_lbl,  ecg_prob,  ecg_probs)
        update_panel(gsr_var,  gsr_lbl,  gsr_prob,  gsr_probs)
        update_panel(face_var, face_lbl, face_prob, face_probs)

        weights = {'eeg': 0.20, 'ecg': 0.20, 'gsr': 0.25, 'face': 0.35}
        final_probs = {}
        for e in EMOTIONS:
            final_probs[e] = (weights['eeg']  * eeg_probs.get(e,0) +
                              weights['ecg']  * ecg_probs.get(e,0) +
                              weights['gsr']  * gsr_probs.get(e,0) +
                              weights['face'] * face_probs.get(e,0))

        final = max(final_probs, key=final_probs.get)
        confidence = final_probs[final] * 100
        color = EMOTION_COLORS.get(final, 'white')

        final_var.set(f"{EMOTION_ICONS.get(final,'')} {final.capitalize()}")
        final_label.config(fg=color)
        conf_var.set(f"Confidence: {confidence:.1f}%")
        right.config(bg=color)
        final_label.config(bg=color)
        action_label.config(bg=color)

        if final in ['stressed', 'sad']:
            action_var.set("🛑 Screen is being BLOCKED\nPlease take a break!")
            screen_frame.pack(fill='x', padx=10, pady=8)
            screen_label.pack(expand=True)
        elif final == 'neutral':
            action_var.set("⚠️ Warning shown to user\nMonitoring continues...")
            screen_frame.pack_forget()
        else:
            action_var.set("✅ All good!\nScreen access normal.")
            screen_frame.pack_forget()

        status_var.set("✅ Done — recording next clip...")
        root.update()

def toggle():
    global running
    if not running:
        running = True
        btn.config(text="⏹  Stop", bg='#FF4444')
        t = threading.Thread(target=run_analysis, daemon=True)
        t.start()
    else:
        running = False
        btn.config(text="▶  Start Live Detection", bg='#7eb8f7')
        status_var.set("Stopped. Press Start to begin again.")

btn = tk.Button(root, text="▶  Start Live Detection",
                font=('Helvetica', 13, 'bold'),
                bg='#7eb8f7', fg='#1e1e2e',
                activebackground='#5090d0',
                relief='flat', cursor='hand2',
                command=toggle, padx=20, pady=8)
btn.pack(pady=10)

tk.Label(root, text="NeuroScreen: AI-Based Emotion Detection System for Smart Screen Time Control Using EEG, ECG/GSR and Facial Recognition",
         font=('Helvetica', 8), bg='#1e1e2e', fg='#555577').pack(pady=(0,6))

root.mainloop()
cap.release()
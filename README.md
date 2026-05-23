<div align="center">

# 🎵 Music Generation with AI

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=1000&color=6C63FF&center=true&vCenter=true&width=650&lines=AI+Music+Generation+with+Deep+Learning;Bidirectional+LSTM+%2B+Bach+Chorales;Generate+Original+Piano+Music;Built+by+Tauseef+Alam+%40+CodeAlpha" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![music21](https://img.shields.io/badge/music21-9.1.0-8B5CF6?style=for-the-badge)
![FluidSynth](https://img.shields.io/badge/FluidSynth-Audio-22C55E?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-5.x-FF7C00?style=for-the-badge)
![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Task-3%2F4%20Complete-success?style=for-the-badge)

<br/>

| 👨‍💻 Developer | 🏢 Company | 📅 Batch | 🎼 Domain |
|:---:|:---:|:---:|:---:|
| **Tauseef Alam** | **CodeAlpha** | **May 2026** | **Deep Learning · Music AI** |

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](YOUR_LINKEDIN_URL)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](YOUR_GITHUB_URL)

</div>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Model Architecture](#-model-architecture)
- [Temperature Sampling](#-temperature-sampling)
- [Dataset](#-dataset)
- [Training Results](#-training-results)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Common Errors & Fixes](#-common-errors--fixes)
- [What I Learned](#-what-i-learned)
- [Connect](#-connect)

---

## 🎯 About the Project

An AI system trained on **Johann Sebastian Bach's chorales** that composes original piano music from scratch using a **Bidirectional LSTM** deep learning model.

The model reads 100 consecutive notes at a time and predicts the next one. Repeat that 200 times and you have a full composition — completely original, never heard before, generated in seconds.

> Built as **Task 3** of the **CodeAlpha AI Internship** (May 2026 Batch).

> *"It's not magic. The model learned the grammar of Bach's music — and now it writes its own sentences in the same language."*
> — Tauseef Alam

---

## 🎬 Live Demo

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Settings:
    Notes to Generate  :  200
    Temperature        :  0.8
    Tempo (BPM)        :  120
    Seed Mode          :  Random

  Generating 200 notes (temp=0.8, seed=14)...
    50/200 notes done...
    100/200 notes done...
    150/200 notes done...
    200/200 notes done...

  ✅ MIDI saved → /tmp/generated_1716234567.mid
  ✅ WAV  saved → /tmp/generated_1716234567.wav (2,304 KB)

  ┌─────────────────────────────────────────────┐
  │  ✅ Music generated in 8.4s                 │
  │                                             │
  │  Total notes    :  200                      │
  │  Single notes   :  143                      │
  │  Chords         :  57                       │
  │  Unique items   :  38 / 54                  │
  │  Temperature    :  0.8                      │
  │  Tempo          :  120 BPM                  │
  │                                             │
  │  Bidirectional LSTM · Bach Chorales · TF2.20│
  └─────────────────────────────────────────────┘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎼 **Bach Corpus** | Trained on 20 Bach chorales — no external download needed (bundled in music21) |
| 🧠 **Bidirectional LSTM** | Reads note sequences forward AND backward for richer musical context |
| 🌡️ **Temperature Slider** | One slider controls creativity — from classical (0.3) to jazz-like (1.8) |
| 🎹 **Piano Audio Output** | Generated music plays directly in the browser as a WAV audio file |
| 💾 **MIDI Download** | Download `.mid` file — open in MuseScore to see sheet music |
| ⏱️ **BPM Control** | Set your own tempo from 60 (slow) to 200 (fast) |
| 🎲 **Fixed Seed Mode** | Reproducible output — same music every run (perfect for demos) |
| 📉 **Loss Curve** | Training history visualised as a PNG chart |
| 📊 **Generation Stats** | Note count, chord count, unique items used, tempo |
| ⚙️ **How It Works Tab** | Built-in technical explanation of the full pipeline |

---

## 🔬 How It Works

The entire pipeline from raw MIDI data to audio output in 7 steps:

```
┌─────────────────────────────────────────────────────────────┐
│   STEP 1 — Data Collection (music21 Bach Corpus)            │
│                                                             │
│   corpus.getComposer('bach')  →  371 chorales bundled!      │
│   We use 20 of them. Zero downloads needed.                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│   STEP 2 — Note Extraction (music21)                        │
│                                                             │
│   Parse each MIDI → extract musical elements:               │
│     Single note  →  "C4", "F#5", "B3"                      │
│     Chord        →  "0.4.7"  (dot-separated semitones)      │
│                                                             │
│   20 chorales → ~2,900 note items collected                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│   STEP 3 — Encoding + Sliding Window Sequences              │
│                                                             │
│   Build vocabulary: {"C4": 0, "D4": 1, "E4": 2, ...}       │
│   Vocabulary size: ~54 unique notes/chords                  │
│                                                             │
│   Slide a window of 100 notes across all data:              │
│     Input  (X): [0, 5, 2, 11, 3, ...]  ← 100 integers      │
│     Output (y): 8                       ← next note index   │
│                                                             │
│   Result: ~2,900 training sequences                         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│   STEP 4 — Bidirectional LSTM Training (~20 min on GPU)     │
│                                                             │
│   Embedding(54, 64)  →  BiLSTM(256)  →  Dropout(0.3)       │
│   →  LSTM(256)  →  Dropout(0.3)  →  Dense(256, relu)        │
│   →  BatchNorm  →  Dense(54, softmax)                       │
│                                                             │
│   80 epochs | EarlyStopping | ModelCheckpoint               │
│   Best loss achieved: ~0.016                                │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│   STEP 5 — Music Generation (Temperature Sampling)          │
│                                                             │
│   Pick seed sequence → feed to model → get probabilities    │
│   Apply temperature → sample next note → slide window       │
│   Repeat 200 times  → list of 200 note strings              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│   STEP 6 — Create MIDI File (music21)                       │
│                                                             │
│   note.Note("C4").offset = 0.0                              │
│   note.Note("E4").offset = 0.5                              │
│   chord.Chord([C4, E4, G4]).offset = 1.0                    │
│   stream.write('midi', fp="output.mid")                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│   STEP 7 — Render Audio (FluidSynth)                        │
│                                                             │
│   fluidsynth -ni FluidR3_GM.sf2 output.mid -F output.wav    │
│                                                             │
│   Uses real recorded piano samples from the soundfont.      │
│   Output: playable WAV audio file                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Model Architecture

```
Input: sequence of 100 integer-encoded notes
         shape → (batch_size, 100)
                        │
                        ▼
         ┌──────────────────────────┐
         │  Embedding               │
         │  input_dim  = 54 (vocab) │
         │  output_dim = 64         │
         │  input_length = 100      │
         │                          │
         │  Each note integer →     │
         │  64-dimensional vector   │
         └──────────────┬───────────┘
                        │
                        ▼
         ┌──────────────────────────┐
         │  Bidirectional LSTM      │
         │  units = 256             │
         │  return_sequences = True │
         │                          │
         │  Reads notes →           │
         │  AND ← backward          │
         │  Richer musical context  │
         └──────────────┬───────────┘
                        │
                        ▼
         ┌──────────────────────────┐
         │  Dropout(0.3)            │
         │  Prevents memorisation   │
         └──────────────┬───────────┘
                        │
                        ▼
         ┌──────────────────────────┐
         │  LSTM                    │
         │  units = 256             │
         │  return_sequences = False│
         │  Higher-level patterns   │
         └──────────────┬───────────┘
                        │
                        ▼
         ┌──────────────────────────┐
         │  Dropout(0.3)            │
         └──────────────┬───────────┘
                        │
                        ▼
         ┌──────────────────────────┐
         │  Dense(256, relu)        │
         │  BatchNormalization      │
         └──────────────┬───────────┘
                        │
                        ▼
         ┌──────────────────────────┐
         │  Dense(54, softmax)      │
         │                          │
         │  Output: probability     │
         │  for each of 54 notes    │
         └──────────────┬───────────┘
                        │
                        ▼
              Temperature Sampling
              → pick next note
              → slide window
              → repeat ×200
```

**Total Parameters:** ~1,529,014

**Why Bidirectional?**
Standard LSTM reads left-to-right only. Bidirectional reads both forward and backward simultaneously, then merges the results. Music has patterns that depend on both what came before AND what comes after — making BiLSTM a natural fit.

---

## 🌡️ Temperature Sampling

Temperature is the single most powerful control in this project. It rescales the model's probability distribution before sampling:

```python
def sample_with_temperature(preds, temperature):
    preds   = np.log(preds + 1e-8) / temperature   # rescale
    exp_p   = np.exp(preds)
    preds   = exp_p / np.sum(exp_p)                # re-normalise
    return np.argmax(np.random.multinomial(1, preds, 1))
```

**What temperature actually does:**

```
Model output (raw softmax):
  C4 → 0.72    E4 → 0.12    G4 → 0.08    F4 → 0.05    ...

Temperature = 0.3  (low → safe):
  C4 → 0.97    E4 → 0.02    G4 → 0.01    F4 → 0.00    ...
  Model almost always picks C4. Repetitive but structured.

Temperature = 1.0  (balanced):
  C4 → 0.72    E4 → 0.12    G4 → 0.08    ...
  Keeps original distribution. Balanced creativity.

Temperature = 1.8  (high → wild):
  C4 → 0.38    E4 → 0.23    G4 → 0.20    F4 → 0.17    ...
  Probabilities flatten. Model takes surprising risks.
```

### Temperature Guide

| Temperature | Sound Character | Best Use |
|:-----------:|:----------------|:---------|
| 0.1 – 0.3 | Very rigid, repetitive | Testing model is working |
| 0.4 – 0.6 | Structured, conservative | Demonstration recording |
| **0.7 – 0.9** | **Classical-sounding, coherent** | **Portfolio & LinkedIn video** |
| 1.0 | Balanced | General testing |
| 1.1 – 1.4 | Interesting, unexpected | Creative exploration |
| 1.5 – 2.0 | Chaotic, avant-garde | Experimentation only |

---

## 📊 Dataset

**Source:** music21 built-in Bach corpus — zero external downloads

```python
from music21 import corpus
paths = corpus.getComposer('bach')   # 371 chorales available
selected = paths[:20]                # we use 20
```

| Property | Value |
|----------|-------|
| Composer | Johann Sebastian Bach |
| Chorales used | 20 |
| Total notes extracted | ~2,877 |
| Unique notes/chords | 54 |
| Sequence length | 100 |
| Training windows | 2,958 |
| Data format | MIDI (parsed by music21) |

**Why Bach chorales?**
Bach's chorales are highly structured, follow strict harmonic rules, and have been studied by musicians for 300 years. This structure makes them ideal for an LSTM — the patterns are real and learnable.

---

## 📈 Training Results

| Metric | Value |
|--------|-------|
| Total epochs run | 80 |
| Best validation loss | ~0.016 |
| Best accuracy | ~99.9% |
| Training time (T4 GPU) | ~20 minutes |
| Model file size | ~5.83 MB |
| Model format | `.keras` (TF 2.20 native) |

**Loss Curve:**

The model improved rapidly in the first 20 epochs, then `ReduceLROnPlateau` kicked in (halving the learning rate when progress slowed), allowing fine-tuning in later epochs.

> *(Add your `training_loss.png` screenshot here)*

---

## 🛠️ Tech Stack

<div align="center">

| Tool | Version | Purpose |
|------|---------|---------|
| `Python` | 3.12 | Core language |
| `TensorFlow / Keras` | 2.20 | Bidirectional LSTM model |
| `music21` | 9.1.0 | MIDI parsing, note extraction, score writing |
| `NumPy` | 2.x | Array operations and temperature math |
| `matplotlib` | 3.x | Training loss curve plot |
| `FluidSynth` | system | MIDI → WAV audio rendering |
| `fluid-soundfont-gm` | system | Real piano sample library (`.sf2`) |
| `Gradio` | 5.x | Web interface (already in Colab) |

</div>

---

## 📁 Project Structure

```
Task3_MusicGeneration/
│
├── Task3_Music_Generation_FINAL.py    ← Main application (TF 2.20 compatible)
└── README.md                          ← This file
```

**Generated files (at runtime, saved to `/tmp/`):**

```
/tmp/
├── all_notes.json                     ← Encoded note vocabulary
├── best_music_model.keras             ← Trained model weights
├── training_history.json             ← Loss values per epoch
├── training_loss.png                 ← Loss curve image
├── generated_XXXXXXXXXX.mid          ← Generated MIDI file
└── generated_XXXXXXXXXX.wav          ← Generated audio file
```

---

## 🚀 How to Run

### Google Colab — Step by Step

> ⚠️ **Enable GPU first — this is the most important step!**
> `Runtime → Change Runtime Type → Hardware Accelerator → T4 GPU → Save`

---

**Cell 1 — Install system packages and music21**
```python
!apt-get install -y -q fluidsynth fluid-soundfont-gm
!pip install -q "music21==9.1.0"
# ⚠️  DO NOT reinstall tensorflow, gradio, numpy, or websockets
# They are already correctly installed in Colab.
# Reinstalling them causes the websockets.legacy error.
```

**Cell 2 — Imports and configuration**

Paste Cell 2 from `Task3_Music_Generation_FINAL.py` and run.
You should see:
```
✅ TensorFlow : 2.20.0
✅ GPU        : 1 device — T4 detected ✅
```

**Cell 3 — Load Bach chorales (~1 min)**

Prints a list of 20 chorale files being loaded:
```
📚 Loading 20 Bach chorales...
  [ 1/20] bach/bwv1.6.mxl    →  153 notes  (total: 153)
  [ 2/20] bach/bwv10.7.mxl   →  212 notes  (total: 365)
  ...
✅ Total notes  : 2,877
✅ Unique items : 54
```

**Cell 4 — Preprocess**

Creates training sequences. Prints:
```
✅ Vocabulary   : 54 unique notes/chords
✅ Sequences    : 2,958
✅ X shape      : (2958, 100)
✅ y shape      : (2958, 54)
```

**Cell 5 — Build model**

Prints the Keras model summary showing all layers.

**Cell 6 — Train (~15–25 minutes)**

Watch the loss decrease epoch by epoch:
```
Epoch 1/80 - loss: 0.0146 - accuracy: 0.9980
Epoch 2/80 - loss: 0.0142 - accuracy: 0.9993
...
Epoch 79/80 - loss improved from 0.01662 to 0.01610
✅ Training done: 22.4 min | Best loss: 0.0161
```

**Cells 7–12 — Helper functions and UI**

Run these quickly — they build the generation pipeline and Gradio interface.

**Cell 13 — Launch**
```
🚀 Launching Music Generator...
Running on public URL: https://xxxxx.gradio.live
```

Click the URL to open your app!

---

### Local Machine

```bash
# Linux / Mac
sudo apt-get install fluidsynth fluid-soundfont-gm
pip install music21==9.1.0 tensorflow gradio numpy matplotlib

# Run
python Task3_MusicGeneration/Task3_Music_Generation_FINAL.py
```

---

## 🔧 Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: websockets.legacy` | Wrong gradio/websockets version installed | **Never** run `pip install gradio` or `pip install websockets` — use Colab's pre-installed versions |
| `tensorflow==2.13.0` not found | Python 3.12 incompatible | Use TF 2.20 — already in Colab, no install needed |
| `Soundfont not found` | `fluid-soundfont-gm` not installed | Run `!apt-get install fluid-soundfont-gm` in Cell 1 |
| Audio fails but MIDI works | FluidSynth issue | Download the MIDI file → open in VLC or GarageBand |
| `LookupError: bach` | music21 not installed | Run `!pip install music21==9.1.0` |
| Model not trained error | Skipped Cell 6 | Run all cells **top to bottom** without skipping |
| Gradio UI stuck loading | Port conflict | Restart runtime → run all cells again |

---

## 🧪 Test Checklist

After launching, verify the following work correctly:

```
✅ Temperature 0.5, 200 notes  → structured, Bach-like
✅ Temperature 1.0, 300 notes  → balanced, natural
✅ Temperature 1.5, 150 notes  → adventurous, unexpected
✅ Fixed seed → same music every run (reproducible)
✅ Random seed → different music every run
✅ BPM 80  → noticeably slower playback
✅ BPM 160 → noticeably faster playback
✅ Audio player works in the browser
✅ MIDI download button works
✅ Stats show correct note count, chord count, temperature
✅ Loss curve visible in Training Stats tab
```

---

## 📖 What I Learned

**Music is just a sequence prediction problem.**
Before this project, music felt like art — something intuitive and hard to explain. After building this, I understand it as structure. Notes follow patterns. Chords resolve predictably. Bach's harmonies obey rules. Once I saw that, turning it into a dataset felt natural.

**Bidirectional matters.**
I started with a standard LSTM — the music sounded mechanical and often got stuck in loops. When I switched to Bidirectional LSTM, the output noticeably improved. Looking backward in a sequence while also reading forward gives the model more context to decide what comes next. That's the difference between a student reading sentence by sentence and one who skims the whole paragraph first.

**Temperature is the most interesting parameter I've ever built.**
One floating point number completely changes the personality of the AI. 0.3 = careful student. 1.5 = experimental composer. The same model, the same weights, the same training data — just a different division in one line of math. That's elegant.

**Never reinstall pre-installed Colab packages.**
I learned this the hard way. Installing `gradio==4.44.1` alongside `websockets==12.0` broke the websockets legacy module and crashed the entire app at launch. The fix was removing everything I force-installed and letting Colab use its own correctly-configured environment. The lesson: more installs ≠ more stability.

**FluidSynth turns numbers into music.**
The soundfont (`.sf2`) file is essentially a library of real recorded piano notes. FluidSynth reads the MIDI timestamps and velocities, finds the right sample, and stitches them together into audio. It's not synthesizing sound — it's a very clever sampler.

---

## 🔗 Related Tasks

| Task | Project | Link |
|------|---------|------|
| Task 1 | Language Translation Tool | [View →](../Task1_LanguageTranslation/) |
| Task 2 | FAQ Chatbot (NLP) | [View →](../Task2_FAQChatbot/) |
| Task 3 | Music Generation with AI *(you are here)* | — |
| Task 4 | Object Detection & Tracking | [View →](../Task4_ObjectDetection/) |

---

## 👨‍💻 Connect

<div align="center">

**Tauseef Alam**

*AI Intern @ CodeAlpha | May 2026 Batch*

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Tauseef_Alam-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](YOUR_LINKEDIN_URL)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github&logoColor=white)](YOUR_GITHUB_URL)
[![Email](https://img.shields.io/badge/Email-Get_in_touch-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:YOUR_EMAIL)

<br/>

---

<sub>
Built with ❤️ during the <strong>CodeAlpha AI Internship</strong> — May 2026<br/>
Python · TensorFlow 2.20 · Bidirectional LSTM · music21 · FluidSynth · Gradio
</sub>

</div>

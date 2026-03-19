# 🚨 Video Annotation for Chain Snatching Detection

## 📌 Overview

This project focuses on **automated video annotation** using deep learning techniques.
It detects **chain snatching events** in videos and annotates each frame with labels and probabilities.

The system processes multiple videos, performs frame-by-frame analysis, and generates structured outputs such as annotated videos and CSV files.

---

## 🎯 Objectives

* Detect chain snatching activities from video data
* Perform **frame-level annotation**
* Identify **temporal events** using consecutive frame logic
* Generate annotation datasets for further training

---

## 🚀 Features

* ✅ Multi-video batch processing
* ✅ Frame-level annotation (SNATCH / NOT-SNATCH)
* ✅ Temporal smoothing using sliding window
* ✅ Event detection (based on consecutive frames)
* ✅ Annotated output video generation
* ✅ CSV logging (summary + frame-level)
* ✅ Probability-based classification
* ✅ Real-time visualization

---

## 🧠 Model Architecture

* Custom **CNN (Convolutional Neural Network)**
* Built using **PyTorch**
* Binary classification:

  * `SNATCH`
  * `NOT-SNATCH`

### Architecture Details:

* Conv2D → ReLU → MaxPool
* Conv2D → ReLU → MaxPool
* Fully Connected Layers
* Sigmoid Activation (Probability Output)

---

## ⚙️ How It Works

1. Input video is read frame-by-frame
2. Each frame is resized and converted into tensor
3. CNN model predicts probability
4. Temporal smoothing is applied
5. If probability exceeds threshold → SNATCH
6. Consecutive frames confirm event detection
7. Results are:

   * Displayed on video
   * Saved in CSV
   * Written into output video

---

## 📂 Project Structure

```
Video_Annotations/
│
├── models/
├── videos_test/
├── outputs/
├── alerts/
│
├── test.py
├── train.py
├── video_to_frames.py
├── results.csv
├── frame_annotations.csv
├── README.md
```

---

## 📊 Output Files

### 1. Summary CSV (`results.csv`)

Contains:

* Video name
* Total frames
* Snatch frames
* Final classification
* Start frame

### 2. Frame-level CSV (`frame_annotations.csv`)

Contains:

* Frame number
* Label
* Probability

---

## 🎥 Output Example

* Annotated videos generated with labels
* Frame-by-frame classification overlay
* Highlighted snatch events
* Real-time detection display

---

## 📊 Sample Result

* SNATCH detected in multiple videos
* NOT-SNATCH classified correctly
* Event start frame identified accurately

---

---

## ▶️ Run the Project

```bash
python test.py
```

---


## ⚠️ Note

* Model file (`.pth`) and videos are not uploaded due to size limitations
* Please place them in:

  * `models/`
  * `videos_test/`

---



## ⭐ Acknowledgement

This project is developed as part of a **deep learning-based video analysis system** for real-world crime detection scenarios.

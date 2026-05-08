# 🥔 Potato Blight Detection — MobileNetV2 Transfer Learning

An automated, image-based diagnostic tool that identifies potato leaf diseases from a single photo using deep learning.

---

## 📋 Problem Statement

Potato crops are highly susceptible to two devastating diseases:

- **Early Blight** (*Alternaria solani*) — causes dark lesions with concentric rings
- **Late Blight** (*Phytophthora infestans*) — the pathogen behind the Great Irish Famine, still causing billions in crop losses annually

Because symptoms can initially resemble nutrient deficiencies or each other, farmers often misdiagnose the issue. This project provides an automated diagnostic tool to identify these diseases from a smartphone photo.

---

## 🎯 Classes

| Class | Description |
|---|---|
| `Potato___Early_blight` | Infected with *Alternaria solani* |
| `Potato___Late_blight` | Infected with *Phytophthora infestans* |
| `Potato___healthy` | No disease detected |

---

## 🏗️ Architecture

- **Base model:** MobileNetV2 (alpha=1.0, pretrained on ImageNet)
- **Input size:** 224 × 224 × 3
- **Total params:** ~2.26M (8.63 MB)
- **Training strategy:** 2-phase transfer learning
  - Phase 1: Freeze base, train classification head only
  - Phase 2: Unfreeze last 30 layers for fine-tuning

```
Input (224×224×3)
    → Augmentation (RandomFlip, RandomRotation, RandomZoom)
    → MobileNetV2 (pretrained backbone)
    → GlobalAveragePooling2D
    → Dropout(0.3)
    → Dense(3, softmax)
```

---

## 📊 Results

| Metric | Score |
|---|---|
| Validation Accuracy | **98%** |
| Macro F1-Score | **0.98** |
| Early Blight F1 | 0.99 |
| Late Blight F1 | 0.99 |
| Healthy F1 | 0.95 |

---

## 🗂️ Dataset

Uses a subset of the [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village):

- ~2,152 images across 3 classes
- 80/20 train/validation split

> **Note:** The dataset is not included in this repository. Download it from Kaggle and place the `Potato___Early_blight`, `Potato___Late_blight`, and `Potato___healthy` folders inside a `PlantVillage/` directory.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/t3ch-N/Potato-Blight-Detection-MobileNetV2-Transfer-Learning.git
cd Potato-Blight-Detection-MobileNetV2-Transfer-Learning
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
Open and run `potatoblight.ipynb` top to bottom. This will:
- Copy the 3 potato classes into a clean `potato_data/` folder
- Train MobileNetV2 in 2 phases
- Save `potato_blight_model.keras` and `class_names.npy`

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 🖥️ Streamlit App

Upload a photo of a potato leaf and the app returns:

- **Predicted class** (Healthy / Early Blight / Late Blight)
- **Confidence score**
- **Grad-CAM heatmap** — highlights the exact lesions the model focused on
- **Actionable recommendations** (e.g. *Apply fungicide*, *Remove affected leaves*)

![App Screenshot](https://placehold.co/800x400?text=Upload+a+leaf+→+Get+diagnosis)

---

## 🔍 Explainability — Grad-CAM

The app generates a **Gradient-weighted Class Activation Map (Grad-CAM)** overlay on the uploaded image, proving to the farmer that the AI is actually looking at the disease symptoms rather than background noise.

---

## 📦 Dependencies

```
tensorflow>=2.13
streamlit>=1.35
numpy
opencv-python-headless
Pillow
scikit-learn
matplotlib
```

---

## 👥 Stakeholders

- **Small-scale Potato Farmers** — immediate, actionable field diagnosis
- **Agricultural Officers** — monitor disease outbreaks across regions
- **NGOs & Cooperatives** — support food security initiatives

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

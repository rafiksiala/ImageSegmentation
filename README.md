# **ImageSegmentation**

## **Overview**

This project provides a fully functional **Image Segmentation API** built with **FastAPI**, capable of generating semantic segmentation masks for test images using two available models:

* **DilatedNet** (Keras model downloaded at runtime)
* **Mask2Former** (Hugging Face model extracted from a ZIP archive)

The API loads images from the repository, applies segmentation, remaps the predictions into **8 Cityscapes-derived superclasses**, and returns a **visual overlay mask** blended with the original RGB image.

The repository includes real images and masks under `data/images` and `data/masks`.

---

## **Related Dashboard Repository**

This API is complemented by a second public repository:

**ImageSegmentationDashboard**
**Purpose:** A user-friendly dashboard interface designed to interact with this API, visualize predictions, switch models, inspect overlays, and browse test images.

The **API repository (this repo)** handles loading models, computing segmentation, and generating masks.
The **Dashboard repository** provides the visual front-end for exploration.

---

## **Repository Structure**

```
ImageSegmentation/
│
├── main.py                     # FastAPI application and segmentation logic
├── requirements.txt            # Python dependencies
├── startup.sh                  # Script to start the API server
├── favicon.ico                 # Icon used by the frontend
│
├── data/
│   ├── images/                 # Input images (PNG test images)
│   └── masks/                  # Real corresponding mask files
│
├── models/                     # Populated automatically at runtime
│   ├── dilatednet.keras
│   └── mask2former/
│       ├── config.json
│       ├── model.safetensors
│       └── ...
│
└── .github/                    # Optional CI/CD workflows
```

---

## **Key Features**

### ✔️ FastAPI-based segmentation service

### ✔️ Two segmentation models:

* **DilatedNet**
* **Mask2Former**

### ✔️ Automatic download of missing models

### ✔️ 8-class segmentation remapped from Cityscapes

### ✔️ Visual mask overlay generation

### ✔️ Endpoints to browse images, retrieve masks, and run predictions

---

## **Installation**

### 1. Clone the project

```bash
git clone https://github.com/rafiksiala/ImageSegmentation.git
cd ImageSegmentation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the API

Using the startup script:

```bash
bash startup.sh
```

Or manually:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## **Segmentation Models**

### **DilatedNet**

Downloaded automatically from Azure Blob Storage and compiled with custom loss functions:

* Dice coefficient
* Dice loss
* Categorical cross entropy

### **Mask2Former**

Extracted automatically from a ZIP archive and used via:

* `Mask2FormerForUniversalSegmentation`
* `Mask2FormerImageProcessor`

---

## **Data Folders**

### `data/images/`

Contains sample images used for testing and for `/predict_from_id`.

### `data/masks/`

Contains ground-truth masks used for `/get_mask/{image_id}`.

---

## **Available API Endpoints**

### **GET /**

Returns a welcome message.

### **GET /list_images/**

Lists all available image IDs.

### **GET /list_images/{image_id}**

Returns the raw RGB image.

### **GET /get_mask/{image_id}**

Returns a mask overlay generated from the ground-truth mask.

### **GET /legend_data/**

Returns the list of 8 segmentation classes + their colors.

### **GET /available_models/**

```json
{
  "models": ["dilatednet", "mask2former"]
}
```

### **POST /predict_from_file/**

Run segmentation on an uploaded image.

### **POST /predict_from_id/**

Run segmentation on an image stored in `data/images`.

---

## **Segmentation Pipeline (Actual Behavior)**

1. Load image (from upload or from ID)
2. Preprocess according to model
3. Model inference
4. Convert logits to class predictions
5. Remap classes → 8 superclasses
6. Apply color palette
7. Blend with original image
8. Return PNG overlay


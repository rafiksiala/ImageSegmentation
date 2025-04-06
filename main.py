from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import numpy as np
import requests
import os
import io

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras import backend as K


# ------------------------------------------------------------------------------
# Mapping des 34 classes → 8 grandes catégories
cats = {
    'void': [0, 1, 2, 3, 4, 5, 6],
    'flat': [7, 8, 9, 10],
    'construction': [11, 12, 13, 14, 15, 16],
    'object': [17, 18, 19, 20],
    'nature': [21, 22],
    'sky': [23],
    'human': [24, 25],
    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1],
}
cat_mapping = {i: idx for idx, (_, ids) in enumerate(cats.items()) for i in ids}

cityscapes_colors = [
    (0, 0, 0),          # void
    (180, 180, 180),    # flat
    (200, 100, 100),    # construction
    (100, 40, 40),      # object
    (107, 142, 35),     # nature
    (70, 130, 180),     # sky
    (220, 20, 60),      # human
    (0, 0, 142),        # vehicle
]
class_names = ['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']
cmap = ListedColormap(np.array(cityscapes_colors) / 255.0)

# ------------------------------------------------------------------------------

app = FastAPI()

DATA_DIR = "data/images"
MASK_DIR = "data/masks"
img_height, img_width, n_classes = 256, 256, len(class_names)

# ------------------------------------------------------------------------------

@app.get("/")
def home():
    return {"message": "Image Segmentation API"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")

# Endpoint pour lister toutes les images disponibles dans DATA_DIR
@app.get("/list_images/")
async def list_available_images():
    print("📂 Lecture des images dans :", DATA_DIR)

    # Liste tous les fichiers dans le dossier de données
    files = os.listdir(DATA_DIR)
    print("Fichiers trouvés :", files)

    # Récupère uniquement les identifiants d’images (sans extension)
    ids = [f.split(".")[0] for f in files if f.endswith(".png")]

    # Retourne la liste des IDs sous forme JSON
    return JSONResponse(content={"ids": ids})


# Endpoint pour récupérer une image par son ID
@app.get("/list_images/{image_id}")
async def fetch_image_by_id(image_id: str):
    # Construit le chemin vers l'image PNG correspondante
    image_path = os.path.join(DATA_DIR, f"{image_id}.png")

    # Vérifie que l'image existe
    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"error": "Image not found"})

    # Retourne l'image en tant que fichier PNG
    return FileResponse(image_path, media_type="image/png")


# Fonction pour charger, redimensionner et encoder une image de masque en format one-hot
def load_and_encode_mask(mask_path):
    # Chargement du masque en niveaux de gris
    mask = image.img_to_array(image.load_img(mask_path, color_mode='grayscale'))

    # Redimensionnement à la taille cible (img_height, img_width)
    mask = tf.image.resize(mask, (img_height, img_width), method='nearest')

    # Conversion en entier (uint8) pour traitement par numpy
    mask = tf.cast(mask, tf.uint8).numpy()

    # Fonction interne pour encoder le masque en one-hot
    def encode_mask(mask_img):
        mask_img = np.squeeze(mask_img).astype(np.uint8)  # Suppression de la dimension canal
        new_mask = np.zeros((img_height, img_width, n_classes), dtype=np.float32)
        for label, class_idx in cat_mapping.items():  # Mapping label -> indice classe
            new_mask[mask_img == label, class_idx] = 1.0
        return new_mask

    encoded_mask = encode_mask(mask)

    # Retourne la version "argmax" (classe dominante par pixel)
    return np.argmax(encoded_mask, axis=-1)

# Endpoint FastAPI pour récupérer une image du masque colorisée
@app.get("/get_mask/{image_id}")
async def get_mask(image_id: str):
    mask_path = os.path.join(MASK_DIR, f"{image_id}.png")

    # Vérifie si le fichier de masque existe
    if not os.path.exists(mask_path):
        return JSONResponse(status_code=404, content={"error": "Mask not found"})

    # Chargement et traitement du masque
    mask = load_and_encode_mask(mask_path)

    # Affichage du masque avec une colormap
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=n_classes - 1)
    ax.axis("off")

    # Enregistrement de l'image dans un buffer mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Retour de l'image en réponse HTTP
    return Response(content=buf.getvalue(), media_type="image/png")

# ------------------------------------------------------------------------------

# Dice coefficient and loss
def dice_coeff(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)

def total_loss(y_true, y_pred):
    ce = CategoricalCrossentropy()(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return ce + 3 * dl

# ------------------------------------------------------------------------------

MODEL_URL = "https://imgsegmodelstorage.blob.core.windows.net/models/unet_model.keras"
MODEL_PATH = "models/unet_model.keras"

def download_model_if_needed():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle depuis Azure Blob Storage...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Modèle téléchargé avec succès !")
    else:
        print("Modèle déjà présent.")

# Télécharger le modèle si nécessaire
download_model_if_needed()

# Charger le modèle
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer=Adam(1e-4), loss=total_loss, metrics=[dice_coeff, 'accuracy'])

# ------------------------------------------------------------------------------

def preprocess_image_from_file(file: UploadFile):
    contents = file.file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    pil_img = pil_img.resize((img_width, img_height))
    img_array = image.img_to_array(pil_img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_image_from_path(path: str):
    pil_img = Image.open(path).convert("RGB")
    pil_img = pil_img.resize((img_width, img_height))
    img_array = image.img_to_array(pil_img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_mask(img_array):
    prediction = model.predict(img_array)
    reshaped = prediction[0].reshape((img_height, img_width, n_classes))
    predicted_mask = np.argmax(reshaped, axis=-1)
    return predicted_mask

# ------------------------------------------------------------------------------

@app.post("/predict_from_file/")
async def predict_from_file(file: UploadFile = File(...)):
    print(f"Image reçue : {file.filename} — type : {file.content_type}")
    preprocessed_img = preprocess_image_from_file(file)
    predicted_mask = predict_mask(preprocessed_img)
    return JSONResponse(content={"mask": predicted_mask.tolist()})

class PredictRequest(BaseModel):
    image_id: str

@app.post("/predict_from_id/")
async def predict_from_id(req: PredictRequest):
    image_path = os.path.join(DATA_DIR, f"{req.image_id}.png")
    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"error": "Image not found"})

    preprocessed_img = preprocess_image_from_path(image_path)
    predicted_mask = predict_mask(preprocessed_img)
    return JSONResponse(content={"mask": predicted_mask.tolist()})

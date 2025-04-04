#!/bin/bash

# Lancer FastAPI avec Gunicorn sur le port 8000 (accessible publiquement)
uvicorn main:app --host 0.0.0.0 --port 8000

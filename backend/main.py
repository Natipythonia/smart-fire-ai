import base64
from functools import lru_cache
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.smoke_detector import SmokeDetector, MAX_ARCHIVO


app = FastAPI(
    title="SMART FIRE AI",
    description="Sistema experimental de deteccion de incendios forestales de ATLANTIA TECH",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

inference_lock = Lock()


@lru_cache(maxsize=1)
def obtener_detector():
    return SmokeDetector()


@app.get("/")
def inicio():
    return {
        "proyecto": "SMART FIRE AI",
        "empresa": "ATLANTIA TECH",
        "version": "MVP 1.0",
        "estado": "Backend funcionando",
        "ambito": "España",
    }


@app.get("/health")
def comprobar_estado():
    return {
        "status": "ok",
        "servicio": "smart-fire-ai-backend",
    }


@app.post("/detect/smoke")
def detectar_humo(
    archivo: UploadFile = File(...)
):
    if archivo.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=415,
            detail="Solo se admiten imagenes JPEG, PNG o WebP.",
        )

    contenido = archivo.file.read(MAX_ARCHIVO + 1)

    if not contenido:
        raise HTTPException(
            status_code=400,
            detail="El archivo esta vacio.",
        )

    if len(contenido) > MAX_ARCHIVO:
        raise HTTPException(
            status_code=413,
            detail="La imagen supera el limite de 10 MB.",
        )

    try:
        detector = obtener_detector()

        with inference_lock:
            informacion, imagen = detector.analizar(
                contenido,
                umbral=0.20,
            )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except (RuntimeError, FileNotFoundError) as error:
        raise HTTPException(
            status_code=503,
            detail="El motor de deteccion no esta disponible.",
        ) from error

    return {
        **informacion,
        "imagen_analizada": (
            "data:image/jpeg;base64,"
            + base64.b64encode(imagen).decode("ascii")
        ),
    }

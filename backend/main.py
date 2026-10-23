
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SMART FIRE AI",
    description="Sistema de detección temprana de incendios forestales de ATLANTIA TECH",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


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

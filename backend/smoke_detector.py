from io import BytesIO
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError


BASE = Path(__file__).resolve().parent.parent
MODELO = BASE / "models" / "best.onnx"

TAMANO = 1024
UMBRAL_PREDETERMINADO = 0.20
IOU_NMS = 0.45
MAX_ARCHIVO = 10 * 1024 * 1024
MAX_PIXELES = 25_000_000


class SmokeDetector:
    def __init__(self):
        if not MODELO.is_file():
            raise FileNotFoundError(
                f"No se encuentra el modelo ONNX: {MODELO}"
            )

        self.sesion = ort.InferenceSession(
            str(MODELO),
            providers=["CPUExecutionProvider"]
        )
        self.nombre_entrada = self.sesion.get_inputs()[0].name

    def analizar(self, contenido: bytes, umbral: float = UMBRAL_PREDETERMINADO):
        if not 0 < umbral <= 1:
            raise ValueError("El umbral debe estar entre 0 y 1.")

        if not contenido or len(contenido) > MAX_ARCHIVO:
            raise ValueError("Imagen vacia o superior a 10 MB.")

        try:
            with Image.open(BytesIO(contenido)) as archivo:
                ancho_original, alto_original = archivo.size

                if ancho_original * alto_original > MAX_PIXELES:
                    raise ValueError("La imagen supera 25 millones de pixeles.")

                imagen = np.asarray(
                    ImageOps.exif_transpose(archivo).convert("RGB")
                )
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("El archivo no es una imagen valida.") from error

        alto, ancho = imagen.shape[:2]

        escala = min(TAMANO / ancho, TAMANO / alto)
        nuevo_ancho = round(ancho * escala)
        nuevo_alto = round(alto * escala)

        redimensionada = cv2.resize(
            imagen,
            (nuevo_ancho, nuevo_alto),
            interpolation=cv2.INTER_LINEAR
        )

        izquierda = (TAMANO - nuevo_ancho) // 2
        arriba = (TAMANO - nuevo_alto) // 2

        lienzo = np.full(
            (TAMANO, TAMANO, 3),
            114,
            dtype=np.uint8
        )

        lienzo[
            arriba:arriba + nuevo_alto,
            izquierda:izquierda + nuevo_ancho
        ] = redimensionada

        entrada = (
            lienzo.astype(np.float32)
            .transpose(2, 0, 1)[None] / 255.0
        )

        inicio = time.perf_counter()

        salida = self.sesion.run(
            None,
            {self.nombre_entrada: entrada}
        )[0][0].T

        tiempo = time.perf_counter() - inicio

        cajas = []
        puntuaciones = []

        for prediccion in salida:
            if not np.isfinite(prediccion).all():
                continue

            x, y, w, h, confianza = map(float, prediccion[:5])

            if confianza < umbral or w <= 0 or h <= 0:
                continue

            x1 = max(0, min(ancho, round((x - w / 2 - izquierda) / escala)))
            y1 = max(0, min(alto, round((y - h / 2 - arriba) / escala)))
            x2 = max(0, min(ancho, round((x + w / 2 - izquierda) / escala)))
            y2 = max(0, min(alto, round((y + h / 2 - arriba) / escala)))

            if x2 <= x1 or y2 <= y1:
                continue

            cajas.append([x1, y1, x2 - x1, y2 - y1])
            puntuaciones.append(confianza)

        if cajas:
            indices = np.asarray(
                cv2.dnn.NMSBoxes(
                    cajas, puntuaciones, umbral, IOU_NMS
                ),
                dtype=int
            ).reshape(-1)
        else:
            indices = np.array([], dtype=int)

        resultado = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
        detecciones = []

        for indice in indices:
            x, y, w, h = cajas[indice]
            confianza = puntuaciones[indice]

            cv2.rectangle(
                resultado,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                resultado,
                f"Posible humo {confianza:.0%}",
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            detecciones.append({
                "clase": "posible_humo",
                "puntuacion": round(confianza, 4),
                "caja": [x, y, w, h]
            })

        correcto, imagen_codificada = cv2.imencode(
            ".jpg", resultado
        )

        if not correcto:
            raise RuntimeError("No se pudo generar la imagen analizada.")

        informacion = {
            "modelo": "PyroNear YOLO11s - experimental",
            "ancho": ancho,
            "alto": alto,
            "umbral": umbral,
            "tiempo_inferencia_segundos": round(tiempo, 3),
            "total_detecciones": len(detecciones),
            "detecciones": detecciones,
            "aviso": (
                "Detecciones experimentales. "
                "No confirman incendios ni sustituyen vigilancia oficial."
            )
        }

        return informacion, imagen_codificada.tobytes()

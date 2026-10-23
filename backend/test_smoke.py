from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent.parent
IMAGEN = BASE / "data" / "test_images" / "humo_prueba.jpg"
MODELO = BASE / "models" / "best.onnx"
RESULTADO = BASE / "data" / "test_images" / "resultado_humo.jpg"

UMBRAL = 0.20
IOU = 0.01
TAMANO = 1024


def main():
    if not IMAGEN.is_file():
        print("ERROR: No se encuentra la fotografía:", IMAGEN)
        return

    if not MODELO.is_file():
        print("ERROR: No se encuentra el modelo:", MODELO)
        return

    print("SMART FIRE AI - DETECTOR EXPERIMENTAL DE HUMO")
    print("Cargando modelo...")

    sesion = ort.InferenceSession(
        str(MODELO),
        providers=["CPUExecutionProvider"]
    )

    with Image.open(IMAGEN) as original:
        imagen = ImageOps.exif_transpose(original).convert("RGB")

    ancho, alto = imagen.size
    print("Imagen original:", ancho, "x", alto)

    # Preparación de imagen según el formato del modelo.
    redimensionada = imagen.resize((TAMANO, TAMANO))
    datos = np.asarray(redimensionada, dtype=np.float32) / 255.0
    entrada = np.transpose(datos, (2, 0, 1))[None]

    print("Analizando fotografía...")
    inicio = time.perf_counter()

    salida = sesion.run(
        None,
        {sesion.get_inputs()[0].name: entrada}
    )[0]

    segundos = time.perf_counter() - inicio
    print("Tiempo de inferencia:", round(segundos, 2), "segundos")

    # Salida YOLO: [x_centro, y_centro, ancho, alto, confianza].
    predicciones = salida[0].T

    cajas = []
    puntuaciones = []

    for prediccion in predicciones:
        x, y, w, h, confianza = map(float, prediccion[:5])

        if not np.isfinite(prediccion).all():
            continue

        if confianza < UMBRAL or w <= 0 or h <= 0:
            continue

        x1 = max(0, min(ancho, int((x - w / 2) * ancho / TAMANO)))
        y1 = max(0, min(alto, int((y - h / 2) * alto / TAMANO)))
        x2 = max(0, min(ancho, int((x + w / 2) * ancho / TAMANO)))
        y2 = max(0, min(alto, int((y + h / 2) * alto / TAMANO)))

        if x2 <= x1 or y2 <= y1:
            continue

        cajas.append([x1, y1, x2 - x1, y2 - y1])
        puntuaciones.append(confianza)

    indices = cv2.dnn.NMSBoxes(cajas, puntuaciones, UMBRAL, IOU)
    indices = np.asarray(indices, dtype=int).reshape(-1)

    resultado = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)

    for indice in indices:
        x, y, w, h = cajas[indice]
        confianza = puntuaciones[indice]

        cv2.rectangle(
            resultado,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        etiqueta = f"Posible humo {confianza:.0%}"

        cv2.putText(
            resultado,
            etiqueta,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

        print("Posible detección:", round(confianza * 100, 1), "%")

    RESULTADO.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(RESULTADO), resultado):
        raise RuntimeError("No se pudo guardar la imagen de resultado")

    print("Candidatas antes de NMS:", len(cajas))
    print("Detecciones finales:", len(indices))
    print("Resultado guardado en:", RESULTADO)
    print("IMPORTANTE: Estas detecciones no confirman un incendio.")


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


BASE = Path(__file__).resolve().parent.parent
MODELO = BASE / "models" / "best.onnx"
RESULTADOS = BASE / "data" / "results"
TAMANO = 1024


def analizar_imagen(ruta_imagen: Path, umbral: float = 0.20):
    if not ruta_imagen.is_file():
        raise FileNotFoundError(
            f"No se encuentra la imagen: {ruta_imagen}"
        )

    if not MODELO.is_file():
        raise FileNotFoundError(
            f"No se encuentra el modelo: {MODELO}"
        )

    with Image.open(ruta_imagen) as archivo:
        imagen = np.asarray(
            ImageOps.exif_transpose(archivo).convert("RGB")
        )

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

    sesion = ort.InferenceSession(
        str(MODELO),
        providers=["CPUExecutionProvider"]
    )

    inicio = time.perf_counter()

    salida = sesion.run(
        None,
        {sesion.get_inputs()[0].name: entrada}
    )[0][0].T

    tiempo = time.perf_counter() - inicio

    cajas = []
    confianzas = []

    for prediccion in salida:
        if not np.isfinite(prediccion).all():
            continue

        x, y, w, h, confianza = map(
            float, prediccion[:5]
        )

        if confianza < umbral or w <= 0 or h <= 0:
            continue

        x1 = int((x - w / 2 - izquierda) / escala)
        y1 = int((y - h / 2 - arriba) / escala)
        x2 = int((x + w / 2 - izquierda) / escala)
        y2 = int((y + h / 2 - arriba) / escala)

        x1 = max(0, min(ancho, x1))
        y1 = max(0, min(alto, y1))
        x2 = max(0, min(ancho, x2))
        y2 = max(0, min(alto, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        cajas.append([
            x1, y1,
            x2 - x1,
            y2 - y1
        ])

        confianzas.append(confianza)

    if cajas:
        indices = cv2.dnn.NMSBoxes(
            cajas,
            confianzas,
            umbral,
            0.45
        )
        indices = np.asarray(
            indices, dtype=int
        ).reshape(-1)
    else:
        indices = np.array([], dtype=int)

    resultado = cv2.cvtColor(
        imagen,
        cv2.COLOR_RGB2BGR
    )

    detecciones = []

    for indice in indices:
        x, y, w, h = cajas[indice]
        confianza = confianzas[indice]

        cv2.rectangle(
            resultado,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        etiqueta = f"Posible humo {confianza:.0%}"

        cv2.putText(
            resultado,
            etiqueta,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        detecciones.append({
            "confianza": round(confianza, 4),
            "caja": [x, y, w, h]
        })

    RESULTADOS.mkdir(parents=True, exist_ok=True)

    destino = RESULTADOS / (
        ruta_imagen.stem + "_analizada.jpg"
    )

    if not cv2.imwrite(str(destino), resultado):
        raise RuntimeError(
            "No se pudo guardar la imagen analizada"
        )

    print("\nSMART FIRE AI - ANALISIS EXPERIMENTAL")
    print("Imagen:", ruta_imagen.name)
    print("Resolucion:", ancho, "x", alto)
    print("Tiempo de inferencia:", round(tiempo, 3), "s")
    print("Candidatas antes de NMS:", len(cajas))
    print("Detecciones finales:", len(detecciones))

    for numero, deteccion in enumerate(
        detecciones, start=1
    ):
        print(
            f"Deteccion {numero}: "
            f"{deteccion['confianza']:.1%}"
        )

    print("Resultado:", destino)
    print(
        "AVISO: Son detecciones experimentales, "
        "no confirmaciones de incendio."
    )

    return detecciones


def main():
    parser = argparse.ArgumentParser(
        description="Detector experimental de humo - Smart Fire AI"
    )

    parser.add_argument(
        "imagen",
        help="Ruta de la fotografía que queremos analizar"
    )

    parser.add_argument(
        "--umbral",
        type=float,
        default=0.20,
        help="Umbral de confianza entre 0 y 1"
    )

    argumentos = parser.parse_args()

    if not 0 < argumentos.umbral <= 1:
        parser.error(
            "El umbral debe ser mayor que 0 y menor o igual que 1"
        )

    analizar_imagen(
        Path(argumentos.imagen),
        argumentos.umbral
    )


if __name__ == "__main__":
    main()

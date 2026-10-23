from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

BASE = Path.cwd()
ORIGEN = BASE / "data/test_images/humo_prueba.jpg"
DESTINO = BASE / "data/test_images/resultado_letterbox.jpg"
MODELO = BASE / "models/best.onnx"

TAMANO = 1024
UMBRAL = 0.20
IOU = 0.45

with Image.open(ORIGEN) as archivo:
    imagen = np.asarray(
        ImageOps.exif_transpose(archivo).convert("RGB")
    )

alto, ancho = imagen.shape[:2]

escala = min(TAMANO / ancho, TAMANO / alto)
nuevo_ancho = round(ancho * escala)
nuevo_alto = round(alto * escala)

redimensionada = cv2.resize(
    imagen, (nuevo_ancho, nuevo_alto)
)

izquierda = (TAMANO - nuevo_ancho) // 2
arriba = (TAMANO - nuevo_alto) // 2

lienzo = np.full(
    (TAMANO, TAMANO, 3), 114, dtype=np.uint8
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

salida = sesion.run(
    None,
    {sesion.get_inputs()[0].name: entrada}
)[0][0].T

cajas = []
confianzas = []

for prediccion in salida:
    x, y, w, h, confianza = map(float, prediccion[:5])

    if not np.isfinite(prediccion).all():
        continue

    if confianza < UMBRAL or w <= 0 or h <= 0:
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

    cajas.append([x1, y1, x2 - x1, y2 - y1])
    confianzas.append(confianza)

indices = cv2.dnn.NMSBoxes(
    cajas, confianzas, UMBRAL, IOU
)
indices = np.asarray(indices, dtype=int).reshape(-1)

resultado = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

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

    cv2.putText(
        resultado,
        f"Posible humo {confianza:.0%}",
        (x, max(18, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )

cv2.imwrite(str(DESTINO), resultado)

print("SMART FIRE AI - RESULTADO LETTERBOX")
print("Candidatas:", len(cajas))
print("Detecciones despues de NMS:", len(indices))
print("Imagen guardada:", DESTINO)
print("Las detecciones requieren verificacion visual.")

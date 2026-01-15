from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response, JSONResponse
import cv2
import numpy as np
import easyocr
import re

app = FastAPI(title="OpenCV Preprocess Service")

reader = easyocr.Reader(["es"], gpu=False)


def blur_score_laplacian(gray: np.ndarray) -> float:
    """
    Mide nitidez con Varianza del Laplaciano.
    Más alto = más nítido. Más bajo = más borroso.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    maxWidth = max(maxWidth, 1)
    maxHeight = max(maxHeight, 1)

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_document_quad(img_bgr: np.ndarray):
    """
    Intenta detectar el contorno del carnet y devolver 4 puntos (quad).
    Devuelve: (found, quad_points)
    """
    h, w = img_bgr.shape[:2]
    target_w = 1200
    scale = target_w / float(w) if w > target_w else 1.0
    resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    quad = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.10 * (resized.shape[0] * resized.shape[1]):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break

    if quad is None:
        return False, None

    quad = quad / scale
    return True, quad


def preprocess_document(img_bgr: np.ndarray):
    """
    Paso 1: detectar documento y hacer warp/recorte.
    Paso 2: medir blur.
    """
    found, quad = detect_document_quad(img_bgr)

    if found and quad is not None:
        doc = four_point_transform(img_bgr, quad)
        doc_detected = True
    else:
        doc = img_bgr.copy()
        doc_detected = False

    gray = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)
    score = blur_score_laplacian(gray)

    meta = {"doc_detected": doc_detected, "blur_score": score}
    return doc, meta


def detect_mrz(doc_bgr: np.ndarray):
    """
    Detecta si hay MRZ en la imagen (pensado para reverso).
    Hace OCR SOLO en la franja inferior y busca patrones con '<' o 'I<BOL'.
    """
    h, w = doc_bgr.shape[:2]
    if h < 50 or w < 50:
        return {"mrz_found": False, "mrz_conf": 0.0, "mrz_sample": None}

    y0 = int(h * 0.60)
    roi = doc_bgr[y0:h, 0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    results = reader.readtext(
        thr,
        detail=1,
        paragraph=False,
        allowlist=allow,
    )

    best = {"mrz_found": False, "mrz_conf": 0.0, "mrz_sample": None}

    for bbox, text, conf in results:
        t = (text or "").replace(" ", "").upper()

        has_many_arrows = t.count("<") >= 6 and len(t) >= 15
        has_ibol = "I<BOL" in t or re.search(r"I<\s*BOL", t) is not None
        looks_like_line = (len(t) >= 20 and re.match(r"^[A-Z0-9<]+$", t) is not None)

        if (has_many_arrows or has_ibol) and looks_like_line:
            if float(conf) > best["mrz_conf"]:
                best = {
                    "mrz_found": True,
                    "mrz_conf": float(conf),
                    "mrz_sample": t[:60], 
                }

    return best

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/preprocess-image")
async def preprocess_image(
    file: UploadFile = File(...),
    blur_threshold: float = Query(110.0, description="Umbral: menor = más borroso (típico 80-160)"),
    annotate: bool = Query(False, description="Si true, dibuja info sobre la imagen para debug"),
    as_json: bool = Query(False, description="Si true, devuelve JSON en vez de imagen"),
    detect_type: bool = Query(True, description="Si true, intenta clasificar TypeA/TypeB por MRZ"),
):
    """
    Devuelve:
    - Imagen tratada (recortada/rectificada) por defecto
    - Meta: doc_detected, blur_score, blurry
    - Y si detect_type=true: ci_type (TypeA/TypeB), mrz_found
    """
    data = await file.read()
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"ok": False, "message": "No se pudo leer la imagen"}, status_code=400)

    processed, meta = preprocess_document(img)

    blur_score = float(meta["blur_score"])
    blurry = blur_score < float(blur_threshold)

    # Type detect por MRZ (TypeB tiene MRZ, TypeA no)
    mrz_info = {"mrz_found": False, "mrz_conf": 0.0, "mrz_sample": None}
    ci_type = "UNKNOWN"

    if detect_type and meta["doc_detected"]:
        mrz_info = detect_mrz(processed)
        ci_type = "TypeB" if mrz_info["mrz_found"] else "TypeA"
    elif detect_type and not meta["doc_detected"]:
        ci_type = "UNKNOWN"

    meta_out = {
        "ok": True,
        "doc_detected": bool(meta["doc_detected"]),
        "blur_score": blur_score,
        "blurry": bool(blurry),
        "blur_threshold": float(blur_threshold),
        "ci_type": ci_type,
        "mrz_found": bool(mrz_info["mrz_found"]),
        "mrz_conf": float(mrz_info["mrz_conf"]),
        "mrz_sample": mrz_info["mrz_sample"],  # útil para debug (puedes quitarlo si no quieres)
    }

    if annotate:
        # SOLO si tú lo pides
        txt1 = f"doc_detected={meta_out['doc_detected']} ci_type={meta_out['ci_type']}"
        txt2 = f"blur={meta_out['blur_score']:.1f} thr={meta_out['blur_threshold']:.1f} blurry={meta_out['blurry']} mrz={meta_out['mrz_found']}"
        cv2.putText(processed, txt1, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(processed, txt2, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if as_json:
        return meta_out

    ok, buf = cv2.imencode(".png", processed)
    if not ok:
        return JSONResponse({"ok": False, "message": "No se pudo codificar la imagen"}, status_code=500)

    headers = {
        "X-Doc-Detected": "1" if meta_out["doc_detected"] else "0",
        "X-Blur-Score": f"{meta_out['blur_score']:.3f}",
        "X-Blurry": "1" if meta_out["blurry"] else "0",
        "X-CI-Type": meta_out["ci_type"],
        "X-MRZ-Found": "1" if meta_out["mrz_found"] else "0",
        "X-MRZ-Conf": f"{meta_out['mrz_conf']:.3f}",
    }

    return Response(content=buf.tobytes(), media_type="image/png", headers=headers)



def bbox_to_json(bbox):
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    out = []
    for pt in bbox:
        x, y = pt
        out.append([int(x), int(y)])
    return out


@app.post("/preprocess-ocr")
async def preprocess_ocr(
    file: UploadFile = File(...),
    blur_threshold: float = Query(110.0),
    detect_type: bool = Query(True, description="TypeA/TypeB por MRZ (mejor con el reverso)"),
    return_processed_image_base64: bool = Query(False, description="Devuelve imagen tratada en base64 (debug)"),
):
    """
    Devuelve JSON con:
    - Imagen tratada (opcional base64)
    - Meta de doc_detected + blur
    - Clasificación TypeA/TypeB
    - OCR items (texto, conf, bbox) del carnet ya rectificado
    """
    data = await file.read()
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"ok": False, "message": "No se pudo leer la imagen"}, status_code=400)

    processed, meta = preprocess_document(img)

    blur_score = float(meta["blur_score"])
    blurry = blur_score < float(blur_threshold)

    mrz_info = {"mrz_found": False, "mrz_conf": 0.0, "mrz_sample": None}
    ci_type = "UNKNOWN"

    if detect_type and meta["doc_detected"]:
        mrz_info = detect_mrz(processed)
        ci_type = "TypeB" if mrz_info["mrz_found"] else "TypeA"
        
    ocr_result = reader.readtext(processed, detail=1, paragraph=False)

    items = []
    for bbox, text, conf in ocr_result:
        items.append({
            "text": str(text),
            "conf": float(conf),
            "bbox": bbox_to_json(bbox),
        })

    out = {
        "ok": True,
        "doc_detected": bool(meta["doc_detected"]),
        "blur_score": blur_score,
        "blurry": bool(blurry),
        "blur_threshold": float(blur_threshold),

        "ci_type": ci_type,
        "mrz_found": bool(mrz_info["mrz_found"]),
        "mrz_conf": float(mrz_info["mrz_conf"]),
        "mrz_sample": mrz_info["mrz_sample"],

        "items": items,
    }

    if return_processed_image_base64:
        ok, buf = cv2.imencode(".png", processed)
        if ok:
            import base64
            out["processed_image_base64_png"] = base64.b64encode(buf.tobytes()).decode("ascii")

    return out

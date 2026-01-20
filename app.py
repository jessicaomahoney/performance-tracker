import streamlit as st
from datetime import date, timedelta, datetime
from pathlib import Path
import json
import pandas as pd

# OCR / extraction (optional)
import re
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import easyocr
from rapidfuzz import process, fuzz

# =========================================================
# PAGE CONFIG (must be first Streamlit call)
# =========================================================
st.set_page_config(page_title="Team Performance Tracker", layout="wide")

# =========================================================
# CONFIG
# =========================================================
METRICS = ["Calls", "EA Calls", "Things Done"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

DEVIATION_OPTIONS = ["Normal", "Half day", "Sick", "Annual leave", "Reward time", "Other"]
DEVIATION_MULT = {
    "Normal": 1.0,
    "Half day": 0.5,
    "Sick": 0.0,
    "Annual leave": 0.0,
    "Reward time": 0.0,
    "Other": 1.0,
}
EXEMPT_DEVIATIONS = {"Sick", "Annual leave", "Reward time"}

# Password (Streamlit secrets if available)
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "password")

# 🔗 Paste your links here
TRACKER_SHEET_URL = "https://docs.google.com/spreadsheets/d/1t3IvVgIrqC8P9Txca5fZM96rBLvWA6NGu1yb-HD4qHM/edit?gid=0#gid=0"
BACKUP_FOLDER_URL = "https://drive.google.com/drive/folders/1GdvL_eUJK9ShiSr3yD-O1A8HFAYyXBC-"

# OCR tuning defaults
DEFAULT_MAX_OCR_WIDTH = 1800
DEFAULT_COLOR_DIST_THRESHOLD = 140

# =========================================================
# HELPERS
# =========================================================

def monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())

def iso(d: date) -> str:
    return d.isoformat()

def week_label(week_start: date) -> str:
    week_end = week_start + timedelta(days=6)
    return f"{week_start.strftime('%d %b %Y')} – {week_end.strftime('%d %b %Y')}"

def to_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def team_paths(team: str):
    base = Path("data") / team
    return {
        "BASE": base,
        "UPLOADS": base / "uploads",
        "INDEX": base / "index.json",
        "PEOPLE": base / "people.json",
        "BASELINES": base / "baselines.json",
        "REPORTS": base / "reports.json",
        "META": base / "meta.json",
    }

def mark_dirty(reason: str):
    st.session_state["dirty"] = True
    st.session_state["dirty_reason"] = reason

def clear_dirty():
    st.session_state["dirty"] = False
    st.session_state["dirty_reason"] = ""

def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


# ---- name normalisation + mapping (prevents "extract works but nothing writes")

def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_name_map(extracted_people, target_people, min_score=85):
    """Map extracted names (e.g. 'Ben Morton') to target names in-app (e.g. 'Ben')."""
    mapping = {}
    target_norm = {p: _norm_name(p) for p in target_people}

    for ep in extracted_people:
        epn = _norm_name(ep)
        best_p, best_score = None, 0
        for tp, tpn in target_norm.items():
            score = fuzz.token_set_ratio(epn, tpn)
            if score > best_score:
                best_score = score
                best_p = tp
        if best_p and best_score >= min_score:
            mapping[ep] = best_p
    return mapping


# =========================================================
# OCR + EXTRACTION (BETA) — stacked chart (legacy)
# =========================================================

@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)

def _avg_color(img_bgr, x, y, r=6):
    h, w = img_bgr.shape[:2]
    x1, x2 = max(0, x - r), min(w, x + r)
    y1, y2 = max(0, y - r), min(h, y + r)
    patch = img_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([0, 0, 0], dtype=np.float32)
    return patch.reshape(-1, 3).mean(axis=0)

def _color_distance(c1, c2):
    return float(np.linalg.norm(c1 - c2))

def preprocess_for_ocr(image_bytes: bytes, max_w: int = DEFAULT_MAX_OCR_WIDTH):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    if img.width > max_w:
        scale = max_w / float(img.width)
        img = img.resize((max_w, int(img.height * scale)), Image.Resampling.LANCZOS)

    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    sharp = cv2.equalizeHist(sharp)
    thr = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        31, 7
    )
    return img, img_np, img_bgr, thr


def extract_stacked_chart_to_monfri(
    image_bytes,
    known_people,
    max_w=DEFAULT_MAX_OCR_WIDTH,
    color_dist_threshold=DEFAULT_COLOR_DIST_THRESHOLD,
):
    img, img_np, img_bgr, thr = preprocess_for_ocr(image_bytes, max_w=max_w)

    reader = get_ocr_reader()
    ocr = reader.readtext(thr)

    # ---- Find x-axis labels like "Jan 5"
    date_re = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{1,2}$", re.I)
    ticks = []
    for bbox, text, conf in ocr:
        t = str(text).strip()
        if conf >= 0.50 and date_re.match(t):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            cx = int(sum(xs) / 4)
            cy = int(sum(ys) / 4)
            ticks.append((cx, cy, t.title(), conf))

    ticks = sorted(ticks, key=lambda x: x[0])
    if len(ticks) < 5:
        return {}, {"error": f"I couldn’t detect 5 date ticks (found {len(ticks)}). Try a less cropped screenshot."}

    ticks = ticks[:5]
    tick_x = [t[0] for t in ticks]

    # ---- Legend mapping
    legend = {}  # person -> sampled colour (BGR)
    for bbox, text, conf in ocr:
        t = str(text).strip()
        if conf < 0.35 or len(t) < 3:
            continue

        match = process.extractOne(t, known_people, scorer=fuzz.partial_ratio)
        if not match:
            continue
        person, score, _ = match
        if score < 80:
            continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        left = int(min(xs))
        cy = int(sum(ys) / 4)

        sample_x = max(0, left - 18)
        color = _avg_color(img_bgr, sample_x, cy, r=8)
        legend[person] = color

    if not legend:
        return {}, {"error": "I couldn’t match any legend names. Make sure the chart legend matches your People list."}

    # ---- Numeric labels
    num_re = re.compile(r"^\d+$")
    numeric = []
    for bbox, text, conf in ocr:
        t = str(text).strip()
        if conf < 0.30:
            continue
        if not num_re.match(t):
            continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = int(sum(xs) / 4)
        cy = int(sum(ys) / 4)
        val = int(t)

        col = _avg_color(img_bgr, cx, cy + 12, r=8)
        numeric.append((cx, cy, val, conf, col))

    day_by_tick_index = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    out = {d: {} for d in DAYS}

    applied = 0
    for cx, cy, val, conf, col in numeric:
        idx = int(np.argmin([abs(cx - tx) for tx in tick_x]))
        day = day_by_tick_index.get(idx)

        best_person, best_dist = None, 1e9
        for person, lcol in legend.items():
            dist = _color_distance(col, lcol)
            if dist < best_dist:
                best_dist = dist
                best_person = person

        if best_person is None or best_dist > color_dist_threshold:
            continue

        out[day][best_person] = val
        applied += 1

    debug = {
        "mode": "stacked",
        "ticks_detected": [t[2] for t in ticks],
        "legend_people_detected": sorted(list(legend.keys())),
        "numbers_detected": len(numeric),
        "numbers_applied": applied,
        "image_size_used": {"w": img.width, "h": img.height},
        "color_dist_threshold": color_dist_threshold,
        "max_width_used": max_w,
    }
    return out, debug

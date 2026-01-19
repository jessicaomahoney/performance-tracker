import streamlit as st
from datetime import date, timedelta
from pathlib import Path
import json
import pandas as pd

# OCR / extraction (optional fallback)
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
APP_PASSWORD = st.secrets["APP_PASSWORD"] if "APP_PASSWORD" in st.secrets else "password"

# Links (set yours)
TRACKER_SHEET_URL = "https://docs.google.com/spreadsheets/d/1t3IvVgIrqC8P9Txca5fZM96rBLvWA6NGu1yb-HD4qHM/edit?gid=0#gid=0"
BACKUP_FOLDER_URL = "https://drive.google.com/drive/folders/1GdvL_eUJK9ShiSr3yD-O1A8HFAYyXBC-"

# OCR tuning defaults (only used for screenshot mode)
DEFAULT_MAX_OCR_WIDTH = 1600
DEFAULT_COLOR_DIST_THRESHOLD = 120

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

def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _clean_value(x):
    """Return int/float if numeric-ish, else None."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    s = s.replace(",", "")
    if not re.match(r"^\d+(\.\d+)?$", s):
        return None
    try:
        f = float(s)
        if f.is_integer():
            return int(f)
        return f
    except Exception:
        return None

def build_name_map(extracted_people, target_people, min_score=85):
    """Map extracted names to existing people names using fuzzy match."""
    mapping = {}
    targets = list(target_people)

    for ep in extracted_people:
        best_p, best_score = None, 0
        epn = _norm_name(ep)
        for tp in targets:
            score = fuzz.token_set_ratio(epn, _norm_name(tp))
            if score > best_score:
                best_score = score
                best_p = tp
        if best_p and best_score >= min_score:
            mapping[ep] = best_p
    return mapping

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
    }

def mark_dirty(reason: str):
    st.session_state["dirty"] = True
    st.session_state["dirty_reason"] = reason

def clear_dirty():
    st.session_state["dirty"] = False
    st.session_state["dirty_reason"] = ""

# =========================================================
# OCR + EXTRACTION (screenshots fallback)
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

def extract_stacked_chart_to_monfri(image_bytes, known_people, max_w=DEFAULT_MAX_OCR_WIDTH, color_dist_threshold=DEFAULT_COLOR_DIST_THRESHOLD):
    """Screenshot extractor (beta): stacked bars w/ legend colours."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    if img.width > max_w:
        scale = max_w / float(img.width)
        new_size = (max_w, int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

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

    reader = get_ocr_reader()
    ocr = reader.readtext(thr)

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
        return {}, {"error": f"Could not detect 5 date ticks reliably (found {len(ticks)}). Try a less cropped screenshot."}

    ticks = ticks[:5]
    tick_x = [t[0] for t in ticks]

    legend = {}
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
        return {}, {"error": "Legend names could not be matched. Make sure People names match the chart legend."}

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
        "ticks_detected": [t[2] for t in ticks],
        "legend_people_detected": sorted(list(legend.keys())),
        "numbers_detected": len(numeric),
        "numbers_applied": applied,
        "image_size_used": {"w": img.width, "h": img.height},
    }
    return out, debug

# =========================================================
# CSV PARSING (recommended)
# =========================================================
def _looks_like_date_series(series: pd.Series) -> bool:
    """Heuristic: detect YYYY-MM-DD, DD/MM/YYYY, or DD-MM-YYYY in a column."""
    vals = [str(x).strip() for x in series.dropna().tolist()[:20]]
    hits = 0
    for v in vals:
        v10 = v[:10]
        if re.match(r"^\d{4}-\d{2}-\d{2}$", v10):
            hits += 1
        elif re.match(r"^\d{2}/\d{2}/\d{4}$", v10):
            hits += 1
        elif re.match(r"^\d{2}-\d{2}-\d{4}$", v10):
            hits += 1
    return hits >= max(2, int(0.35 * max(1, len(vals))))


def _week_date_aliases(week_start: date):
    """Return a set of acceptable string forms for Mon-Fri dates of the week."""
    out = set()
    for i in range(5):
        d = week_start + timedelta(days=i)
        out.add(d.isoformat())
        out.add(d.strftime('%d/%m/%Y'))
        out.add(d.strftime('%d-%m-%Y'))
    return out


def _parse_date_to_iso(s: str):
    s = (s or '').strip()
    if s == '':
        return None
    s10 = s[:10]
    # YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s10):
        return s10
    # DD/MM/YYYY
    if re.match(r"^\d{2}/\d{2}/\d{4}$", s10):
        dd, mm, yyyy = s10.split('/')
        return f"{yyyy}-{mm}-{dd}"
    # DD-MM-YYYY
    if re.match(r"^\d{2}-\d{2}-\d{4}$", s10):
        dd, mm, yyyy = s10.split('-')
        return f"{yyyy}-{mm}-{dd}"
    return None


def parse_metric_csv_to_monfri(csv_bytes: bytes, week_start: date):
    """
    ✅ Supported CSVs (very forgiving):

    A) Wide weekly (best):
       Person,Baseline,Mon,Tue,Wed,Thu,Fri

    B) Date rows (also good):
       Date,Ben,Rebecca,...
       2026-01-05,83,112,...

    C) Two-row header export like your Total Calls CSV:
       ,User Name,Ben Morton,Rebecca...
       ,Started Date,Count,Count,...
       1,2026-01-09,102,112,...

    Returns:
      out: { 'Mon': {'Person': value, ...}, ... }
      debug: dict
    """
    try:
        df = pd.read_csv(BytesIO(csv_bytes), dtype=str, encoding='utf-8-sig')
    except Exception as e:
        return {}, {'error': f'Could not read CSV: {e}'}

    df.columns = [str(c).strip() for c in df.columns]
    cols_norm = {_norm_name(c): c for c in df.columns}

    # ---- A) Wide weekly ----
    has_person = 'person' in cols_norm
    has_days = all(d.lower() in cols_norm for d in ['mon','tue','wed','thu','fri'])
    if has_person and has_days:
        person_col = cols_norm['person']
        out = {d: {} for d in DAYS}
        for _, row in df.iterrows():
            person = str(row.get(person_col, '')).strip()
            if not person:
                continue
            for d in DAYS:
                v = _clean_value(row.get(cols_norm[d.lower()]))
                if v is not None:
                    out[d][person] = v
        return out, {'format_detected': 'wide_person_monfri', 'rows': int(df.shape[0])}

    # ---- B) Date rows ----
    date_col = None
    for c in df.columns:
        if 'date' in _norm_name(c):
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            if _looks_like_date_series(df[c]):
                date_col = c
                break

    if date_col is not None:
        out = {d: {} for d in DAYS}
        acceptable = _week_date_aliases(week_start)

        # drop index-ish columns from people
        people_cols = []
        for c in df.columns:
            cn = _norm_name(c)
            if c == date_col:
                continue
            if cn == '' or cn.startswith('unnamed') or cn in {'index', 'row'}:
                continue
            people_cols.append(c)

        rows_in_week = 0
        for _, row in df.iterrows():
            raw = str(row.get(date_col, '')).strip()
            raw10 = raw[:10]
            if raw10 not in acceptable:
                continue
            rows_in_week += 1
            iso_d = _parse_date_to_iso(raw10)
            if not iso_d:
                continue
            d_obj = date.fromisoformat(iso_d)
            day_idx = (d_obj - week_start).days
            if day_idx < 0 or day_idx > 4:
                continue
            day_name = DAYS[day_idx]

            for pc in people_cols:
                v = _clean_value(row.get(pc))
                if v is not None:
                    out[day_name][str(pc).strip()] = v

        return out, {'format_detected': 'date_rows_people_cols', 'rows_in_week': rows_in_week, 'date_col': date_col, 'people_cols': people_cols[:25]}

    # ---- C) Two-row header export ----
    try:
        df2 = pd.read_csv(BytesIO(csv_bytes), dtype=str, encoding='utf-8-sig', header=[0, 1])
    except Exception as e:
        sample_cols = ', '.join(df.columns[:6])
        return {}, {'error': f'❌ Couldn\'t understand that CSV format ({e}). Columns looked like: {sample_cols}'}

    flat_cols = []
    for a, b in df2.columns:
        a = str(a).strip()
        b = str(b).strip()
        bn = _norm_name(b)
        an = _norm_name(a)
        if 'date' in bn:
            flat_cols.append('Date')
        elif an and an not in {'user name'} and not an.startswith('unnamed'):
            flat_cols.append(a)
        else:
            flat_cols.append(a if a else b)

    df2.columns = flat_cols

    # drop obvious index cols
    drop_cols = []
    for c in df2.columns:
        cn = _norm_name(c)
        if cn == '' or cn.startswith('unnamed') or cn in {'index', 'row'}:
            drop_cols.append(c)
    if drop_cols:
        df2 = df2.drop(columns=drop_cols, errors='ignore')

    if 'Date' not in df2.columns:
        for c in df2.columns:
            if _looks_like_date_series(df2[c]):
                df2 = df2.rename(columns={c: 'Date'})
                break

    if 'Date' not in df2.columns:
        return {}, {'error': '❌ Couldn\'t find a Date column in that CSV.'}

    out = {d: {} for d in DAYS}
    acceptable = _week_date_aliases(week_start)

    people_cols = []
    for c in df2.columns:
        cn = _norm_name(c)
        if c == 'Date':
            continue
        if cn == '' or cn.startswith('unnamed') or cn in {'index','row'}:
            continue
        people_cols.append(c)

    rows_in_week = 0
    for _, row in df2.iterrows():
        raw = str(row.get('Date', '')).strip()
        raw10 = raw[:10]
        if raw10 not in acceptable:
            continue
        rows_in_week += 1
        iso_d = _parse_date_to_iso(raw10)
        if not iso_d:
            continue
        d_obj = date.fromisoformat(iso_d)
        day_idx = (d_obj - week_start).days
        if day_idx < 0 or day_idx > 4:
            continue
        day_name = DAYS[day_idx]
        for pc in people_cols:
            v = _clean_value(row.get(pc))
            if v is not None:
                out[day_name][str(pc).strip()] = v

    return out, {'format_detected': 'two_row_header_export', 'rows_in_week': rows_in_week, 'people_cols': people_cols[:25]}

# =========================================================
# AUTH
# =========================================================
def check_password() -> bool:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 Login required")
    pw = st.text_input("Enter password", type="password", key="login_pw")

    if pw == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    elif pw:
        st.error("Incorrect password")

    return False

if not check_password():
    st.stop()

# =========================================================
# TEAM SELECT
# =========================================================
def choose_team() -> str:
    if "team" not in st.session_state:
        st.session_state.team = None

    if st.session_state.team:
        return st.session_state.team

    st.title("👋 Quick one — which team are we working on?")
    team = st.radio("Pick one:", ["North", "South"], horizontal=True, key="team_pick")
    if st.button("Continue ➜", use_container_width=True, key="team_continue"):
        st.session_state.team = team
        st.rerun()

    st.stop()

TEAM = choose_team()
P = team_paths(TEAM)
P["BASE"].mkdir(parents=True, exist_ok=True)
P["UPLOADS"].mkdir(parents=True, exist_ok=True)

if "dirty" not in st.session_state:
    clear_dirty()

# =========================================================
# DATA ACCESS
# =========================================================
def default_people_state():
    return {"active": [], "archived": []}


def load_people_state():
    if not P["PEOPLE"].exists():
        save_json(P["PEOPLE"], default_people_state())
    raw = load_json(P["PEOPLE"], default_people_state())

    if isinstance(raw, list):
        raw = {"active": raw, "archived": []}
        save_json(P["PEOPLE"], raw)

    if not isinstance(raw, dict):
        raw = default_people_state()

    raw.setdefault("active", [])
    raw.setdefault("archived", [])

    def clean(lst):
        out, seen = [], set()
        for x in lst:
            name = str(x).strip()
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    raw["active"] = clean(raw["active"])
    raw["archived"] = clean(raw["archived"])
    active_set = set(raw["active"])
    raw["archived"] = [x for x in raw["archived"] if x not in active_set]
    return raw


def save_people_state(state):
    state["active"] = sorted(state.get("active", []), key=lambda x: x.lower())
    state["archived"] = sorted(state.get("archived", []), key=lambda x: x.lower())
    save_json(P["PEOPLE"], state)


def load_baselines():
    return load_json(P["BASELINES"], {})


def save_baselines(b):
    save_json(P["BASELINES"], b)


def load_reports():
    return load_json(P["REPORTS"], {})


def save_reports(r):
    save_json(P["REPORTS"], r)


def load_index():
    return load_json(P["INDEX"], [])


def save_index(idx):
    save_json(P["INDEX"], idx)

# =========================================================
# REPORT STRUCTURE
# =========================================================
def ensure_week_structure(reports: dict, week_iso: str, active_people: list):
    if week_iso not in reports:
        reports[week_iso] = {"people": list(active_people), "metrics": {}, "actions": {}, "deviations": {}}
    else:
        reports[week_iso].setdefault("people", [])
        for p in active_people:
            if p not in reports[week_iso]["people"]:
                reports[week_iso]["people"].append(p)

    reports[week_iso].setdefault("metrics", {})
    reports[week_iso].setdefault("actions", {})
    reports[week_iso].setdefault("deviations", {})

    for m in METRICS:
        reports[week_iso]["metrics"].setdefault(m, {})
        for p in reports[week_iso]["people"]:
            reports[week_iso]["metrics"][m].setdefault(p, {"Baseline": "", **{d: "" for d in DAYS}})

    for p in reports[week_iso]["people"]:
        reports[week_iso]["actions"].setdefault(p, "")
        reports[week_iso]["deviations"].setdefault(p, {d: "Normal" for d in DAYS})


def _ensure_person_in_week(reports: dict, week_iso: str, person: str):
    if person not in reports[week_iso]["people"]:
        reports[week_iso]["people"].append(person)
    reports[week_iso]["actions"].setdefault(person, "")
    reports[week_iso]["deviations"].setdefault(person, {d: "Normal" for d in DAYS})
    for m in METRICS:
        reports[week_iso]["metrics"].setdefault(m, {})
        reports[week_iso]["metrics"][m].setdefault(person, {"Baseline": "", **{d: "" for d in DAYS}})


def generate_report_txt(team: str, week_start: date, block: dict, display_people: list) -> str:
    metrics = block.get("metrics", {})
    actions = block.get("actions", {})
    deviations = block.get("deviations", {})

    lines = []
    lines.append(f"{team} — Weekly performance report — {week_label(week_start)}")
    lines.append(f"Week start (ISO): {iso(week_start)}")
    lines.append("")
    lines.append("🧮 Deviations adjust baseline fairly (Sick/AL/Reward time = EXEMPT, Half day = 50%).")
    lines.append("📌 This report is designed to stand alone (no spreadsheet context).")
    lines.append("")
    lines.append("------------------------------------------------------------")
    lines.append("")

    for person in display_people:
        lines.append(person)
        lines.append("-" * len(person))

        p_dev = deviations.get(person, {d: "Normal" for d in DAYS})
        for metric in METRICS:
            md = metrics.get(metric, {}).get(person, {"Baseline": "", **{d: "" for d in DAYS}})
            base = to_float(md.get("Baseline", ""))

            parts = []
            for d in DAYS:
                val = str(md.get(d, "")).strip()
                dev = p_dev.get(d, "Normal")
                mult = DEVIATION_MULT.get(dev, 1.0)

                if base is None:
                    parts.append(f"{d} - {val}")
                    continue

                adj = base * mult
                if adj == 0:
                    parts.append(f"{d} - {val}/EXEMPT ({dev})")
                else:
                    adj_show = str(int(adj)) if float(adj).is_integer() else str(adj)
                    parts.append(f"{d} - {val}/{adj_show} ({dev})")

            lines.append(f"{metric}: " + " | ".join(parts))

        lines.append("")
        lines.append("TL action (what I will do about it):")
        act = (actions.get(person, "") or "").strip()
        lines.append(act if act else "[NOT FILLED IN]")
        lines.append("")
        lines.append("------------------------------------------------------------")
        lines.append("")

    return "\n".join(lines)


def generate_report_tsv(block: dict, display_people: list) -> str:
    metrics = block.get("metrics", {})
    rows = []
    for p in display_people:
        for m in METRICS:
            md = metrics.get(m, {}).get(p, {"Baseline": "", **{d: "" for d in DAYS}})
            row = [p, m, str(md.get("Baseline", ""))] + [str(md.get(d, "")) for d in DAYS]
            rows.append("\t".join(row))
    return "\n".join(rows)


def week_items(idx, week_iso: str):
    return [x for x in idx if x.get("week_start") == week_iso]


def metric_items(items, metric: str):
    return [x for x in items if x.get("metric") == metric]


def file_exists(filename: str) -> bool:
    return (P["UPLOADS"] / filename).exists()

# =========================================================
# CHECKS
# =========================================================
def run_checks(block: dict, display_people: list, baselines: dict, week_iso: str):
    issues_red, issues_amber = [], []

    for p in display_people:
        act = (block.get("actions", {}).get(p, "") or "").strip()
        if not act:
            issues_red.append(f"TL action missing for **{p}**.")

    deviations = block.get("deviations", {})
    metrics = block.get("metrics", {})

    for p in display_people:
        p_dev = deviations.get(p, {d: "Normal" for d in DAYS})

        for m in METRICS:
            md = metrics.get(m, {}).get(p, {"Baseline": "", **{d: "" for d in DAYS}})
            weekly_base = str(md.get("Baseline", "")).strip()
            default_base = str(baselines.get(p, {}).get(m, "")).strip()
            used_base = weekly_base if weekly_base != "" else default_base

            if used_base == "":
                issues_red.append(f"Baseline missing for **{p}** — **{m}**.")
            elif to_float(used_base) is None:
                issues_amber.append(f"Baseline is not a number for **{p}** — **{m}** (value: `{used_base}`).")

            for d in DAYS:
                dev = p_dev.get(d, "Normal")
                v = str(md.get(d, "")).strip()

                if dev in EXEMPT_DEVIATIONS:
                    if v != "":
                        issues_amber.append(f"Value entered on EXEMPT day for **{p}** — **{m}** — {d} ({dev}).")
                    continue

                if v == "":
                    issues_red.append(f"Missing value for **{p}** — **{m}** — {d} (deviation: {dev}).")

    idx = load_index()
    week_uploads = week_items(idx, week_iso)
    for m in METRICS:
        cnt = sum(1 for x in metric_items(week_uploads, m) if x.get("filename") and file_exists(x["filename"]))
        if cnt == 0:
            issues_amber.append(f"No upload found for **{m}** this week (CSV recommended).")

    if issues_red:
        return "red", issues_red + issues_amber
    if issues_amber:
        return "amber", issues_amber
    return "green", []

# =========================================================
# BACKUP / EXPORT HELPERS
# =========================================================
def build_team_backup(team: str):
    tp = team_paths(team)
    tp["BASE"].mkdir(parents=True, exist_ok=True)
    tp["UPLOADS"].mkdir(parents=True, exist_ok=True)

    people = load_json(tp["PEOPLE"], {"active": [], "archived": []})
    baselines = load_json(tp["BASELINES"], {})
    reports = load_json(tp["REPORTS"], {})
    idx = load_json(tp["INDEX"], [])

    return {
        "team": team,
        "people": people,
        "baselines": baselines,
        "reports": reports,
        "uploads_index": idx,
        "note": "Backup includes people/baselines/reports/index, not the uploaded image files.",
    }


def restore_team_backup(backup_obj: dict, target_team: str):
    tp = team_paths(target_team)
    tp["BASE"].mkdir(parents=True, exist_ok=True)
    tp["UPLOADS"].mkdir(parents=True, exist_ok=True)

    save_json(tp["PEOPLE"], backup_obj.get("people", {"active": [], "archived": []}))
    save_json(tp["BASELINES"], backup_obj.get("baselines", {}))
    save_json(tp["REPORTS"], backup_obj.get("reports", {}))
    save_json(tp["INDEX"], backup_obj.get("uploads_index", []))


def export_week_wide_csv(team: str, week_iso: str) -> bytes:
    b = build_team_backup(team)
    reports = b["reports"]
    block = reports.get(week_iso, None)
    if not block:
        df = pd.DataFrame([{"Team": team, "week_iso": week_iso, "error": "No data for this week"}])
        return df.to_csv(index=False).encode("utf-8")

    rows = []
    for metric in METRICS:
        for person, md in block.get("metrics", {}).get(metric, {}).items():
            row = {
                "Team": team,
                "week_iso": week_iso,
                "Person": person,
                "Metric": metric,
                "Baseline": md.get("Baseline", ""),
            }
            for d in DAYS:
                row[d] = md.get(d, "")
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

# =========================================================
# UI SHELL
# =========================================================
st.title(f"📊 Team Performance Tracker — {TEAM}")

st.caption("✨ Tip: CSV is the most reliable. Screenshots are beta.")
st.warning(
    "⚠️ Streamlit Cloud can reset and wipe saved data. "
    "Please use **💾 Backup & Export** weekly."
)
if st.session_state.get("dirty", False):
    st.info(f"🟦 Quick safety step: download a backup (reason: {st.session_state.get('dirty_reason','changes made')}).")

with st.sidebar:
    st.subheader("👥 Team")
    st.write(f"**{TEAM}**")
    st.divider()
    if st.button("🔁 Switch team", key="switch_team_btn"):
        st.session_state.team = None
        st.rerun()
    if st.button("🚪 Log out", key="logout_btn"):
        st.session_state.authenticated = False
        st.session_state.team = None
        st.rerun()


tab_uploads, tab_baselines, tab_deviations, tab_report, tab_backup = st.tabs(
    ["✅ Uploads", "🎯 Baselines", "🗓️ Deviations", "📝 Weekly report", "💾 Backup & Export"]
)

# =========================================================
# TAB: UPLOADS
# =========================================================
with tab_uploads:
    st.subheader("✅ Uploads (CSV recommended)")

    default_week = monday_of(date.today())
    week_start = monday_of(st.date_input("Week starting (Monday)", value=default_week, key="uploads_week"))
    w_iso = iso(week_start)

    idx = load_index()
    items = week_items(idx, w_iso)

    st.caption(f"📅 Selected week: **{week_label(week_start)}**")
    st.divider()

    st.markdown("### 📥 Upload files (one box per metric)")
    st.caption("Upload **CSV** (best) and/or **PNG/JPG** (beta). Then use the big button to apply everything.")

    colA, colB, colC = st.columns(3)

    def _save_upload(file_obj, metric, kind):
        if not file_obj:
            return None
        safe_name = f"{w_iso}__{metric}__{kind}__{file_obj.name}".replace(" ", "_")
        (P["UPLOADS"] / safe_name).write_bytes(file_obj.getbuffer())
        idx2 = load_index()
        idx2.append({"week_start": w_iso, "metric": metric, "kind": kind, "filename": safe_name})
        save_index(idx2)
        return safe_name

    with colA:
        st.markdown("#### 📞 Calls")
        calls_csv = st.file_uploader("Upload Calls CSV", type=["csv"], key=f"calls_csv_{TEAM}_{w_iso}")
        calls_img = st.file_uploader("Upload Calls screenshot (beta)", type=["png", "jpg", "jpeg"], key=f"calls_img_{TEAM}_{w_iso}")
        if st.button("Save Calls uploads 💾", use_container_width=True, key=f"save_calls_{TEAM}_{w_iso}"):
            saved = 0
            if calls_csv:
                _save_upload(calls_csv, "Calls", "csv"); saved += 1
            if calls_img:
                _save_upload(calls_img, "Calls", "img"); saved += 1
            if saved == 0:
                st.error("Upload a CSV or screenshot first 🙂")
            else:
                mark_dirty("uploads saved")
                st.success(f"Saved {saved} file(s) for Calls ✅")
                st.rerun()

    with colB:
        st.markdown("#### 📞 EA Calls")
        ea_csv = st.file_uploader("Upload EA Calls CSV", type=["csv"], key=f"ea_csv_{TEAM}_{w_iso}")
        ea_img = st.file_uploader("Upload EA Calls screenshot (beta)", type=["png", "jpg", "jpeg"], key=f"ea_img_{TEAM}_{w_iso}")
        if st.button("Save EA Calls uploads 💾", use_container_width=True, key=f"save_ea_{TEAM}_{w_iso}"):
            saved = 0
            if ea_csv:
                _save_upload(ea_csv, "EA Calls", "csv"); saved += 1
            if ea_img:
                _save_upload(ea_img, "EA Calls", "img"); saved += 1
            if saved == 0:
                st.error("Upload a CSV or screenshot first 🙂")
            else:
                mark_dirty("uploads saved")
                st.success(f"Saved {saved} file(s) for EA Calls ✅")
                st.rerun()

    with colC:
        st.markdown("#### ✅ Things Done")
        td_csv = st.file_uploader("Upload Things Done CSV", type=["csv"], key=f"td_csv_{TEAM}_{w_iso}")
        td_img = st.file_uploader("Upload Things Done screenshot (beta)", type=["png", "jpg", "jpeg"], key=f"td_img_{TEAM}_{w_iso}")
        if st.button("Save Things Done uploads 💾", use_container_width=True, key=f"save_td_{TEAM}_{w_iso}"):
            saved = 0
            if td_csv:
                _save_upload(td_csv, "Things Done", "csv"); saved += 1
            if td_img:
                _save_upload(td_img, "Things Done", "img"); saved += 1
            if saved == 0:
                st.error("Upload a CSV or screenshot first 🙂")
            else:
                mark_dirty("uploads saved")
                st.success(f"Saved {saved} file(s) for Things Done ✅")
                st.rerun()

    st.divider()

    st.markdown("### 🧾 This week’s uploads")
    idx = load_index()
    items = week_items(idx, w_iso)
    if not items:
        st.info("No uploads saved for this week yet.")
    else:
        for m in METRICS:
            mi = [x for x in items if x.get("metric") == m]
            if not mi:
                continue
            st.markdown(f"**{m}** — {len(mi)} file(s)")
            for it in reversed(mi[:5]):
                st.caption(f"• {it.get('kind','?').upper()} — {it.get('filename')}")
        st.caption("Tip: if you upload twice, the app uses the most recent CSV first.")

    st.divider()
    st.markdown("### 🗑️ Delete uploads (this week)")
    week_list = [x for x in idx if x.get("week_start") == w_iso]
    if not week_list:
        st.info("No uploads for this week.")
    else:
        options = []
        for i, it in enumerate(week_list):
            options.append(f"{i} | {it.get('metric')} | {it.get('kind','?')} | {it.get('filename')}")
        to_delete = st.multiselect("Select uploads to delete", options=options, key=f"del_{TEAM}_{w_iso}")
        if st.button("Delete selected 🧹", use_container_width=True, key=f"del_btn_{TEAM}_{w_iso}"):
            delete_set = set(to_delete)
            keep = []
            for i, it in enumerate(week_list):
                label = f"{i} | {it.get('metric')} | {it.get('kind','?')} | {it.get('filename')}"
                if label in delete_set:
                    fp = P["UPLOADS"] / it.get("filename", "")
                    try:
                        if fp.exists():
                            fp.unlink()
                    except Exception:
                        pass
                else:
                    keep.append(it)
            new_idx = [x for x in idx if x.get("week_start") != w_iso] + keep
            save_index(new_idx)
            mark_dirty("uploads deleted")
            st.success("Deleted ✅")
            st.rerun()

    st.divider()

    st.markdown("### ✨ Extract data & apply to Weekly report")
    st.caption("One click. We’ll use CSV if present; otherwise we’ll try the screenshot extractor (beta).")
    strict_unreadable = st.checkbox("Fill blanks as UNREADABLE (strict)", value=False, key=f"strict_unreadable_{TEAM}_{w_iso}")

    if st.button("✨ Extract all metrics & apply to Weekly report", use_container_width=True, key=f"extract_all_{TEAM}_{w_iso}"):
        people_state = load_people_state()
        known_people = people_state["active"] + people_state["archived"]

        reports = load_reports()
        ensure_week_structure(reports, w_iso, people_state["active"])
        block = reports[w_iso]

        if not block.get("people"):
            block["people"] = list(known_people)

        summary = {}
        with st.status("🔎 Working on it…", expanded=True) as status:
            status.write(f"Week: {w_iso}")
            idx = load_index()
            week_entries = week_items(idx, w_iso)

            for metric in METRICS:
                status.write(f"➡️ {metric} …")
                entries = [x for x in week_entries if x.get("metric") == metric]
                entries = list(reversed(entries))

                chosen = None
                for e in entries:
                    if e.get("kind") == "csv":
                        chosen = e
                        break
                if chosen is None and entries:
                    chosen = entries[0]

                if not chosen:
                    summary[metric] = "No upload"
                    status.write(f"⚠️ {metric}: no upload found")
                    continue

                fp = P["UPLOADS"] / chosen["filename"]
                if not fp.exists():
                    summary[metric] = "Missing file on disk"
                    status.write(f"⚠️ {metric}: file missing on disk (Streamlit reset?)")
                    continue

                extracted = {}
                dbg = {}
                if chosen.get("kind") == "csv":
                    extracted, dbg = parse_metric_csv_to_monfri(fp.read_bytes(), week_start)
                    if "error" in dbg:
                        summary[metric] = f"CSV error: {dbg['error']}"
                        status.write(f"❌ {metric}: {dbg['error']}")
                        continue
                    status.write(f"✅ {metric}: CSV parsed ({dbg.get('format_detected')})")
                else:
                    extracted, dbg = extract_stacked_chart_to_monfri(fp.read_bytes(), known_people)
                    if "error" in dbg:
                        summary[metric] = f"IMG error: {dbg['error']}"
                        status.write(f"❌ {metric}: {dbg['error']}")
                        continue
                    status.write(f"✅ {metric}: screenshot parsed (beta)")

                extracted_names = set()
                for d in DAYS:
                    extracted_names |= set(extracted.get(d, {}).keys())

                if not block.get("people"):
                    block["people"] = sorted(list(extracted_names), key=lambda x: x.lower())

                name_map = build_name_map(list(extracted_names), list(block["people"]), min_score=85)

                for ep in extracted_names:
                    if ep not in name_map:
                        _ensure_person_in_week(reports, w_iso, ep)
                        name_map[ep] = ep

                wrote = 0
                metric_store = block["metrics"].get(metric, {})
                for day in DAYS:
                    for ep, val in extracted.get(day, {}).items():
                        tp = name_map.get(ep)
                        if not tp:
                            continue
                        metric_store.setdefault(tp, {"Baseline": "", **{d: "" for d in DAYS}})
                        metric_store[tp][day] = str(val)
                        wrote += 1

                if strict_unreadable:
                    for day in DAYS:
                        for tp in block["people"]:
                            metric_store.setdefault(tp, {"Baseline": "", **{d: "" for d in DAYS}})
                            if str(metric_store[tp].get(day, "")).strip() == "":
                                metric_store[tp][day] = "UNREADABLE"

                block["metrics"][metric] = metric_store
                reports[w_iso] = block
                summary[metric] = f"Wrote {wrote} cells"
                status.write(f"📌 {metric}: wrote {wrote} cells")

            save_reports(reports)
            mark_dirty("extracted + applied")
            status.update(label="🎉 Done! Data applied to Weekly report ✅", state="complete")
            st.json(summary)
        st.rerun()

# =========================================================
# TAB: BASELINES
# =========================================================
with tab_baselines:
    st.subheader("🎯 Baselines (Active people)")
    people_state = load_people_state()
    active = people_state["active"]
    archived = people_state["archived"]
    baselines = load_baselines()

    if not active:
        st.info("No active people yet 🙂 Add them below (or restore from backup).")
    else:
        rows = []
        for p in active:
            rows.append({
                "Person": p,
                "Calls": baselines.get(p, {}).get("Calls", ""),
                "EA Calls": baselines.get(p, {}).get("EA Calls", ""),
                "Things Done": baselines.get(p, {}).get("Things Done", ""),
            })
        df = pd.DataFrame(rows)

        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={"Person": st.column_config.TextColumn(disabled=True)},
            key=f"baselines_editor_{TEAM}",
        )

        if st.button("Save baselines 💾", use_container_width=True, key=f"save_baselines_{TEAM}"):
            new_b = load_baselines()
            for _, r in edited.iterrows():
                person = str(r["Person"]).strip()
                if not person:
                    continue
                new_b[person] = {
                    "Calls": str(r.get("Calls", "")).strip(),
                    "EA Calls": str(r.get("EA Calls", "")).strip(),
                    "Things Done": str(r.get("Things Done", "")).strip(),
                }
            save_baselines(new_b)
            mark_dirty("baselines saved")
            st.success("Saved ✅")

    st.divider()
    with st.expander("👥 People management (add / archive / restore)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Active")
            st.markdown("\n".join([f"- {p}" for p in active]) if active else "—")
        with c2:
            st.markdown("### Archived")
            st.markdown("\n".join([f"- {p}" for p in archived]) if archived else "—")

        st.divider()
        new_name = st.text_input("Add person name", key=f"add_person_{TEAM}").strip()
        if st.button("Add to Active ➕", use_container_width=True, key=f"add_person_btn_{TEAM}"):
            if not new_name:
                st.error("Enter a name first 🙂")
            else:
                if new_name in archived:
                    archived.remove(new_name)
                if new_name not in active:
                    active.append(new_name)
                save_people_state({"active": active, "archived": archived})
                mark_dirty("people changed")
                st.success(f"Added {new_name} ✅")
                st.rerun()

        st.divider()
        to_archive = st.multiselect("Archive selected", options=active, key=f"archive_select_{TEAM}")
        if st.button("Archive 🗄️", use_container_width=True, key=f"archive_btn_{TEAM}"):
            for p in to_archive:
                if p in active:
                    active.remove(p)
                if p not in archived:
                    archived.append(p)
            save_people_state({"active": active, "archived": archived})
            mark_dirty("people changed")
            st.success("Archived ✅")
            st.rerun()

        st.divider()
        to_restore = st.multiselect("Restore selected", options=archived, key=f"restore_select_{TEAM}")
        if st.button("Restore ♻️", use_container_width=True, key=f"restore_btn_{TEAM}"):
            for p in to_restore:
                if p in archived:
                    archived.remove(p)
                if p not in active:
                    active.append(p)
            save_people_state({"active": active, "archived": archived})
            mark_dirty("people changed")
            st.success("Restored ✅")
            st.rerun()

# =========================================================
# TAB: DEVIATIONS
# =========================================================
with tab_deviations:
    st.subheader("🗓️ Deviations")
    people_state = load_people_state()
    active = people_state["active"]
    active_set = set(active)

    default_week = monday_of(date.today())
    dev_week = monday_of(st.date_input("Week starting (Monday)", value=default_week, key="dev_week"))
    dev_week_iso = iso(dev_week)

    reports = load_reports()
    ensure_week_structure(reports, dev_week_iso, active)
    save_reports(reports)

    block = load_reports()[dev_week_iso]
    week_people_all = block.get("people", [])

    show_archived = st.checkbox("Show archived people", value=False, key=f"dev_show_archived_{TEAM}_{dev_week_iso}")

    if active:
        display_people = week_people_all if show_archived else [p for p in week_people_all if p in active_set]
    else:
        display_people = week_people_all

    st.caption(f"📅 Selected week: **{week_label(dev_week)}**")

    if not display_people:
        st.info("No people to show yet. Add people in Baselines or apply a CSV extraction first 🙂")
        st.stop()

    rows = []
    for p in display_people:
        dmap = block.get("deviations", {}).get(p, {d: "Normal" for d in DAYS})
        rows.append({"Person": p, **{d: dmap.get(d, "Normal") for d in DAYS}})
    df = pd.DataFrame(rows)

    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Person": st.column_config.TextColumn(disabled=True),
            **{d: st.column_config.SelectboxColumn(options=DEVIATION_OPTIONS) for d in DAYS},
        },
        key=f"deviations_editor_{TEAM}_{dev_week_iso}_{'all' if show_archived else 'active'}",
    )

    if st.button("Save deviations 💾", use_container_width=True, key=f"save_dev_{TEAM}_{dev_week_iso}_{'all' if show_archived else 'active'}"):
        reports = load_reports()
        ensure_week_structure(reports, dev_week_iso, active)
        stored = reports[dev_week_iso].get("deviations", {})
        for _, r in edited.iterrows():
            person = str(r["Person"]).strip()
            stored[person] = {d: r[d] for d in DAYS}
        reports[dev_week_iso]["deviations"] = stored
        save_reports(reports)
        mark_dirty("deviations saved")
        st.success("Saved ✅")

# =========================================================
# TAB: WEEKLY REPORT
# =========================================================
with tab_report:
    st.subheader("📝 Weekly report")
    people_state = load_people_state()
    active = people_state["active"]
    active_set = set(active)

    default_week = monday_of(date.today())
    rep_week = monday_of(st.date_input("Report week starting (Monday)", value=default_week, key="rep_week"))
    rep_week_iso = iso(rep_week)

    baselines = load_baselines()

    reports = load_reports()
    ensure_week_structure(reports, rep_week_iso, active)
    save_reports(reports)

    block = load_reports()[rep_week_iso]
    week_people_all = block.get("people", [])

    show_archived = st.checkbox("Show archived people", value=False, key=f"rep_show_archived_{TEAM}_{rep_week_iso}")

    if active:
        display_people = week_people_all if show_archived else [p for p in week_people_all if p in active_set]
    else:
        display_people = week_people_all

    st.caption(f"📅 Selected week: **{week_label(rep_week)}**")

    if not display_people:
        st.info("No people to show yet. Go to Uploads and run “Extract all metrics & apply” first 🙂")
        st.stop()

    with st.expander("⚠️ Reset this week (danger zone)", expanded=False):
        st.caption("Clears all saved numbers + TL actions for the selected week. (Does not delete uploads.)")
        confirm = st.checkbox("I understand this will clear the week data", key=f"reset_confirm_{TEAM}_{rep_week_iso}")
        if st.button("RESET week now 🧨", use_container_width=True, key=f"reset_week_{TEAM}_{rep_week_iso}", disabled=not confirm):
            reports = load_reports()
            ensure_week_structure(reports, rep_week_iso, active)
            for m in METRICS:
                for p in reports[rep_week_iso]["people"]:
                    reports[rep_week_iso]["metrics"].setdefault(m, {})
                    reports[rep_week_iso]["metrics"][m][p] = {"Baseline": "", **{d: "" for d in DAYS}}
            for p in reports[rep_week_iso]["people"]:
                reports[rep_week_iso]["actions"][p] = ""
            save_reports(reports)
            mark_dirty("week reset")
            st.success("Week reset ✅")
            st.rerun()

    st.divider()
    st.write("✍️ Enter daily actuals. Baseline auto-fills from Baselines (you can override per week).")

    for metric in METRICS:
        st.markdown(f"### {metric}")

        rows = []
        metric_dict = block.get("metrics", {}).get(metric, {})

        for p in display_people:
            saved = metric_dict.get(p, {"Baseline": "", **{d: "" for d in DAYS}})
            default_b = baselines.get(p, {}).get(metric, "")

            baseline = str(saved.get("Baseline", "")).strip()
            if baseline == "" and str(default_b).strip() != "":
                baseline = str(default_b).strip()

            row = {"Person": p, "Baseline": baseline}
            for d in DAYS:
                row[d] = saved.get(d, "")
            rows.append(row)

        df = pd.DataFrame(rows)

        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={"Person": st.column_config.TextColumn(disabled=True)},
            key=f"report_editor_{TEAM}_{rep_week_iso}_{metric}_{'all' if show_archived else 'active'}",
        )

        if st.button(f"Save {metric} 💾", use_container_width=True, key=f"save_metric_{TEAM}_{rep_week_iso}_{metric}_{'all' if show_archived else 'active'}"):
            reports = load_reports()
            ensure_week_structure(reports, rep_week_iso, active)
            stored = reports[rep_week_iso]["metrics"].get(metric, {})
            for _, r in edited.iterrows():
                person = str(r["Person"]).strip()
                stored[person] = {"Baseline": r.get("Baseline", ""), **{d: r.get(d, "") for d in DAYS}}
            reports[rep_week_iso]["metrics"][metric] = stored
            save_reports(reports)
            mark_dirty(f"{metric} saved")
            st.success(f"Saved {metric} ✅")
            st.rerun()

        st.divider()

    st.subheader("🧑‍💼 TL actions (required)")
    for p in display_people:
        st.text_area(
            f"{p} — TL action (what will you do about it?)",
            value=block.get("actions", {}).get(p, ""),
            height=90,
            key=f"action_{TEAM}_{rep_week_iso}_{p}_{'all' if show_archived else 'active'}",
        )

    if st.button("Save TL actions 💾", use_container_width=True, key=f"save_actions_{TEAM}_{rep_week_iso}_{'all' if show_archived else 'active'}"):
        reports = load_reports()
        ensure_week_structure(reports, rep_week_iso, active)
        for p in display_people:
            k = f"action_{TEAM}_{rep_week_iso}_{p}_{'all' if show_archived else 'active'}"
            reports[rep_week_iso]["actions"][p] = st.session_state.get(k, "")
        save_reports(reports)
        mark_dirty("TL actions saved")
        st.success("Saved ✅")

    st.divider()

    reports = load_reports()
    block = reports[rep_week_iso]

    report_txt = generate_report_txt(TEAM, rep_week, block, display_people)
    report_tsv = generate_report_tsv(block, display_people)

    status, issues = run_checks(block, display_people, baselines, rep_week_iso)

    st.subheader("🧪 Readiness checks")
    if status == "green":
        st.success("✅ Ready to send. No issues found.")
    elif status == "amber":
        st.warning("🟠 Some items need attention before sending.")
    else:
        st.error("🔴 Not ready to send. Fix the red items first.")

    if issues:
        st.markdown("**Issues:**")
        for it in issues:
            st.markdown(f"- {it}")

    st.divider()

    st.subheader("✅ TL checklist (do this every week)")
    if status != "green":
        st.warning("Finish the missing items above until the report is GREEN. Then complete the checklist below 🙂")
    else:
        st.success("Report is GREEN — nice work 🎉 Now do the exports below.")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**1) Save backup JSON to the right folder 💾**")
        st.markdown(f"[📁 Open backup folder]({BACKUP_FOLDER_URL})")
        cur_backup = build_team_backup(TEAM)
        st.download_button(
            "Download CURRENT backup.json",
            data=json.dumps(cur_backup, indent=2).encode("utf-8"),
            file_name=f"{TEAM.lower()}_backup.json",
            mime="application/json",
            use_container_width=True,
            disabled=(status != "green"),
            key=f"dl_weekly_backup_{TEAM}_{rep_week_iso}",
        )

    with cB:
        st.markdown("**2) Export Weekly WIDE CSV to the tracker 📤**")
        st.markdown(f"[📊 Open performance tracker]({TRACKER_SHEET_URL})")
        st.download_button(
            "Download Weekly WIDE CSV",
            data=export_week_wide_csv(TEAM, rep_week_iso),
            file_name=f"{TEAM.lower()}_weekly_wide_{rep_week_iso}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=(status != "green"),
            key=f"dl_weekly_wide_{TEAM}_{rep_week_iso}",
        )

    st.caption("Google Sheets import: **File → Import → Upload → choose the CSV** ✅")

    st.divider()

    st.subheader("📋 Report preview (copy/paste)")
    st.caption("Tip: click inside → Ctrl+A → Ctrl+C.")
    st.text_area("Weekly report text", value=report_txt, height=520, key=f"preview_{TEAM}_{rep_week_iso}")

    st.divider()

    st.subheader("⬇️ Other downloads")
    st.download_button(
        "Download weekly report (TXT)",
        data=report_txt.encode("utf-8"),
        file_name=f"{TEAM.lower()}_weekly_report_{rep_week_iso}.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.download_button(
        "Download weekly data (TSV)",
        data=report_tsv.encode("utf-8"),
        file_name=f"{TEAM.lower()}_weekly_data_{rep_week_iso}.tsv",
        mime="text/tab-separated-values",
        use_container_width=True,
    )

# =========================================================
# TAB: BACKUP & EXPORT
# =========================================================
with tab_backup:
    st.subheader("💾 Backup & Export (safe mode)")
    st.markdown("Use this weekly (or after edits). Backups prevent data loss if Streamlit resets 💪")

    st.divider()
    st.markdown("### 📦 Backup JSON (download)")

    c1, c2, c3 = st.columns(3)
    with c1:
        b = build_team_backup("North")
        st.download_button(
            "Download NORTH backup.json",
            data=json.dumps(b, indent=2).encode("utf-8"),
            file_name="north_backup.json",
            mime="application/json",
            use_container_width=True,
            key="dl_backup_north",
        )
    with c2:
        b = build_team_backup("South")
        st.download_button(
            "Download SOUTH backup.json",
            data=json.dumps(b, indent=2).encode("utf-8"),
            file_name="south_backup.json",
            mime="application/json",
            use_container_width=True,
            key="dl_backup_south",
        )
    with c3:
        b = build_team_backup(TEAM)
        st.download_button(
            f"Download CURRENT ({TEAM}) backup.json",
            data=json.dumps(b, indent=2).encode("utf-8"),
            file_name=f"{TEAM.lower()}_backup.json",
            mime="application/json",
            use_container_width=True,
            key="dl_backup_current",
        )

    st.caption("Note: backup does not include uploaded image files (only the index/metadata).")

    st.divider()
    st.markdown("### ♻️ Restore from backup JSON (overwrites saved data)")
    up = st.file_uploader("Upload backup JSON", type=["json"], key="restore_upload")
    target_team = st.radio("Restore into team", ["North", "South"], horizontal=True, key="restore_target_team")

    if st.button("Restore NOW (overwrite) 🚨", use_container_width=True, key="restore_now_btn"):
        if not up:
            st.error("Upload a backup JSON first 🙂")
        else:
            try:
                data = json.loads(up.getvalue().decode("utf-8"))
                restore_team_backup(data, target_team)
                mark_dirty("restored from backup")
                st.success(f"Restored backup into {target_team} ✅")
                st.info("Use “Switch team” (sidebar) to view it.")
            except Exception as e:
                st.error(f"Restore failed: {e}")

    st.divider()
    st.markdown("### ✅ Mark as backed up")
    if st.button("I’ve backed up now ✅", use_container_width=True, key="clear_dirty_btn"):
        clear_dirty()
        st.success("Backup reminder cleared 🎉")

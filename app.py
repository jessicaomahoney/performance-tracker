import streamlit as st
from datetime import date, timedelta
from pathlib import Path
import json
import pandas as pd

# OCR / extraction
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

# Password
APP_PASSWORD = st.secrets["APP_PASSWORD"] if "APP_PASSWORD" in st.secrets else "password"

# Links
TRACKER_SHEET_URL = "https://docs.google.com/spreadsheets/d/1t3IvVgIrqC8P9Txca5fZM96rBLvWA6NGu1yb-HD4qHM/edit?gid=0#gid=0"
BACKUP_FOLDER_URL = "https://drive.google.com/drive/folders/1GdvL_eUJK9ShiSr3yD-O1A8HFAYyXBC-"

DEFAULT_MAX_OCR_WIDTH = 1600
DEFAULT_COLOR_DIST_THRESHOLD = 120

# Default order list (optional; can be edited in UI)
DEFAULT_ORDER_NORTH = [
    "Ben Morton",
    "Hanah Marzook",
    "Jessica O'Mahoney",
    "Mark Williamson",
    "Melissa Hawkey",
    "Mohammed Shahbaz",
    "Rebecca Oldridge",
    "Sara Zahid",
    "Siobhan Arnel",
    "Vanessa O'Reilly",
]
DEFAULT_ORDER_SOUTH = []

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
    }

def mark_dirty(reason: str):
    st.session_state["dirty"] = True
    st.session_state["dirty_reason"] = reason

def clear_dirty():
    st.session_state["dirty"] = False
    st.session_state["dirty_reason"] = ""

def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_name_map(extracted_people, target_people, min_score=80):
    """
    Map extracted names to the names in your app.
    Returns dict: extracted_name -> target_name
    """
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

def parse_order_text(text: str) -> list:
    lines = [x.strip() for x in (text or "").splitlines()]
    lines = [x for x in lines if x]
    out, seen = [], set()
    for x in lines:
        k = _norm_name(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

# =========================================================
# OCR + EXTRACTION
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
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
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
        return {}, {"error": f"Could not detect 5 date ticks reliably (found {len(ticks)})."}

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
        return {}, {"error": "Legend names could not be matched. Check People names match the chart legend."}

    num_re = re.compile(r"^\d+$")
    numeric = []
    for bbox, text, conf in ocr:
        t = str(text).strip()
        if conf < 0.30 or not num_re.match(t):
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = int(sum(xs) / 4)
        cy = int(sum(ys) / 4)
        val = int(t)
        col = _avg_color(img_bgr, cx, cy + 12, r=8)
        numeric.append((cx, val, col))

    day_by_tick_index = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    out = {d: {} for d in DAYS}
    applied = 0

    for cx, val, col in numeric:
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

    return out, {"ticks_detected": [t[2] for t in ticks], "legend_people_detected": sorted(list(legend.keys())), "numbers_applied": applied}

def extract_small_multiples_by_order(image_bytes, ordered_people):
    """
    Recommended extractor:
    For each person title, reads up to 5 numbers below it, left->right => Mon..Fri
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Upscale for OCR
    img_bgr = cv2.resize(img_bgr, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    sharp = cv2.equalizeHist(sharp)

    thr = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )

    reader = get_ocr_reader()
    ocr_all = reader.readtext(thr)
    ocr_nums = reader.readtext(thr, allowlist="0123456789")

    def bbox_bounds(b):
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    name_hits = {}
    for bbox, text, conf in ocr_all:
        t = str(text).strip()
        if conf < 0.30 or len(t) < 2:
            continue
        match = process.extractOne(t, ordered_people, scorer=fuzz.partial_ratio)
        if not match:
            continue
        person, score, _ = match
        if score < 85:
            continue
        if person not in name_hits or conf > name_hits[person]["conf"]:
            name_hits[person] = {"bbox": bbox, "conf": conf, "raw": t}

    out = {d: {} for d in DAYS}

    for person in ordered_people:
        if person not in name_hits:
            continue
        x1, y1, x2, y2 = bbox_bounds(name_hits[person]["bbox"])

        region_left = max(0, x1 - 300)
        region_right = min(img_bgr.shape[1], x2 + 300)
        region_top = y2 + 8
        region_bottom = min(img_bgr.shape[0], region_top + 260)

        nums = []
        for bbox, text, conf in ocr_nums:
            t = str(text).strip()
            if conf < 0.25 or not t.isdigit():
                continue
            nx1, ny1, nx2, ny2 = bbox_bounds(bbox)
            cx = int((nx1 + nx2) / 2)
            cy = int((ny1 + ny2) / 2)
            if region_left <= cx <= region_right and region_top <= cy <= region_bottom:
                nums.append((cx, int(t)))

        nums = sorted(nums, key=lambda x: x[0])
        vals = [v for _, v in nums][:5]

        for i, d in enumerate(DAYS):
            if i < len(vals):
                out[d][person] = vals[i]

    dbg = {
        "people_found_titles": sorted(list(name_hits.keys())),
        "people_missing_titles": [p for p in ordered_people if p not in name_hits],
        "cells_preview": sum(len(out[d]) for d in DAYS),
    }
    return out, dbg

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

    st.title("Select team")
    team = st.radio("Where do you want to work?", ["North", "South"], horizontal=True, key="team_pick")
    if st.button("Continue", use_container_width=True, key="team_continue"):
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
    return {"active": ["Rebecca", "Nicole", "Sonia"], "archived": []}

def load_people_state():
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
            if not name or name in seen:
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

def apply_extracted(reports: dict, week_iso: str, metric: str, extracted: dict, active_people: list, strict_unreadable: bool):
    """
    Applies extracted dict {Day:{Person:val}} into reports for week/metric using name mapping.
    Returns summary dict for UI.
    """
    ensure_week_structure(reports, week_iso, active_people)
    block = reports[week_iso]
    metric_store = block["metrics"].get(metric, {})

    extracted_names = set()
    for d in DAYS:
        extracted_names |= set(extracted.get(d, {}).keys())

    name_map = build_name_map(list(extracted_names), list(block["people"]), min_score=80)

    wrote = 0
    expected = len(block["people"]) * len(DAYS)

    # write mapped values
    for d in DAYS:
        for ep, val in extracted.get(d, {}).items():
            tp = name_map.get(ep)
            if not tp:
                continue
            metric_store.setdefault(tp, {"Baseline": "", **{dd: "" for dd in DAYS}})
            metric_store[tp][d] = str(val)
            wrote += 1

    if strict_unreadable:
        mapped_targets_by_day = {d: set() for d in DAYS}
        for d in DAYS:
            for ep in extracted.get(d, {}).keys():
                tp = name_map.get(ep)
                if tp:
                    mapped_targets_by_day[d].add(tp)

        for d in DAYS:
            for p in block["people"]:
                metric_store.setdefault(p, {"Baseline": "", **{dd: "" for dd in DAYS}})
                if p not in mapped_targets_by_day[d] and str(metric_store[p].get(d, "")).strip() == "":
                    metric_store[p][d] = "UNREADABLE"

    block["metrics"][metric] = metric_store
    reports[week_iso] = block

    unmapped = sorted([n for n in extracted_names if n not in name_map])
    return {"wrote": wrote, "expected": expected, "name_map": name_map, "unmapped_extracted_names": unmapped}

# =========================================================
# REPORT TEXT OUTPUTS
# =========================================================
def generate_report_txt(team: str, week_start: date, block: dict, display_people: list) -> str:
    metrics = block.get("metrics", {})
    actions = block.get("actions", {})
    deviations = block.get("deviations", {})

    lines = []
    lines.append(f"{team} — Weekly performance report — {week_label(week_start)}")
    lines.append(f"Week start (ISO): {iso(week_start)}")
    lines.append("")
    lines.append("Deviations adjust baseline fairly (Sick/AL/Reward time = EXEMPT, Half day = 50%).")
    lines.append("This report is designed to stand alone (no spreadsheet context).")
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
        "note": "Backup includes people/baselines/reports/index, not the uploaded image files."
    }

def export_week_wide_csv(team: str, week_iso: str) -> bytes:
    reports = load_json(team_paths(team)["REPORTS"], {})
    block = reports.get(week_iso, None)
    if not block:
        df = pd.DataFrame([{"Team": team, "week_iso": week_iso, "error": "No data for this week"}])
        return df.to_csv(index=False).encode("utf-8")

    rows = []
    for metric in METRICS:
        for person, md in block.get("metrics", {}).get(metric, {}).items():
            row = {"Team": team, "week_iso": week_iso, "Person": person, "Metric": metric, "Baseline": md.get("Baseline", "")}
            for d in DAYS:
                row[d] = md.get(d, "")
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

# =========================================================
# UI
# =========================================================
st.title(f"📊 Team Performance Tracker — {TEAM}")

st.warning(
    "⚠️ Streamlit Cloud can reset and wipe saved data. "
    "Use **💾 Backup & Export** to download backups regularly (especially after edits)."
)
if st.session_state.get("dirty", False):
    st.info(f"🟦 Safety step: download a backup (reason: {st.session_state.get('dirty_reason','changes made')}).")

with st.sidebar:
    st.subheader("Team")
    st.write(f"**{TEAM}**")
    if st.button("Switch team"):
        st.session_state.team = None
        st.rerun()
    if st.button("Log out"):
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
    st.subheader("Uploads (screenshots)")

    default_week = monday_of(date.today())
    week_start = monday_of(st.date_input("Week starting (Monday)", value=default_week, key="uploads_week"))
    w_iso = iso(week_start)

    st.caption(f"Selected week: **{week_label(week_start)}**")
    st.divider()

    # Metric status
    idx = load_index()
    items = [x for x in idx if x.get("week_start") == w_iso]
    cols = st.columns(3)
    for i, m in enumerate(METRICS):
        c = sum(1 for x in items if x.get("metric") == m and (P["UPLOADS"] / x.get("filename","")).exists())
        cols[i].metric(m, f"{'✅' if c > 0 else '❌'}  {c} uploaded")

    st.divider()

    metric = st.selectbox("Metric type (for uploading)", METRICS, key="uploads_metric")
    files = st.file_uploader(
        "Upload chart screenshots (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="uploads_uploader"
    )

    if st.button("Save uploads", use_container_width=True):
        if not files:
            st.error("Upload at least one screenshot.")
        else:
            idx = load_index()
            saved = 0
            for f in files:
                safe_name = f"{w_iso}_{metric}_{f.name}".replace(" ", "_")
                (P["UPLOADS"] / safe_name).write_bytes(f.getbuffer())
                idx.append({"week_start": w_iso, "metric": metric, "filename": safe_name})
                saved += 1
            save_index(idx)
            mark_dirty("uploads saved")
            st.success(f"Saved {saved} file(s) for {metric}.")
            st.rerun()

    st.divider()
    st.subheader("Extractor settings")

    people_state = load_people_state()
    active_people = people_state["active"]
    known_people = people_state["active"] + people_state["archived"]

    extractor_mode = st.selectbox(
        "Extractor mode",
        ["Small multiples (recommended)", "Stacked chart (legacy)"],
        index=0,
        key=f"extract_mode_{TEAM}_{w_iso}"
    )

    c1, c2 = st.columns(2)
    with c1:
        strict_unreadable = st.checkbox("Strict: fill blanks as UNREADABLE", value=False, key=f"strict_{TEAM}_{w_iso}")
    with c2:
        st.caption("Tip: if names aren’t writing in, reduce matching strictness by using the recommended charts.")

    ordered_people = []
    if extractor_mode.startswith("Small multiples"):
        default_order = DEFAULT_ORDER_NORTH if TEAM == "North" else (DEFAULT_ORDER_SOUTH if DEFAULT_ORDER_SOUTH else active_people)
        order_text = st.text_area(
            "Order list (one per line) — must match the chart order top→bottom",
            value="\n".join(default_order),
            height=180,
            key=f"order_text_{TEAM}_{w_iso}"
        )
        ordered_people = parse_order_text(order_text)

    st.divider()
    st.subheader("✨ Extract data & apply to weekly report")
    st.caption("This runs **Calls + EA Calls + Things Done** for the selected week and writes into the Weekly report tables.")

    if st.button("✨ Extract data & apply to weekly report", use_container_width=True, key=f"run_all_{TEAM}_{w_iso}"):
        reports = load_reports()
        ensure_week_structure(reports, w_iso, active_people)

        idx = load_index()

        # Ensure weekly report tab jumps to the same week
        st.session_state["rep_week"] = week_start

        results = {}
        with st.status("Working… extracting and applying all metrics", expanded=True) as status:
            status.update(state="running")

            for m in METRICS:
                status.write(f"📌 Metric: **{m}** — locating latest upload…")
                candidates = [x for x in idx if x.get("week_start") == w_iso and x.get("metric") == m]
                candidates = list(reversed(candidates))
                if not candidates:
                    results[m] = {"error": "No upload found"}
                    status.write(f"❌ {m}: No upload found")
                    continue

                path = P["UPLOADS"] / candidates[0]["filename"]
                if not path.exists():
                    results[m] = {"error": "File missing on disk (re-upload)"}
                    status.write(f"❌ {m}: File missing on disk (re-upload)")
                    continue

                status.write(f"🔎 {m}: extracting text…")
                if extractor_mode.startswith("Small multiples"):
                    if not ordered_people:
                        results[m] = {"error": "Order list is empty"}
                        status.write(f"❌ {m}: Order list empty")
                        continue
                    extracted, dbg = extract_small_multiples_by_order(path.read_bytes(), ordered_people)
                    status.write(f"✅ {m}: preview cells read: {dbg.get('cells_preview', 0)}")
                    if dbg.get("people_missing_titles"):
                        status.write(f"⚠️ {m}: missing title matches: {dbg['people_missing_titles'][:6]}")
                else:
                    extracted, dbg = extract_stacked_chart_to_monfri(path.read_bytes(), known_people)
                    if "error" in dbg:
                        results[m] = {"error": dbg["error"]}
                        status.write(f"❌ {m}: {dbg['error']}")
                        continue
                    status.write(f"✅ {m}: numbers applied (preview): {dbg.get('numbers_applied', 0)}")

                status.write(f"🧩 {m}: applying values into Weekly report…")
                summary = apply_extracted(
                    reports=reports,
                    week_iso=w_iso,
                    metric=m,
                    extracted=extracted,
                    active_people=active_people,
                    strict_unreadable=bool(strict_unreadable),
                )
                results[m] = summary
                status.write(f"✅ {m}: wrote {summary['wrote']}/{summary['expected']} cells")
                if summary["unmapped_extracted_names"]:
                    status.write(f"⚠️ {m}: unmapped extracted names (first 6): {summary['unmapped_extracted_names'][:6]}")

            save_reports(reports)
            mark_dirty("extraction applied (all metrics)")
            status.update(label="Done ✅ Data extracted and applied", state="complete")

        st.session_state["last_extract_results"] = {"week_iso": w_iso, "results": results}
        st.success("✅ Finished. Go to **📝 Weekly report** tab (it will now be on the same week).")
        st.json(results)

# =========================================================
# TAB: BASELINES
# =========================================================
with tab_baselines:
    st.subheader("Baselines (Active people only)")

    people_state = load_people_state()
    active = people_state["active"]
    archived = people_state["archived"]
    baselines = load_baselines()

    if not active:
        st.info("No active people yet. Use People management below to add someone.")
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
            key=f"baselines_editor_{TEAM}"
        )

        if st.button("Save baselines", use_container_width=True):
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
            st.success("Saved baselines.")

    st.divider()
    with st.expander("People management (add / archive / restore)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Active")
            st.markdown("\n".join([f"- {p}" for p in active]) if active else "—")
        with c2:
            st.markdown("### Archived")
            st.markdown("\n".join([f"- {p}" for p in archived]) if archived else "—")

        st.divider()
        new_name = st.text_input("Add person name").strip()
        if st.button("Add to Active", use_container_width=True):
            if not new_name:
                st.error("Enter a name first.")
            else:
                if new_name in archived:
                    archived.remove(new_name)
                if new_name not in active:
                    active.append(new_name)
                save_people_state({"active": active, "archived": archived})
                mark_dirty("people changed")
                st.success(f"Added {new_name}.")
                st.rerun()

        st.divider()
        to_archive = st.multiselect("Archive selected", options=active)
        if st.button("Archive", use_container_width=True):
            for p in to_archive:
                if p in active:
                    active.remove(p)
                if p not in archived:
                    archived.append(p)
            save_people_state({"active": active, "archived": archived})
            mark_dirty("people changed")
            st.success("Archived.")
            st.rerun()

        st.divider()
        to_restore = st.multiselect("Restore selected", options=archived)
        if st.button("Restore", use_container_width=True):
            for p in to_restore:
                if p in archived:
                    archived.remove(p)
                if p not in active:
                    active.append(p)
            save_people_state({"active": active, "archived": archived})
            mark_dirty("people changed")
            st.success("Restored.")
            st.rerun()

# =========================================================
# TAB: DEVIATIONS
# =========================================================
with tab_deviations:
    st.subheader("Deviations")

    people_state = load_people_state()
    active = people_state["active"]
    active_set = set(active)

    default_week = monday_of(date.today())
    dev_week = monday_of(st.date_input("Week starting (Monday)", value=default_week, key="dev_week"))
    dev_week_iso = iso(dev_week)

    show_archived = st.checkbox("Show archived people", value=False, key=f"dev_show_arch_{TEAM}_{dev_week_iso}")

    reports = load_reports()
    ensure_week_structure(reports, dev_week_iso, active)
    save_reports(reports)

    block = load_reports()[dev_week_iso]
    week_people_all = block.get("people", [])
    display_people = week_people_all if show_archived else [p for p in week_people_all if p in active_set]

    st.caption(f"Selected week: **{week_label(dev_week)}**")
    if not display_people:
        st.info("No active people to show.")
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
        key=f"dev_editor_{TEAM}_{dev_week_iso}_{'all' if show_archived else 'active'}"
    )

    if st.button("Save deviations", use_container_width=True):
        reports = load_reports()
        ensure_week_structure(reports, dev_week_iso, active)
        stored = reports[dev_week_iso].get("deviations", {})
        for _, r in edited.iterrows():
            person = str(r["Person"]).strip()
            stored[person] = {d: r[d] for d in DAYS}
        reports[dev_week_iso]["deviations"] = stored
        save_reports(reports)
        mark_dirty("deviations saved")
        st.success("Saved deviations.")

# =========================================================
# TAB: WEEKLY REPORT
# =========================================================
with tab_report:
    st.subheader("Weekly report")

    people_state = load_people_state()
    active = people_state["active"]
    active_set = set(active)

    default_week = monday_of(date.today())

    rep_week = monday_of(st.date_input("Report week starting (Monday)", value=default_week, key="rep_week"))
    rep_week_iso = iso(rep_week)

    # Show last extraction results clearly
    last = st.session_state.get("last_extract_results")
    if last and last.get("week_iso") == rep_week_iso:
        st.info("✅ Latest extraction results for this week are shown below.")
        st.json(last["results"])

    show_archived = st.checkbox("Show archived people", value=False, key=f"rep_show_arch_{TEAM}_{rep_week_iso}")
    baselines = load_baselines()

    reports = load_reports()
    ensure_week_structure(reports, rep_week_iso, active)
    save_reports(reports)

    block = load_reports()[rep_week_iso]
    week_people_all = block.get("people", [])
    display_people = week_people_all if show_archived else [p for p in week_people_all if p in active_set]

    st.caption(f"Selected week: **{week_label(rep_week)}**")
    if not display_people:
        st.info("No active people to show.")
        st.stop()

    st.divider()
    with st.expander("⚠️ Reset this week (danger zone)", expanded=False):
        st.caption("Clears all saved numbers + TL actions for this week. (Does NOT delete uploads.)")
        confirm = st.checkbox("I understand this will clear the week data", key=f"reset_confirm_{TEAM}_{rep_week_iso}")
        if st.button("RESET week now", use_container_width=True, disabled=not confirm):
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
            st.success("Week reset complete.")
            st.rerun()

    st.divider()
    st.write("Enter daily actuals. Baseline auto-fills from Baselines (you can override per week).")

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
            key=f"report_editor_{TEAM}_{rep_week_iso}_{metric}_{'all' if show_archived else 'active'}"
        )

        if st.button(f"Save {metric}", use_container_width=True, key=f"save_metric_{TEAM}_{rep_week_iso}_{metric}_{'all' if show_archived else 'active'}"):
            reports = load_reports()
            ensure_week_structure(reports, rep_week_iso, active)
            stored = reports[rep_week_iso]["metrics"].get(metric, {})
            for _, r in edited.iterrows():
                person = str(r["Person"]).strip()
                stored[person] = {"Baseline": r.get("Baseline", ""), **{d: r.get(d, "") for d in DAYS}}
            reports[rep_week_iso]["metrics"][metric] = stored
            save_reports(reports)
            mark_dirty(f"{metric} saved")
            st.success(f"Saved {metric}.")
            st.rerun()

        st.divider()

    st.subheader("TL actions (required)")
    for p in display_people:
        st.text_area(
            f"{p} — TL action (what will you do about it?)",
            value=block.get("actions", {}).get(p, ""),
            height=90,
            key=f"action_{TEAM}_{rep_week_iso}_{p}_{'all' if show_archived else 'active'}"
        )

    if st.button("Save TL actions", use_container_width=True):
        reports = load_reports()
        ensure_week_structure(reports, rep_week_iso, active)
        for p in display_people:
            k = f"action_{TEAM}_{rep_week_iso}_{p}_{'all' if show_archived else 'active'}"
            reports[rep_week_iso]["actions"][p] = st.session_state.get(k, "")
        save_reports(reports)
        mark_dirty("TL actions saved")
        st.success("Saved TL actions.")

    st.divider()
    reports = load_reports()
    block = reports[rep_week_iso]
    report_txt = generate_report_txt(TEAM, rep_week, block, display_people)
    report_tsv = generate_report_tsv(block, display_people)

    st.subheader("Report preview (copy/paste)")
    st.text_area("Weekly report text", value=report_txt, height=520)

    st.download_button(
        "Download weekly report (TXT)",
        data=report_txt.encode("utf-8"),
        file_name=f"{TEAM.lower()}_weekly_report_{rep_week_iso}.txt",
        mime="text/plain",
        use_container_width=True
    )
    st.download_button(
        "Download weekly data (TSV)",
        data=report_tsv.encode("utf-8"),
        file_name=f"{TEAM.lower()}_weekly_data_{rep_week_iso}.tsv",
        mime="text/tab-separated-values",
        use_container_width=True
    )

    st.divider()
    st.subheader("Weekly wide CSV export")
    st.markdown(f"[Open performance tracker]({TRACKER_SHEET_URL})")
    st.download_button(
        "Download Weekly WIDE CSV",
        data=export_week_wide_csv(TEAM, rep_week_iso),
        file_name=f"{TEAM.lower()}_weekly_wide_{rep_week_iso}.csv",
        mime="text/csv",
        use_container_width=True
    )

# =========================================================
# TAB: BACKUP & EXPORT
# =========================================================
with tab_backup:
    st.subheader("💾 Backup & Export")

    st.markdown("Backups prevent data loss if Streamlit resets.")
    st.markdown(f"[Open backup folder]({BACKUP_FOLDER_URL})")

    b = build_team_backup(TEAM)
    st.download_button(
        f"Download {TEAM} backup.json",
        data=json.dumps(b, indent=2).encode("utf-8"),
        file_name=f"{TEAM.lower()}_backup.json",
        mime="application/json",
        use_container_width=True
    )

    st.divider()
    if st.button("I’ve backed up now", use_container_width=True):
        clear_dirty()
        st.success("Backup reminder cleared.")

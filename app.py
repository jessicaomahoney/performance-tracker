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

# Password (Streamlit secrets if available)
APP_PASSWORD = st.secrets["APP_PASSWORD"] if "APP_PASSWORD" in st.secrets else "password"

# Links
TRACKER_SHEET_URL = "https://docs.google.com/spreadsheets/d/1t3IvVgIrqC8P9Txca5fZM96rBLvWA6NGu1yb-HD4qHM/edit?gid=0#gid=0"
BACKUP_FOLDER_URL = "https://drive.google.com/drive/folders/1GdvL_eUJK9ShiSr3yD-O1A8HFAYyXBC-"

# OCR tuning defaults
DEFAULT_MAX_OCR_WIDTH = 1600
DEFAULT_COLOR_DIST_THRESHOLD = 120

# Small multiples: default ordered list (you can edit in the UI any time)
DEFAULT_SMALL_MULTIPLES_ORDER_NORTH = [
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
DEFAULT_SMALL_MULTIPLES_ORDER_SOUTH = []  # optional, UI will auto-fill from active people if empty

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

def build_name_map(extracted_people, target_people, min_score=85):
    """
    Map extracted names (from OCR) onto the names in your week table.
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
    # de-dupe preserving order
    out, seen = [], set()
    for x in lines:
        key = _norm_name(x)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out

# =========================================================
# OCR + EXTRACTION (BETA)
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
    """
    Legacy stacked chart extractor (colour + legend).
    Returns:
      out: { "Mon": {"Person": 12, ...}, ... }
      debug: dict
    """
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
        "color_dist_threshold": color_dist_threshold,
        "max_width_used": max_w,
    }
    return out, debug

def extract_small_multiples_by_order(image_bytes, ordered_people, days=DAYS):
    """
    Small multiples extractor (recommended):
    - Each person has a mini-chart with values inside bars
    - We find the person's name, then read up to 5 numbers just below it
    - Assign left->right to Mon..Fri
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Upscale for OCR (helps with tiny fonts)
    img_bgr = cv2.resize(img_bgr, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)

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

    # OCR pass for names and general text
    ocr_all = reader.readtext(thr)

    # OCR pass for digits only
    ocr_nums = reader.readtext(thr, allowlist="0123456789")

    def bbox_bounds(b):
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    # Find best title hit per person
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

    out = {d: {} for d in days}

    for person in ordered_people:
        if person not in name_hits:
            continue

        x1, y1, x2, y2 = bbox_bounds(name_hits[person]["bbox"])

        # Search region below name (tuned for your layout)
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
                nums.append((cx, int(t), conf))

        nums = sorted(nums, key=lambda x: x[0])
        vals = [v for _, v, _ in nums][:5]

        for i, d in enumerate(days):
            if i < len(vals):
                out[d][person] = vals[i]

    debug = {
        "people_expected": list(ordered_people),
        "people_found_as_titles": sorted(list(name_hits.keys())),
        "people_missing_titles": [p for p in ordered_people if p not in name_hits],
        "numbers_total_seen": len(ocr_nums),
        "cells_written_preview": sum(len(out[d]) for d in days),
    }
    return out, debug

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

def week_items(idx, week_iso: str):
    return [x for x in idx if x.get("week_start") == week_iso]

def metric_items(items, metric: str):
    return [x for x in items if x.get("metric") == metric]

def file_exists(filename: str) -> bool:
    return (P["UPLOADS"] / filename).exists()

def apply_extracted_to_week(reports: dict, week_iso: str, metric: str, extracted: dict, active_people: list,
                            strict_unreadable: bool, known_people: list, name_map_min_score: int = 85):
    """
    Applies extracted dict {Day:{Person:val}} into reports[week_iso]["metrics"][metric],
    using fuzzy name mapping so OCR names still populate the week table.
    """
    ensure_week_structure(reports, week_iso, active_people)
    block = reports[week_iso]
    metric_store = block["metrics"].get(metric, {})

    # Build name mapping extracted -> week names
    extracted_names = set()
    for d in DAYS:
        extracted_names |= set(extracted.get(d, {}).keys())

    name_map = build_name_map(
        extracted_people=list(extracted_names),
        target_people=list(block["people"]),
        min_score=name_map_min_score
    )

    wrote = 0
    expected = len(block["people"]) * len(DAYS)

    # Write confident reads
    for day in DAYS:
        day_map = extracted.get(day, {})
        for extracted_person, val in day_map.items():
            target_person = name_map.get(extracted_person)
            if not target_person:
                continue
            metric_store.setdefault(target_person, {"Baseline": "", **{d: "" for d in DAYS}})
            metric_store[target_person][day] = str(val)
            wrote += 1

    # Optional strict fill
    if strict_unreadable:
        for day in DAYS:
            day_map = extracted.get(day, {})
            for person in block["people"]:
                if person in known_people:
                    # if the person has no value extracted for that day
                    # AND the cell is currently blank, fill UNREADABLE
                    if person not in [name_map.get(ep) for ep in day_map.keys()]:
                        metric_store.setdefault(person, {"Baseline": "", **{d: "" for d in DAYS}})
                        if str(metric_store[person].get(day, "")).strip() == "":
                            metric_store[person][day] = "UNREADABLE"

    block["metrics"][metric] = metric_store
    reports[week_iso] = block
    return wrote, expected, name_map

# =========================================================
# CHECKS
# =========================================================
def run_checks(block: dict, display_people: list, active_set: set, baselines: dict, week_iso: str):
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
            issues_amber.append(f"No screenshot uploaded for **{m}** this week (optional).")

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
        "note": "Backup includes people/baselines/reports/index, not the uploaded image files."
    }

def restore_team_backup(backup_obj: dict, target_team: str):
    tp = team_paths(target_team)
    tp["BASE"].mkdir(parents=True, exist_ok=True)
    tp["UPLOADS"].mkdir(parents=True, exist_ok=True)

    save_json(tp["PEOPLE"], backup_obj.get("people", {"active": [], "archived": []}))
    save_json(tp["BASELINES"], backup_obj.get("baselines", {}))
    save_json(tp["REPORTS"], backup_obj.get("reports", {}))
    save_json(tp["INDEX"], backup_obj.get("uploads_index", []))

def export_people_csv(team: str) -> bytes:
    b = build_team_backup(team)
    people = b["people"]
    rows = []
    for p in people.get("active", []):
        rows.append({"Team": team, "Person": p, "Status": "Active"})
    for p in people.get("archived", []):
        rows.append({"Team": team, "Person": p, "Status": "Archived"})
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

def export_baselines_csv(team: str) -> bytes:
    b = build_team_backup(team)
    baselines = b["baselines"]
    people = b["people"]
    all_people = sorted(set(people.get("active", []) + people.get("archived", [])), key=lambda x: x.lower())
    rows = []
    for p in all_people:
        rows.append({
            "Team": team,
            "Person": p,
            "Calls": baselines.get(p, {}).get("Calls", ""),
            "EA Calls": baselines.get(p, {}).get("EA Calls", ""),
            "Things Done": baselines.get(p, {}).get("Things Done", ""),
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

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
    if st.button("Switch team", key="switch_team_btn"):
        st.session_state.team = None
        st.rerun()
    if st.button("Log out", key="logout_btn"):
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

    idx = load_index()
    items = week_items(idx, w_iso)

    st.caption(f"Selected week: **{week_label(week_start)}**")
    st.divider()

    cols = st.columns(3)
    for i, m in enumerate(METRICS):
        c = sum(1 for x in metric_items(items, m) if x.get("filename") and file_exists(x["filename"]))
        cols[i].metric(m, f"{'✅' if c > 0 else '❌'}  {c} uploaded")

    st.divider()

    metric = st.selectbox("Metric type", METRICS, key="uploads_metric")
    files = st.file_uploader(
        "Upload chart screenshots (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="uploads_uploader"
    )

    if st.button("Save uploads", use_container_width=True, key="save_uploads_btn"):
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
    st.subheader("This week’s uploads")
    idx = load_index()
    items = week_items(idx, w_iso)

    if not items:
        st.info("No uploads saved for this week yet.")
    else:
        for m in METRICS:
            mi = metric_items(items, m)
            if not mi:
                continue
            st.markdown(f"### {m}")
            for it in reversed(mi):
                pth = P["UPLOADS"] / it["filename"]
                if pth.exists():
                    st.image(str(pth), width=900)
            st.divider()

    st.subheader("Delete uploads (this week)")
    idx = load_index()
    week_list = [x for x in idx if x.get("week_start") == w_iso]
    if not week_list:
        st.info("No uploads for this week.")
    else:
        options = []
        for i, it in enumerate(week_list):
            options.append(f"{i} | {it.get('metric')} | {it.get('filename')}")
        to_delete = st.multiselect("Select uploads to delete", options=options, key=f"del_{TEAM}_{w_iso}")
        if st.button("Delete selected", use_container_width=True, key=f"del_btn_{TEAM}_{w_iso}"):
            delete_set = set(to_delete)
            keep_week = []
            for i, it in enumerate(week_list):
                label = f"{i} | {it.get('metric')} | {it.get('filename')}"
                if label in delete_set:
                    fp = P["UPLOADS"] / it.get("filename", "")
                    try:
                        if fp.exists():
                            fp.unlink()
                    except Exception:
                        pass
                else:
                    keep_week.append(it)
            new_idx = [x for x in idx if x.get("week_start") != w_iso] + keep_week
            save_index(new_idx)
            mark_dirty("uploads deleted")
            st.success("Deleted selected uploads.")
            st.rerun()

    st.divider()
    st.subheader("Extract & apply (beta)")
    st.caption("Recommended: use **Small multiples** screenshots. This fills what it can read confidently.")

    people_state = load_people_state()
    known_people = people_state["active"] + people_state["archived"]

    extractor_mode = st.selectbox(
        "Extractor mode",
        ["Small multiples (recommended)", "Stacked chart (legacy)"],
        index=0,
        key=f"extractor_mode_{TEAM}_{w_iso}"
    )

    if extractor_mode.startswith("Small multiples"):
        if TEAM == "North":
            default_order = DEFAULT_SMALL_MULTIPLES_ORDER_NORTH
        else:
            default_order = DEFAULT_SMALL_MULTIPLES_ORDER_SOUTH if DEFAULT_SMALL_MULTIPLES_ORDER_SOUTH else (people_state["active"] or [])

        order_text = st.text_area(
            "Order list (one per line) — must match the chart order top→bottom",
            value="\n".join(default_order),
            height=200,
            key=f"order_text_{TEAM}_{w_iso}"
        )
        ordered_people = parse_order_text(order_text)
        if not ordered_people:
            st.warning("Add at least 1 name in the order list (one per line).")
    else:
        ordered_people = []

    # Controls (stacked chart only, but shown anyway)
    c1, c2, c3 = st.columns(3)
    with c1:
        max_w = st.selectbox("OCR max width", [1200, 1400, 1600, 1800], index=2, key=f"ocr_maxw_{TEAM}_{w_iso}")
    with c2:
        dist_thr = st.selectbox("Colour match strictness", [90, 110, 120, 140], index=2, key=f"ocr_thr_{TEAM}_{w_iso}")
    with c3:
        strict_unreadable = st.checkbox("Strict: fill blanks as UNREADABLE", value=False, key=f"strict_unreadable_{TEAM}_{w_iso}")

    # Latest file for the selected metric
    idx = load_index()
    candidates = [x for x in idx if x.get("week_start") == w_iso and x.get("metric") == metric]
    candidates = list(reversed(candidates))

    if not candidates:
        st.info("Upload a screenshot for the selected metric first (and click Save uploads).")
    else:
        latest = candidates[0]
        img_path = P["UPLOADS"] / latest["filename"]

        if not img_path.exists():
            st.warning("Latest upload file not found on disk (Streamlit may have reset uploads). Re-upload the screenshot.")
        else:
            st.image(str(img_path), width=900)

            if st.button("Attempt extract (preview)", use_container_width=True, key=f"extract_preview_{TEAM}_{w_iso}_{metric}"):
                if extractor_mode.startswith("Small multiples"):
                    extracted, dbg = extract_small_multiples_by_order(img_path.read_bytes(), ordered_people)
                    st.write("People found as titles:", dbg.get("people_found_as_titles"))
                    missing = dbg.get("people_missing_titles", [])
                    if missing:
                        st.warning(f"Missing title matches: {missing[:10]}{'...' if len(missing) > 10 else ''}")
                    st.write("Cells written (preview):", dbg.get("cells_written_preview"))
                else:
                    extracted, dbg = extract_stacked_chart_to_monfri(
                        img_path.read_bytes(),
                        known_people,
                        max_w=int(max_w),
                        color_dist_threshold=int(dist_thr)
                    )
                    if "error" in dbg:
                        st.error(dbg["error"])
                        extracted = {}
                    else:
                        st.write("Detected ticks:", dbg["ticks_detected"])
                        st.write("Detected legend names:", dbg["legend_people_detected"])
                        st.write(f"Numbers detected: {dbg['numbers_detected']} | Applied: {dbg['numbers_applied']}")

                st.json(extracted)
                st.session_state["last_extracted"] = {"week_iso": w_iso, "metric": metric, "data": extracted, "mode": extractor_mode}

            if st.button("Apply extraction to weekly tables", use_container_width=True, key=f"extract_apply_{TEAM}_{w_iso}_{metric}"):
                payload = st.session_state.get("last_extracted")
                if not payload or payload.get("week_iso") != w_iso or payload.get("metric") != metric:
                    st.error("Run 'Attempt extract (preview)' first for this week + metric.")
                else:
                    extracted = payload["data"]
                    reports = load_reports()
                    people_state = load_people_state()
                    active = people_state["active"]

                    wrote, expected, name_map = apply_extracted_to_week(
                        reports=reports,
                        week_iso=w_iso,
                        metric=metric,
                        extracted=extracted,
                        active_people=active,
                        strict_unreadable=bool(strict_unreadable),
                        known_people=known_people,
                        name_map_min_score=85
                    )

                    save_reports(reports)
                    mark_dirty(f"extraction applied for {metric}")

                    pct = int(round((wrote / expected) * 100)) if expected else 0
                    st.success(f"Applied extraction to {metric} for week {w_iso}. Filled {wrote}/{expected} cells ({pct}%).")
                    st.info(f"Name mapping used (OCR → Week table): {name_map}")
                    st.rerun()

    st.divider()
    st.subheader("Extract & apply ALL metrics (this week)")
    st.caption("Uploads must exist for each metric. This runs the current extractor mode for all three metrics.")

    if st.button("Run ALL metrics now", use_container_width=True, key=f"run_all_metrics_{TEAM}_{w_iso}"):
        reports = load_reports()
        people_state = load_people_state()
        active = people_state["active"]
        ensure_week_structure(reports, w_iso, active)

        idx = load_index()
        summary = {}
        total_wrote = 0
        total_expected = 0

        for m in METRICS:
            cand = [x for x in idx if x.get("week_start") == w_iso and x.get("metric") == m]
            cand = list(reversed(cand))
            if not cand:
                summary[m] = "No upload"
                continue

            path = P["UPLOADS"] / cand[0]["filename"]
            if not path.exists():
                summary[m] = "File missing on disk"
                continue

            if extractor_mode.startswith("Small multiples"):
                if not ordered_people:
                    summary[m] = "Order list empty"
                    continue
                extracted, dbg = extract_small_multiples_by_order(path.read_bytes(), ordered_people)
            else:
                extracted, dbg = extract_stacked_chart_to_monfri(
                    path.read_bytes(),
                    known_people,
                    max_w=int(max_w),
                    color_dist_threshold=int(dist_thr)
                )
                if "error" in dbg:
                    summary[m] = f"Error: {dbg['error']}"
                    continue

            wrote, expected, name_map = apply_extracted_to_week(
                reports=reports,
                week_iso=w_iso,
                metric=m,
                extracted=extracted,
                active_people=active,
                strict_unreadable=bool(strict_unreadable),
                known_people=known_people,
                name_map_min_score=85
            )

            total_wrote += wrote
            total_expected += expected
            summary[m] = f"Wrote {wrote}/{expected} cells"

        save_reports(reports)
        mark_dirty("all metrics extracted+applied")
        st.success("Done. Summary:")
        st.json(summary)
        if total_expected:
            st.info(f"Overall filled: {total_wrote}/{total_expected} cells ({int(round((total_wrote/total_expected)*100))}%).")
        st.rerun()

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
        st.info("No active people yet. Expand 'People management' below to add someone.")
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

        if st.button("Save baselines", use_container_width=True, key=f"save_baselines_{TEAM}"):
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
        new_name = st.text_input("Add person name", key=f"add_person_{TEAM}").strip()
        if st.button("Add to Active", use_container_width=True, key=f"add_person_btn_{TEAM}"):
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
        to_archive = st.multiselect("Archive selected", options=active, key=f"archive_select_{TEAM}")
        if st.button("Archive", use_container_width=True, key=f"archive_btn_{TEAM}"):
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
        to_restore = st.multiselect("Restore selected", options=archived, key=f"restore_select_{TEAM}")
        if st.button("Restore", use_container_width=True, key=f"restore_btn_{TEAM}"):
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

    show_archived = st.checkbox("Show archived people", value=False, key=f"dev_show_archived_{TEAM}_{dev_week_iso}")

    reports = load_reports()
    ensure_week_structure(reports, dev_week_iso, active)
    save_reports(reports)

    block = load_reports()[dev_week_iso]
    week_people_all = block.get("people", [])
    display_people = week_people_all if show_archived else [p for p in week_people_all if p in active_set]

    st.caption(f"Selected week: **{week_label(dev_week)}**")

    if not display_people:
        st.info("No active people to show. Add/restore people in Baselines.")
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
        key=f"deviations_editor_{TEAM}_{dev_week_iso}_{'all' if show_archived else 'active'}"
    )

    if st.button("Save deviations", use_container_width=True, key=f"save_dev_{TEAM}_{dev_week_iso}_{'all' if show_archived else 'active'}"):
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

    show_archived = st.checkbox("Show archived people", value=False, key=f"rep_show_archived_{TEAM}_{rep_week_iso}")

    baselines = load_baselines()

    reports = load_reports()
    ensure_week_structure(reports, rep_week_iso, active)
    save_reports(reports)

    block = load_reports()[rep_week_iso]
    week_people_all = block.get("people", [])
    display_people = week_people_all if show_archived else [p for p in week_people_all if p in active_set]

    st.caption(f"Selected week: **{week_label(rep_week)}**")

    if not display_people:
        st.info("No active people to show. Add/restore people in Baselines.")
        st.stop()

    st.divider()
    with st.expander("⚠️ Reset this week (danger zone)", expanded=False):
        st.caption("Clears all saved numbers + TL actions for this week. (Does NOT delete uploads.)")
        confirm = st.checkbox("I understand this will clear the week data", key=f"reset_confirm_{TEAM}_{rep_week_iso}")
        if st.button("RESET week now", use_container_width=True, key=f"reset_week_{TEAM}_{rep_week_iso}", disabled=not confirm):
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

    if st.button("Save TL actions", use_container_width=True, key=f"save_actions_{TEAM}_{rep_week_iso}_{'all' if show_archived else 'active'}"):
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

    status, issues = run_checks(block, display_people, active_set, baselines, rep_week_iso)

    st.subheader("Readiness checks")
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
        st.warning("Finish the missing items above until the report is GREEN. Then complete the checklist below.")
    else:
        st.success("Report is GREEN — now complete the export steps below.")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**1) Save backup JSON to the right folder**")
        st.markdown(f"[Open backup folder]({BACKUP_FOLDER_URL})")
        cur_backup = build_team_backup(TEAM)
        st.download_button(
            "Download CURRENT backup.json",
            data=json.dumps(cur_backup, indent=2).encode("utf-8"),
            file_name=f"{TEAM.lower()}_backup.json",
            mime="application/json",
            use_container_width=True,
            disabled=(status != "green"),
            key=f"dl_weekly_backup_{TEAM}_{rep_week_iso}"
        )

    with cB:
        st.markdown("**2) Export Weekly WIDE CSV to the tracker**")
        st.markdown(f"[Open performance tracker]({TRACKER_SHEET_URL})")
        st.download_button(
            "Download Weekly WIDE CSV",
            data=export_week_wide_csv(TEAM, rep_week_iso),
            file_name=f"{TEAM.lower()}_weekly_wide_{rep_week_iso}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=(status != "green"),
            key=f"dl_weekly_wide_{TEAM}_{rep_week_iso}"
        )

    st.caption("Google Sheets import: File → Import → Upload → choose the CSV.")

    st.divider()

    st.subheader("Report preview (copy/paste)")
    st.caption("Tip: click inside → Ctrl+A → Ctrl+C.")
    st.text_area("Weekly report text", value=report_txt, height=520, key=f"preview_{TEAM}_{rep_week_iso}")

    st.divider()
    st.subheader("Other downloads")
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

# =========================================================
# TAB: BACKUP & EXPORT
# =========================================================
with tab_backup:
    st.subheader("💾 Backup & Export (safe mode)")

    st.markdown("**Use this weekly** (or after edits). Backups prevent data loss if Streamlit resets.")

    st.divider()
    st.markdown("### Backup JSON (download)")

    c1, c2, c3 = st.columns(3)
    with c1:
        b = build_team_backup("North")
        st.download_button(
            "Download NORTH backup.json",
            data=json.dumps(b, indent=2).encode("utf-8"),
            file_name="north_backup.json",
            mime="application/json",
            use_container_width=True,
            key="dl_backup_north"
        )
    with c2:
        b = build_team_backup("South")
        st.download_button(
            "Download SOUTH backup.json",
            data=json.dumps(b, indent=2).encode("utf-8"),
            file_name="south_backup.json",
            mime="application/json",
            use_container_width=True,
            key="dl_backup_south"
        )
    with c3:
        b = build_team_backup(TEAM)
        st.download_button(
            f"Download CURRENT ({TEAM}) backup.json",
            data=json.dumps(b, indent=2).encode("utf-8"),
            file_name=f"{TEAM.lower()}_backup.json",
            mime="application/json",
            use_container_width=True,
            key="dl_backup_current"
        )

    st.caption("Note: backup does not include uploaded image files (only the index/metadata).")

    st.divider()
    st.markdown("### Restore from backup JSON (overwrites saved data)")
    up = st.file_uploader("Upload backup JSON", type=["json"], key="restore_upload")
    target_team = st.radio("Restore into team", ["North", "South"], horizontal=True, key="restore_target_team")

    if st.button("Restore NOW (overwrite)", use_container_width=True, key="restore_now_btn"):
        if not up:
            st.error("Upload a backup JSON first.")
        else:
            try:
                data = json.loads(up.getvalue().decode("utf-8"))
                restore_team_backup(data, target_team)
                mark_dirty("restored from backup")
                st.success(f"Restored backup into {target_team}.")
                st.info("Switch team (sidebar) to view it.")
            except Exception as e:
                st.error(f"Restore failed: {e}")

    st.divider()
    st.markdown("### Manual exports (CSV) for Google Sheets")
    c4, c5 = st.columns(2)
    with c4:
        st.download_button(
            "Export People CSV (this team)",
            data=export_people_csv(TEAM),
            file_name=f"{TEAM.lower()}_people.csv",
            mime="text/csv",
            use_container_width=True,
            key="exp_people_team"
        )
    with c5:
        st.download_button(
            "Export Baselines CSV (this team)",
            data=export_baselines_csv(TEAM),
            file_name=f"{TEAM.lower()}_baselines.csv",
            mime="text/csv",
            use_container_width=True,
            key="exp_baselines_team"
        )

    st.divider()
    st.markdown("### Mark as backed up")
    if st.button("I’ve backed up now", use_container_width=True, key="clear_dirty_btn"):
        clear_dirty()
        st.success("Backup reminder cleared.")

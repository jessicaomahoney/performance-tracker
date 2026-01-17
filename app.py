import streamlit as st
from datetime import date, timedelta, datetime
from pathlib import Path
import json
import pandas as pd

import re
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import easyocr
from rapidfuzz import fuzz

# =========================================================
# PAGE CONFIG (must be first Streamlit call)
# =========================================================
st.set_page_config(page_title="Team Performance Tracker", layout="wide")

# =========================================================
# CONFIG
# =========================================================
METRICS = ["Calls", "EA Calls", "Things Done"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

DEVIATION_OPTIONS = ["Normal", "Half day", "Sick", "Annual leave", "Reward time", "Excluded", "Other"]
DEVIATION_MULT = {
    "Normal": 1.0,
    "Half day": 0.5,
    "Sick": 0.0,
    "Annual leave": 0.0,
    "Reward time": 0.0,
    "Excluded": 0.0,   # <-- important
    "Other": 1.0,
}
EXEMPT_DEVIATIONS = {"Sick", "Annual leave", "Reward time", "Excluded"}

APP_PASSWORD = st.secrets["APP_PASSWORD"] if "APP_PASSWORD" in st.secrets else "password"

# 🔗 Your links
TRACKER_SHEET_URL = "https://docs.google.com/spreadsheets/d/1t3IvVgIrqC8P9Txca5fZM96rBLvWA6NGu1yb-HD4qHM/edit?gid=0#gid=0"
BACKUP_FOLDER_URL = "https://drive.google.com/drive/folders/1GdvL_eUJK9ShiSr3yD-O1A8HFAYyXBC-"

# OCR tuning
DEFAULT_OCR_MAX_W = 2200

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

def file_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_name_map(extracted_people, target_people, min_score=85):
    """
    Map extracted names -> app names (handles Ben vs Ben Morton, apostrophes, etc.)
    Returns dict: extracted_name -> target_name
    """
    mapping = {}
    targets = list(target_people)

    for ep in extracted_people:
        epn = _norm_name(ep)
        best_tp, best_score = None, 0
        for tp in targets:
            score = fuzz.token_set_ratio(epn, _norm_name(tp))
            if score > best_score:
                best_score = score
                best_tp = tp
        if best_tp and best_score >= min_score:
            mapping[ep] = best_tp
    return mapping

# =========================================================
# OCR
# =========================================================
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)

def _prep_for_ocr(img: Image.Image, max_w=DEFAULT_OCR_MAX_W):
    img = img.convert("RGB")
    if img.width > max_w:
        scale = max_w / float(img.width)
        img = img.resize((max_w, int(img.height * scale)), Image.Resampling.LANCZOS)

    img_np = np.array(img)
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # improve clarity
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    sharp = cv2.equalizeHist(sharp)

    thr = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        31, 7
    )
    return img, thr

def _center(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (sum(xs) / 4.0, sum(ys) / 4.0)

def extract_table_screenshot_to_monfri(image_bytes, known_people, week_start: date):
    """
    Reads the TABLE screenshot like:
      Completed Date | Ben Morton (Count) | Hanah ... etc
    Then maps by actual date -> Mon..Fri of selected week.

    Returns:
      out: { "Mon": {"Person": 12, ...}, ... }
      debug: dict
    """
    img = Image.open(BytesIO(image_bytes))
    _, thr = _prep_for_ocr(img)

    reader = get_ocr_reader()
    ocr = reader.readtext(thr)

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    num_re = re.compile(r"^\d+$")

    # 1) collect date cells
    dates = []
    for bbox, text, conf in ocr:
        t = str(text).strip()
        if conf >= 0.55 and date_re.match(t):
            cx, cy = _center(bbox)
            dates.append({"text": t, "cx": cx, "cy": cy})

    if not dates:
        return {}, {"error": "No YYYY-MM-DD dates detected. Use the table screenshot (not the chart)."}

    # group rows by y (dates define rows)
    dates = sorted(dates, key=lambda r: r["cy"])

    # 2) detect header names (top area)
    header_candidates = []
    for bbox, text, conf in ocr:
        t = str(text).strip()
        if conf < 0.40:
            continue
        if len(t) < 3:
            continue
        if date_re.match(t) or num_re.match(t):
            continue

        cx, cy = _center(bbox)
        # header region tends to be near top (before first date row)
        if cy < dates[0]["cy"] - 10:
            header_candidates.append((t, cx, cy, conf))

    # match header candidates to known people
    people_x = {}
    for t, cx, cy, conf in header_candidates:
        # ignore obvious non-names
        low = t.lower()
        if "count" in low or "completed" in low or "date" in low or "things" in low:
            continue

        best_p, best_score = None, 0
        for p in known_people:
            score = fuzz.token_set_ratio(_norm_name(t), _norm_name(p))
            if score > best_score:
                best_score = score
                best_p = p
        if best_p and best_score >= 85:
            # store x center for that person column
            # if already exists, keep the one with higher confidence-ish (closer to top)
            if best_p not in people_x or cy < people_x[best_p]["cy"]:
                people_x[best_p] = {"x": cx, "cy": cy, "raw": t, "score": best_score}

    if not people_x:
        return {}, {"error": "Could not detect any person headers. Ensure names are visible at the top of the table."}

    # 3) collect numeric cells
    nums = []
    for bbox, text, conf in ocr:
        t = str(text).strip()
        if conf < 0.45:
            continue
        if num_re.match(t):
            cx, cy = _center(bbox)
            nums.append({"val": int(t), "cx": cx, "cy": cy})

    # helper: map date -> day label if within selected Mon-Fri
    def date_to_day(dstr):
        try:
            dt = datetime.strptime(dstr, "%Y-%m-%d").date()
        except Exception:
            return None
        if dt < week_start or dt > (week_start + timedelta(days=4)):
            return None
        idx = dt.weekday()  # Mon=0
        if idx < 0 or idx > 4:
            return None
        return DAYS[idx]

    out = {d: {} for d in DAYS}
    applied = 0

    # for each date row, find numeric values in same row area and nearest to each person x
    # "row band" = around the date's cy
    row_band = 18

    for drow in dates:
        day = date_to_day(drow["text"])
        if not day:
            continue

        row_y = drow["cy"]
        row_nums = [n for n in nums if abs(n["cy"] - row_y) <= row_band]

        for person, meta in people_x.items():
            px = meta["x"]
            # nearest number by x distance within row band
            best = None
            best_dx = 1e9
            for n in row_nums:
                dx = abs(n["cx"] - px)
                if dx < best_dx:
                    best_dx = dx
                    best = n
            if best is None:
                continue
            # strict-ish: must be reasonably close to column x
            if best_dx > 90:
                continue
            out[day][person] = best["val"]
            applied += 1

    debug = {
        "people_headers_found": {p: people_x[p]["raw"] for p in sorted(people_x.keys())},
        "dates_found": len(dates),
        "numbers_found": len(nums),
        "cells_applied": applied,
    }
    return out, debug

# =========================================================
# CSV PARSING (BEST)
# =========================================================
def extract_csv_to_monfri(csv_bytes, known_people, week_start: date):
    """
    CSV should have a date column and one column per person.
    We map by actual date -> Mon..Fri.
    """
    raw = csv_bytes.decode("utf-8", errors="ignore")

    # Try normal header first
    df = None
    try:
        df = pd.read_csv(BytesIO(raw.encode("utf-8")))
    except Exception:
        pass

    # Try 2-row header (sometimes exports include Count row)
    if df is None or df.shape[1] <= 1:
        try:
            df = pd.read_csv(BytesIO(raw.encode("utf-8")), header=[0, 1])
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        except Exception:
            return {}, {"error": "Could not read CSV. Try downloading the table as CSV again."}

    cols = list(df.columns)

    # find date column
    date_col = None
    for c in cols:
        if "date" in str(c).lower():
            date_col = c
            break
    if date_col is None:
        date_col = cols[0]  # fallback

    # normalize person columns
    person_cols = [c for c in cols if c != date_col]
    if not person_cols:
        return {}, {"error": "CSV has no person columns. Ensure you downloaded the data table, not an empty file."}

    def date_to_day(dval):
        try:
            if isinstance(dval, str):
                dt = datetime.strptime(dval.strip()[:10], "%Y-%m-%d").date()
            else:
                dt = pd.to_datetime(dval).date()
        except Exception:
            return None
        if dt < week_start or dt > (week_start + timedelta(days=4)):
            return None
        idx = dt.weekday()
        if idx < 0 or idx > 4:
            return None
        return DAYS[idx]

    out = {d: {} for d in DAYS}
    applied = 0

    # Build mapping of CSV col names -> app people names
    name_map = build_name_map(person_cols, known_people, min_score=80)

    for _, row in df.iterrows():
        day = date_to_day(row.get(date_col))
        if not day:
            continue
        for col in person_cols:
            target = name_map.get(col)
            if not target:
                continue
            v = row.get(col, "")
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s == "" or s in {"Ø", "ø"}:
                continue
            if re.match(r"^\d+$", s):
                out[day][target] = int(s)
                applied += 1

    debug = {"columns_mapped": name_map, "cells_applied": applied}
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

# =========================================================
# REPORT OUTPUTS
# =========================================================
def generate_report_txt(team: str, week_start: date, block: dict, display_people: list) -> str:
    metrics = block.get("metrics", {})
    actions = block.get("actions", {})
    deviations = block.get("deviations", {})

    lines = []
    lines.append(f"{team} — Weekly performance report — {week_label(week_start)}")
    lines.append(f"Week start (ISO): {iso(week_start)}")
    lines.append("")
    lines.append("Deviations adjust baseline fairly (Sick/AL/Reward time/Excluded = EXEMPT, Half day = 50%).")
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

                if dev in EXEMPT_DEVIATIONS:
                    parts.append(f"{d} - {val}/EXEMPT ({dev})" if val else f"{d} - EXEMPT ({dev})")
                    continue

                if base is None:
                    parts.append(f"{d} - {val}")
                    continue

                adj = base * mult
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
                    continue

                if v == "":
                    issues_red.append(f"Missing value for **{p}** — **{m}** — {d} (deviation: {dev}).")

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
        "note": "Backup includes people/baselines/reports/index. Uploaded files may be lost if Streamlit resets."
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
            row = {"Team": team, "week_iso": week_iso, "Person": person, "Metric": metric, "Baseline": md.get("Baseline", "")}
            for d in DAYS:
                row[d] = md.get(d, "")
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

# =========================================================
# UPLOAD INDEX HELPERS
# =========================================================
def week_items(idx, week_iso: str):
    return [x for x in idx if x.get("week_start") == week_iso]

def metric_items(items, metric: str):
    return [x for x in items if x.get("metric") == metric]

def latest_file_for(idx, week_iso, metric, kind):
    # kind = "csv" or "img"
    items = [x for x in idx if x.get("week_start") == week_iso and x.get("metric") == metric and x.get("kind") == kind]
    items = list(reversed(items))
    return items[0] if items else None

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
    st.subheader("Uploads")

    default_week = monday_of(date.today())
    week_start = monday_of(st.date_input("Week starting (Monday)", value=default_week, key="uploads_week"))
    w_iso = iso(week_start)

    people_state = load_people_state()
    known_people = people_state["active"] + people_state["archived"]

    idx = load_index()
    items = week_items(idx, w_iso)

    st.caption(f"Selected week: **{week_label(week_start)}**")
    st.divider()

    # show status
    cols = st.columns(3)
    for i, m in enumerate(METRICS):
        csv_count = sum(1 for x in metric_items(items, m) if x.get("kind") == "csv")
        img_count = sum(1 for x in metric_items(items, m) if x.get("kind") == "img")
        cols[i].metric(m, f"CSV: {csv_count} | Images: {img_count}")

    st.divider()
    st.markdown("### Upload a file for the selected metric")
    st.caption("Best: upload the CSV from the table (near-perfect). If not, upload the table screenshot.")

    metric = st.selectbox("Metric", METRICS, key="uploads_metric")

    c1, c2 = st.columns(2)
    with c1:
        up_csv = st.file_uploader("Upload CSV export (recommended)", type=["csv"], key=f"csv_{TEAM}_{w_iso}_{metric}")
    with c2:
        up_img = st.file_uploader("Upload TABLE screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"], key=f"img_{TEAM}_{w_iso}_{metric}")

    if st.button("Save upload for this metric", use_container_width=True, key=f"save_one_{TEAM}_{w_iso}_{metric}"):
        if not up_csv and not up_img:
            st.error("Upload a CSV or an image first.")
        else:
            idx = load_index()
            saved = []

            if up_csv:
                safe_name = f"{w_iso}_{metric}_table.csv".replace(" ", "_")
                out_path = P["UPLOADS"] / safe_name
                out_path.write_bytes(up_csv.getbuffer())
                idx.append({"week_start": w_iso, "metric": metric, "filename": safe_name, "kind": "csv"})
                saved.append("CSV")

            if up_img:
                safe_name = f"{w_iso}_{metric}_table.png".replace(" ", "_")
                out_path = P["UPLOADS"] / safe_name
                out_path.write_bytes(up_img.getbuffer())
                idx.append({"week_start": w_iso, "metric": metric, "filename": safe_name, "kind": "img"})
                saved.append("Image")

            save_index(idx)
            mark_dirty("upload saved")
            st.success(f"Saved: {', '.join(saved)} for {metric}.")
            st.rerun()

    st.divider()

    st.markdown("### ✅ One-click automation")
    st.caption("This will extract **Calls + EA Calls + Things Done** and apply them to the Weekly report for this week.")

    auto_exclude = st.checkbox(
        "Auto-mark people as EXCLUDED if they have no values in all extracted metrics",
        value=True,
        key=f"auto_excl_{TEAM}_{w_iso}"
    )

    if st.button("✨ Extract data & apply to weekly report", use_container_width=True, key=f"run_all_apply_{TEAM}_{w_iso}"):
        status = st.status("Working… extracting and applying data", expanded=True)

        reports = load_reports()
        ensure_week_structure(reports, w_iso, people_state["active"])
        block = reports[w_iso]

        idx = load_index()
        per_metric_summary = {}
        all_written_people = set()

        for m in METRICS:
            status.write(f"• {m}: locating latest upload…")
            rec_csv = latest_file_for(idx, w_iso, m, "csv")
            rec_img = latest_file_for(idx, w_iso, m, "img")

            extracted = None
            dbg = None

            if rec_csv:
                fp = P["UPLOADS"] / rec_csv["filename"]
                if file_exists(fp):
                    status.write(f"  - Using CSV: {rec_csv['filename']}")
                    extracted, dbg = extract_csv_to_monfri(fp.read_bytes(), known_people, week_start)
                else:
                    status.write("  - CSV file missing on disk (re-upload).")

            if extracted is None and rec_img:
                fp = P["UPLOADS"] / rec_img["filename"]
                if file_exists(fp):
                    status.write(f"  - Using image OCR: {rec_img['filename']}")
                    extracted, dbg = extract_table_screenshot_to_monfri(fp.read_bytes(), known_people, week_start)
                else:
                    status.write("  - Image file missing on disk (re-upload).")

            if extracted is None:
                per_metric_summary[m] = "No usable upload found"
                continue
            if dbg and "error" in dbg:
                per_metric_summary[m] = f"Error: {dbg['error']}"
                continue

            # Apply into weekly report
            metric_store = block["metrics"].get(m, {})
            wrote = 0

            # map extracted names -> week people names
            extracted_names = set()
            for d in DAYS:
                extracted_names |= set(extracted.get(d, {}).keys())
            name_map = build_name_map(list(extracted_names), list(block["people"]), min_score=85)

            for d in DAYS:
                for ep, val in extracted.get(d, {}).items():
                    tp = name_map.get(ep)
                    if not tp:
                        continue
                    metric_store.setdefault(tp, {"Baseline": "", **{day: "" for day in DAYS}})
                    metric_store[tp][d] = str(val)
                    wrote += 1
                    all_written_people.add(tp)

            block["metrics"][m] = metric_store
            per_metric_summary[m] = f"Applied {wrote} cells"
            status.write(f"  - Done. {per_metric_summary[m]}")

        # Auto-exclude (whole week) if no values anywhere
        if auto_exclude:
            no_values_people = [p for p in block["people"] if p not in all_written_people]
            if no_values_people:
                status.write(f"• Auto-excluding: {', '.join(no_values_people)}")
                block.setdefault("deviations", {})
                for p in no_values_people:
                    block["deviations"].setdefault(p, {d: "Normal" for d in DAYS})
                    for d in DAYS:
                        block["deviations"][p][d] = "Excluded"

        reports[w_iso] = block
        save_reports(reports)
        mark_dirty("extracted+applied")

        status.update(label="Done ✅ Data extracted & applied to Weekly report", state="complete", expanded=False)
        st.success("Finished. Summary:")
        st.json(per_metric_summary)
        st.info("Now go to **📝 Weekly report** and you should see the numbers filled in.")
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
        st.caption("Clears all saved numbers + TL actions for the selected week. (Does not delete uploads.)")
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
    st.markdown("### Mark as backed up")
    if st.button("I’ve backed up now", use_container_width=True, key="clear_dirty_btn"):
        clear_dirty()
        st.success("Backup reminder cleared.")

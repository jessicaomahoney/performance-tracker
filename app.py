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


# =========================================================
# CSV PARSING (recommended)
# =========================================================

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _safe_int(x):
    try:
        if pd.isna(x):
            return None
        s = str(x).strip()
        if s in ("", "Ø", "ø", "0/", "-"):
            return None
        # strip commas
        s = s.replace(",", "")
        if re.fullmatch(r"\d+", s):
            return int(s)
        # sometimes "0" should be 0
        if re.fullmatch(r"\d+\.0", s):
            return int(float(s))
        return None
    except Exception:
        return None


def parse_csv_to_monfri(df: pd.DataFrame, week_start: date, known_people: list):
    """
    Accepts several CSV shapes:

    1) Wide per metric table:
       Person,Baseline,Mon,Tue,Wed,Thu,Fri

    2) Our exported long-ish shape:
       Team,week_iso,Person,Metric,Baseline,Mon,Tue,Wed,Thu,Fri

    3) Date-based table:
       Completed Date,<person columns...>
       (or any date column) + numeric columns

    Returns:
      out: {Mon:{person:val}, ...}
      dbg
    """
    out = {d: {} for d in DAYS}
    dbg = {"mode": "csv", "rows": int(df.shape[0]), "cols": int(df.shape[1])}

    # Normalize column names
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    cols_lower = {c.lower(): c for c in df2.columns}

    # Case 2: includes Metric column
    if "metric" in cols_lower and "person" in cols_lower and "mon" in cols_lower:
        # Filter should be done outside per metric, but we can still parse to Mon-Fri
        for _, r in df2.iterrows():
            person = str(r[cols_lower["person"]]).strip()
            if not person:
                continue
            # Map to known people if needed
            if known_people:
                m = build_name_map([person], known_people, min_score=80)
                person = m.get(person, person)

            for d in DAYS:
                v = _safe_int(r.get(d, r.get(d.lower(), None)))
                if v is None:
                    continue
                out[d][person] = v
        dbg["shape"] = "team_week_metric"
        return out, dbg

    # Case 1: simple wide
    if "person" in cols_lower and "mon" in cols_lower:
        for _, r in df2.iterrows():
            person = str(r[cols_lower["person"]]).strip()
            if not person:
                continue
            if known_people:
                m = build_name_map([person], known_people, min_score=80)
                person = m.get(person, person)

            for d in DAYS:
                v = _safe_int(r.get(d))
                if v is None:
                    continue
                out[d][person] = v
        dbg["shape"] = "person_monfri_wide"
        return out, dbg

    # Case 3: date column
    date_col = None
    for c in df2.columns:
        cl = c.lower()
        if "date" in cl:
            date_col = c
            break

    if date_col:
        # Coerce date column
        dates = pd.to_datetime(df2[date_col], errors="coerce")
        df2 = df2.assign(__date=dates)

        start = pd.Timestamp(week_start)
        end = pd.Timestamp(week_start + timedelta(days=6))
        dfw = df2[(df2["__date"] >= start) & (df2["__date"] <= end)].copy()

        if dfw.empty:
            dbg["shape"] = "date_table_no_rows_in_week"
            dbg["error"] = "No rows in this week range."
            return out, dbg

        # Build weekday mapping
        for _, r in dfw.iterrows():
            dt = r["__date"]
            if pd.isna(dt):
                continue
            weekday = dt.weekday()  # Mon=0
            if weekday < 0 or weekday > 4:
                continue
            day = DAYS[weekday]

            # For each person column, if numeric then assign
            for c in dfw.columns:
                if c in (date_col, "__date"):
                    continue
                # skip obvious non-data
                if str(c).lower() in ("baseline", "metric", "team", "week_iso"):
                    continue

                v = _safe_int(r.get(c))
                if v is None:
                    continue

                # Column name might include "Count" or similar
                person_guess = str(c)
                person_guess = re.sub(r"\bcount\b", "", person_guess, flags=re.I).strip()

                if known_people:
                    m = build_name_map([person_guess], known_people, min_score=75)
                    person = m.get(person_guess)
                    if not person:
                        continue
                else:
                    person = person_guess

                out[day][person] = v

        dbg["shape"] = "date_wide_person_cols"
        return out, dbg

    dbg["shape"] = "unknown"
    dbg["error"] = "Couldn’t understand that CSV format. Try exporting with Person/Baseline/Mon–Fri, or include a Date column." 
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
    pw = st.text_input("Password", type="password", key="login_pw")

    if pw == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    elif pw:
        st.error("Nope — that password doesn’t match.")

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

    st.title("👋 Which team are you working on?")
    team = st.radio("", ["North", "South"], horizontal=True, key="team_pick")
    if st.button("➡️ Continue", use_container_width=True, key="team_continue"):
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
    # IMPORTANT: do NOT auto-overwrite saved people.
    return {"active": [], "archived": []}


def load_people_state():
    raw = load_json(P["PEOPLE"], default_people_state())

    # If some old version saved a list, upgrade safely
    if isinstance(raw, list):
        raw = {"active": raw, "archived": []}

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

    # Remove duplicates across lists
    active_set = set(raw["active"])
    raw["archived"] = [x for x in raw["archived"] if x not in active_set]

    # Persist file if missing (prevents "vanish" when Streamlit reruns)
    if not P["PEOPLE"].exists():
        save_json(P["PEOPLE"], raw)

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


def load_meta():
    return load_json(P["META"], {"last_backup_at": ""})


def save_meta(m):
    save_json(P["META"], m)


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
    lines.append("📌 Deviations adjust baselines fairly (Sick/AL/Reward time = EXEMPT, Half day = 50%).")
    lines.append("This report is designed to stand alone (no spreadsheet context needed).")
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
                    parts.append(f"{d} — {val}")
                    continue

                adj = base * mult
                if adj == 0:
                    parts.append(f"{d} — {val}/EXEMPT ({dev})")
                else:
                    adj_show = str(int(adj)) if float(adj).is_integer() else str(adj)
                    parts.append(f"{d} — {val}/{adj_show} ({dev})")

            lines.append(f"{metric}: " + " | ".join(parts))

        lines.append("")
        lines.append("🧭 TL action (what I will do about it):")
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
# UPLOAD INDEX HELPERS
# =========================================================

def week_items(idx, week_iso: str):
    return [x for x in idx if x.get("week_start") == week_iso]


def metric_items(items, metric: str):
    return [x for x in items if x.get("metric") == metric]


def file_exists(filename: str) -> bool:
    return (P["UPLOADS"] / filename).exists()


# =========================================================
# CHECKS
# =========================================================

def run_checks(block: dict, display_people: list, active_set: set, baselines: dict, week_iso: str):
    issues_red, issues_amber = [], []

    # TL actions
    for p in display_people:
        act = (block.get("actions", {}).get(p, "") or "").strip()
        if not act:
            issues_red.append(f"🧭 TL action missing for **{p}**.")

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
                issues_red.append(f"🎯 Baseline missing for **{p}** — **{m}**.")
            elif to_float(used_base) is None:
                issues_amber.append(f"⚠️ Baseline is not a number for **{p}** — **{m}** (value: `{used_base}`).")

            for d in DAYS:
                dev = p_dev.get(d, "Normal")
                v = str(md.get(d, "")).strip()

                if dev in EXEMPT_DEVIATIONS:
                    # If exempt, leaving blank is fine.
                    continue

                if v == "":
                    issues_red.append(f"📅 Missing value for **{p}** — **{m}** — {d} (deviation: {dev}).")

    idx = load_index()
    week_uploads = week_items(idx, week_iso)
    for m in METRICS:
        cnt = sum(1 for x in metric_items(week_uploads, m) if x.get("filename") and file_exists(x["filename"]))
        if cnt == 0:
            issues_amber.append(f"📎 No upload/CSV for **{m}** this week (optional).")

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
    meta = load_json(tp["META"], {"last_backup_at": ""})

    return {
        "team": team,
        "people": people,
        "baselines": baselines,
        "reports": reports,
        "uploads_index": idx,
        "meta": meta,
        "note": "Backup includes people/baselines/reports/index/meta. Uploaded files are NOT included (Streamlit Cloud may wipe them).",
    }


def restore_team_backup(backup_obj: dict, target_team: str):
    tp = team_paths(target_team)
    tp["BASE"].mkdir(parents=True, exist_ok=True)
    tp["UPLOADS"].mkdir(parents=True, exist_ok=True)

    save_json(tp["PEOPLE"], backup_obj.get("people", {"active": [], "archived": []}))
    save_json(tp["BASELINES"], backup_obj.get("baselines", {}))
    save_json(tp["REPORTS"], backup_obj.get("reports", {}))
    save_json(tp["INDEX"], backup_obj.get("uploads_index", []))
    save_json(tp["META"], backup_obj.get("meta", {"last_backup_at": ""}))


def export_week_wide_csv(team: str, week_iso: str) -> bytes:
    tp = team_paths(team)
    reports = load_json(tp["REPORTS"], {})
    block = reports.get(week_iso)

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
# APPLY HELPERS
# =========================================================

def apply_extracted_to_week(reports: dict, week_iso: str, metric: str, extracted: dict, known_people: list):
    """Write extracted {Mon:{name:val}} into reports[week_iso] for the given metric."""
    block = reports[week_iso]
    metric_store = block["metrics"].get(metric, {})

    # map extracted names -> week people
    extracted_names = set()
    for d in DAYS:
        extracted_names |= set(extracted.get(d, {}).keys())

    name_map = build_name_map(list(extracted_names), list(block["people"]), min_score=80)

    wrote = 0
    for d in DAYS:
        for ep, val in extracted.get(d, {}).items():
            tp = name_map.get(ep)
            if not tp:
                continue
            metric_store.setdefault(tp, {"Baseline": "", **{dd: "" for dd in DAYS}})
            metric_store[tp][d] = "" if val is None else str(val)
            wrote += 1

    block["metrics"][metric] = metric_store
    reports[week_iso] = block
    return wrote, name_map


# =========================================================
# UI
# =========================================================

st.title(f"📊 Team Performance Tracker — {TEAM}")

meta = load_meta()
last_backup = meta.get("last_backup_at", "")

st.info(
    "👋 Hey! This app saves to a small on-server folder — but Streamlit Cloud can sometimes reset it.\n\n"
    "✅ Make your life easy: use **💾 Backup & Export** every week (it’s 2 clicks)."
)

if st.session_state.get("dirty", False):
    st.warning(f"🟦 Quick safety nudge: please download a backup (reason: {st.session_state.get('dirty_reason','changes made')}).")

if not last_backup:
    st.warning("💾 I can’t see a ‘last backup’ timestamp yet. Once you download a backup, we’ll show it here.")
else:
    st.caption(f"💾 Last backup recorded: **{last_backup}**")

with st.sidebar:
    st.subheader("🧭 Navigation")
    st.write(f"Team: **{TEAM}**")

    if st.button("🔁 Switch team", use_container_width=True, key="switch_team_btn"):
        st.session_state.team = None
        st.rerun()

    if st.button("🚪 Log out", use_container_width=True, key="logout_btn"):
        st.session_state.authenticated = False
        st.session_state.team = None
        st.rerun()

    st.divider()
    st.markdown("### 🔗 Quick links")
    st.markdown(f"📁 [Backup folder]({BACKUP_FOLDER_URL})")
    st.markdown(f"📊 [Performance tracker]({TRACKER_SHEET_URL})")


# Load people early + show a gentle prompt if empty
people_state = load_people_state()
if not people_state.get("active") and not people_state.get("archived"):
    st.warning("👥 Your People list is empty right now. Pop into **🎯 Baselines → People management** to add people (or restore from backup).")


tab_uploads, tab_baselines, tab_deviations, tab_report, tab_backup = st.tabs(
    ["📥 Uploads", "🎯 Baselines", "🗓️ Deviations", "📝 Weekly report", "💾 Backup & Export"]
)


# =========================================================
# TAB: UPLOADS
# =========================================================
with tab_uploads:
    st.subheader("📥 Uploads")
    st.caption("Upload either **screenshots** *or* **CSVs**. Then hit one button to extract + apply into the Weekly report.")

    default_week = monday_of(date.today())
    week_start = monday_of(st.date_input("Week starting (Monday)", value=default_week, key="uploads_week"))
    w_iso = iso(week_start)

    idx = load_index()
    items = week_items(idx, w_iso)

    st.caption(f"🗓️ Selected week: **{week_label(week_start)}**")
    st.divider()

    # Summary tiles
    cols = st.columns(3)
    for i, m in enumerate(METRICS):
        mi = metric_items(items, m)
        count_files = sum(1 for x in mi if x.get("filename") and file_exists(x["filename"]))
        cols[i].metric(m, f"{count_files} file(s)")

    st.divider()

    st.markdown("### 1) Add uploads")
    metric = st.selectbox("Which metric is this for?", METRICS, key="uploads_metric")

    up_img = st.file_uploader(
        "📸 Upload a screenshot (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="uploads_img",
    )

    up_csv = st.file_uploader(
        "📄 Upload a CSV (recommended)",
        type=["csv"],
        accept_multiple_files=True,
        key="uploads_csv",
    )

    if st.button("✅ Save uploads", use_container_width=True, key="save_uploads_btn"):
        if not up_img and not up_csv:
            st.error("Please upload at least one screenshot or CSV.")
        else:
            idx = load_index()
            saved = 0

            # Save screenshots
            for f in (up_img or []):
                safe_name = f"{w_iso}_{metric}_image_{f.name}".replace(" ", "_")
                (P["UPLOADS"] / safe_name).write_bytes(f.getbuffer())
                idx.append({"week_start": w_iso, "metric": metric, "filename": safe_name, "kind": "image"})
                saved += 1

            # Save CSVs
            for f in (up_csv or []):
                safe_name = f"{w_iso}_{metric}_csv_{f.name}".replace(" ", "_")
                (P["UPLOADS"] / safe_name).write_bytes(f.getbuffer())
                idx.append({"week_start": w_iso, "metric": metric, "filename": safe_name, "kind": "csv"})
                saved += 1
            save_index(idx)
            mark_dirty("uploads saved")
            st.success(f"🎉 Saved {saved} file(s) for **{metric}**.")
            st.rerun()

    st.divider()
    st.markdown("### 2) Review + delete uploads")

    idx = load_index()
    items = week_items(idx, w_iso)

    if not items:
        st.info("No uploads saved for this week yet.")
    else:
        # Show per metric
        for m in METRICS:
            mi = metric_items(items, m)
            if not mi:
                continue
            st.markdown(f"#### {m}")
            for it in reversed(mi[-10:]):
                pth = P["UPLOADS"] / it["filename"]
                if not pth.exists():
                    st.warning(f"Missing file on disk: {it['filename']} (Streamlit may have reset).")
                    continue
                if it.get("kind") == "csv" or pth.suffix.lower() == ".csv":
                    st.caption(f"📄 CSV: {it['filename']}")
                else:
                    st.caption(f"📸 Image: {it['filename']}")
                    st.image(str(pth), width=900)
            st.divider()

        # Delete UI
        options = []
        for i, it in enumerate(items):
            options.append(f"{i} | {it.get('metric')} | {it.get('kind','?')} | {it.get('filename')}")

        to_delete = st.multiselect("🗑️ Select uploads to delete", options=options, key=f"del_{TEAM}_{w_iso}")
        if st.button("🗑️ Delete selected uploads", use_container_width=True, key=f"del_btn_{TEAM}_{w_iso}"):
            delete_set = set(to_delete)
            new_items = []
            for i, it in enumerate(items):
                label = f"{i} | {it.get('metric')} | {it.get('kind','?')} | {it.get('filename')}"
                if label in delete_set:
                    fp = P["UPLOADS"] / it.get("filename", "")
                    try:
                        if fp.exists():
                            fp.unlink()
                    except Exception:
                        pass
                else:
                    new_items.append(it)

            new_idx = [x for x in idx if x.get("week_start") != w_iso] + new_items
            save_index(new_idx)
            mark_dirty("uploads deleted")
            st.success("✅ Deleted selected uploads.")
            st.rerun()

    st.divider()
    st.markdown("### 3) ✨ Extract data and apply to Weekly report")
    st.caption("One button. All metrics. We’ll use CSVs first (best), and screenshots as fallback.")

    # OCR controls (only used if screenshots are used)
    with st.expander("⚙️ Screenshot extraction settings (only if needed)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            max_w = st.selectbox("OCR max width", [1200, 1400, 1600, 1800], index=3, key=f"ocr_maxw_{TEAM}_{w_iso}")
        with c2:
            dist_thr = st.selectbox("Colour match strictness", [90, 110, 120, 140, 160], index=3, key=f"ocr_thr_{TEAM}_{w_iso}")

        st.caption("Tip: if screenshots are blurry, 1800 + 160 can help — but CSV is always best.")

    if st.button("✨ Extract data and apply to Weekly report", use_container_width=True, key=f"run_all_apply_{TEAM}_{w_iso}"):
        people_state = load_people_state()
        known_people = people_state.get("active", []) + people_state.get("archived", [])

        reports = load_reports()
        ensure_week_structure(reports, w_iso, people_state.get("active", []))

        idx = load_index()
        items = week_items(idx, w_iso)

        # progress UI
        use_status = hasattr(st, "status")
        status_ctx = None
        if use_status:
            status_ctx = st.status("⏳ Working on it… extracting and applying", expanded=True)

        def _log(msg):
            if use_status and status_ctx is not None:
                status_ctx.write(msg)
            else:
                st.write(msg)

        summary = {}
        overall_wrote = 0

        for m in METRICS:
            _log(f"🔎 **{m}** — looking for files…")

            mi = metric_items(items, m)
            # Prefer latest CSV, else latest image
            latest_csv = next((x for x in reversed(mi) if x.get("kind") == "csv" or str(x.get("filename", "")).lower().endswith(".csv")), None)
            latest_img = next((x for x in reversed(mi) if x.get("kind") == "image"), None)

            extracted = {d: {} for d in DAYS}
            dbg = {}

            if latest_csv:
                fp = P["UPLOADS"] / latest_csv["filename"]
                if fp.exists():
                    _log("📄 Using CSV (best).")
                    try:
                        df = pd.read_csv(fp)
                        extracted, dbg = parse_csv_to_monfri(df, week_start=week_start, known_people=known_people)
                    except Exception as e:
                        dbg = {"error": f"CSV read failed: {e}"}
                else:
                    dbg = {"error": "CSV file missing on disk (Streamlit reset?)"}

            elif latest_img:
                fp = P["UPLOADS"] / latest_img["filename"]
                if fp.exists():
                    _log("📸 Using screenshot fallback (beta).")
                    extracted, dbg = extract_stacked_chart_to_monfri(
                        fp.read_bytes(),
                        known_people,
                        max_w=int(max_w),
                        color_dist_threshold=int(dist_thr),
                    )
                else:
                    dbg = {"error": "Image file missing on disk (Streamlit reset?)"}
            else:
                summary[m] = "No files uploaded"
                _log("⚠️ No files found for this metric.")
                continue

            if dbg.get("error"):
                summary[m] = f"❌ {dbg['error']}"
                _log(f"❌ {dbg['error']}")
                continue

            wrote, name_map = apply_extracted_to_week(reports, w_iso, m, extracted, known_people)
            overall_wrote += wrote

            # Helpful diagnostics
            filled_people = sorted({p for d in DAYS for p in extracted.get(d, {}).keys()})
            summary[m] = {
                "✅ wrote_cells": wrote,
                "👥 extracted_people": filled_people,
                "🔁 name_map": name_map,
                "🧪 debug": dbg,
            }
            _log(f"✅ Applied **{m}** — wrote **{wrote}** cells.")

        save_reports(reports)
        mark_dirty("extracted + applied")

        if use_status and status_ctx is not None:
            status_ctx.update(label="✅ Done! Extraction + apply finished", state="complete", expanded=False)

        st.success(f"🎉 All done! I applied **{overall_wrote}** cells into the Weekly report for **{week_label(week_start)}**.")
        st.caption("If you expected more values: check the summary below — especially the name_map.")
        st.json(summary)
        st.rerun()


# =========================================================
# TAB: BASELINES
# =========================================================
with tab_baselines:
    st.subheader("🎯 Baselines")
    st.caption("Set baselines per person. These auto-fill into the Weekly report (you can override per week if needed).")

    people_state = load_people_state()
    active = people_state["active"]
    archived = people_state["archived"]
    baselines = load_baselines()

    if not active:
        st.info("No active people yet. Add them below 👇")
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

        if st.button("✅ Save baselines", use_container_width=True, key=f"save_baselines_{TEAM}"):
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
            st.success("🎉 Baselines saved!")

    st.divider()

    with st.expander("👥 People management (add / archive / restore)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ✅ Active")
            st.markdown("\n".join([f"- {p}" for p in active]) if active else "—")
        with c2:
            st.markdown("### 🗄️ Archived")
            st.markdown("\n".join([f"- {p}" for p in archived]) if archived else "—")

        st.divider()
        new_name = st.text_input("Add person name", key=f"add_person_{TEAM}").strip()
        if st.button("➕ Add to Active", use_container_width=True, key=f"add_person_btn_{TEAM}"):
            if not new_name:
                st.error("Type a name first 🙂")
            else:
                if new_name in archived:
                    archived.remove(new_name)
                if new_name not in active:
                    active.append(new_name)
                save_people_state({"active": active, "archived": archived})
                mark_dirty("people changed")
                st.success(f"✅ Added **{new_name}**")
                st.rerun()

        st.divider()
        to_archive = st.multiselect("Archive selected", options=active, key=f"archive_select_{TEAM}")
        if st.button("🗄️ Archive", use_container_width=True, key=f"archive_btn_{TEAM}"):
            for p in to_archive:
                if p in active:
                    active.remove(p)
                if p not in archived:
                    archived.append(p)
            save_people_state({"active": active, "archived": archived})
            mark_dirty("people changed")
            st.success("✅ Archived.")
            st.rerun()

        st.divider()
        to_restore = st.multiselect("Restore selected", options=archived, key=f"restore_select_{TEAM}")
        if st.button("♻️ Restore", use_container_width=True, key=f"restore_btn_{TEAM}"):
            for p in to_restore:
                if p in archived:
                    archived.remove(p)
                if p not in active:
                    active.append(p)
            save_people_state({"active": active, "archived": archived})
            mark_dirty("people changed")
            st.success("✅ Restored.")
            st.rerun()


# =========================================================
# TAB: DEVIATIONS
# =========================================================
with tab_deviations:
    st.subheader("🗓️ Deviations")
    st.caption("Use this for sick days, half days etc — it adjusts baselines fairly in the report text.")

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

    st.caption(f"🗓️ Selected week: **{week_label(dev_week)}**")

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
        key=f"deviations_editor_{TEAM}_{dev_week_iso}_{'all' if show_archived else 'active'}",
    )

    if st.button("✅ Save deviations", use_container_width=True, key=f"save_dev_{TEAM}_{dev_week_iso}_{'all' if show_archived else 'active'}"):
        reports = load_reports()
        ensure_week_structure(reports, dev_week_iso, active)
        stored = reports[dev_week_iso].get("deviations", {})
        for _, r in edited.iterrows():
            person = str(r["Person"]).strip()
            stored[person] = {d: r[d] for d in DAYS}
        reports[dev_week_iso]["deviations"] = stored
        save_reports(reports)
        mark_dirty("deviations saved")
        st.success("🎉 Deviations saved!")


# =========================================================
# TAB: WEEKLY REPORT
# =========================================================
with tab_report:
    st.subheader("📝 Weekly report")
    st.caption("This is your ‘send-to-Jess’ view — designed to stand alone.")

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

    st.caption(f"🗓️ Selected week: **{week_label(rep_week)}**")

    if not display_people:
        st.info("No active people to show. Add/restore people in Baselines.")
        st.stop()

    # Reset week
    with st.expander("🧨 Reset this week (if things got messy)", expanded=False):
        st.caption("This clears the numbers + TL actions for this week. It does NOT delete uploads.")
        confirm = st.checkbox("Yes, reset this week", key=f"reset_confirm_{TEAM}_{rep_week_iso}")
        if st.button("🧼 Reset week now", use_container_width=True, key=f"reset_week_{TEAM}_{rep_week_iso}", disabled=not confirm):
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
            st.success("✅ Week reset complete.")
            st.rerun()

    st.divider()
    st.write("👇 You can type values manually, or use **📥 Uploads → Extract data and apply** to fill these automatically.")

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

        if st.button(f"✅ Save {metric}", use_container_width=True, key=f"save_metric_{TEAM}_{rep_week_iso}_{metric}_{'all' if show_archived else 'active'}"):
            reports = load_reports()
            ensure_week_structure(reports, rep_week_iso, active)
            stored = reports[rep_week_iso]["metrics"].get(metric, {})
            for _, r in edited.iterrows():
                person = str(r["Person"]).strip()
                stored[person] = {"Baseline": r.get("Baseline", ""), **{d: r.get(d, "") for d in DAYS}}
            reports[rep_week_iso]["metrics"][metric] = stored
            save_reports(reports)
            mark_dirty(f"{metric} saved")
            st.success(f"🎉 Saved {metric}.")
            st.rerun()

        st.divider()

    st.subheader("🧭 TL actions (required)")
    st.caption("Please write what you’ll do about performance — this is the accountability bit.")

    for p in display_people:
        st.text_area(
            f"{p} — what will you do about it?",
            value=block.get("actions", {}).get(p, ""),
            height=90,
            key=f"action_{TEAM}_{rep_week_iso}_{p}_{'all' if show_archived else 'active'}",
        )

    if st.button("✅ Save TL actions", use_container_width=True, key=f"save_actions_{TEAM}_{rep_week_iso}_{'all' if show_archived else 'active'}"):
        reports = load_reports()
        ensure_week_structure(reports, rep_week_iso, active)
        for p in display_people:
            k = f"action_{TEAM}_{rep_week_iso}_{p}_{'all' if show_archived else 'active'}"
            reports[rep_week_iso]["actions"][p] = st.session_state.get(k, "")
        save_reports(reports)
        mark_dirty("TL actions saved")
        st.success("🎉 TL actions saved!")

    st.divider()

    reports = load_reports()
    block = reports[rep_week_iso]

    report_txt = generate_report_txt(TEAM, rep_week, block, display_people)
    report_tsv = generate_report_tsv(block, display_people)

    status, issues = run_checks(block, display_people, active_set, baselines, rep_week_iso)

    st.subheader("✅ Readiness checks")
    if status == "green":
        st.success("🟢 Ready to send — nice.")
    elif status == "amber":
        st.warning("🟠 Nearly there — a few bits to tidy up.")
    else:
        st.error("🔴 Not ready yet — please fix the red items.")

    if issues:
        st.markdown("**What needs attention:**")
        for it in issues:
            st.markdown(f"- {it}")

    st.divider()

    st.subheader("✅ Weekly checklist (2 minutes)")
    st.markdown("**1)** Download the backup JSON and save it to the backup folder")
    st.markdown(f"📁 [Open backup folder]({BACKUP_FOLDER_URL})")
    st.markdown("**2)** Download the Weekly WIDE CSV and import into the tracker")
    st.markdown(f"📊 [Open performance tracker]({TRACKER_SHEET_URL})")

    cA, cB = st.columns(2)
    with cA:
        cur_backup = build_team_backup(TEAM)
        st.download_button(
            "💾 Download CURRENT backup.json",
            data=json.dumps(cur_backup, indent=2).encode("utf-8"),
            file_name=f"{TEAM.lower()}_backup.json",
            mime="application/json",
            use_container_width=True,
            disabled=(status != "green"),
            key=f"dl_weekly_backup_{TEAM}_{rep_week_iso}",
        )

    with cB:
        st.download_button(
            "📤 Download Weekly WIDE CSV",
            data=export_week_wide_csv(TEAM, rep_week_iso),
            file_name=f"{TEAM.lower()}_weekly_wide_{rep_week_iso}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=(status != "green"),
            key=f"dl_weekly_wide_{TEAM}_{rep_week_iso}",
        )

    st.caption("Google Sheets import: File → Import → Upload → choose the CSV.")

    if st.button("✅ Mark backup done (records timestamp)", use_container_width=True, disabled=(status != "green"), key=f"mark_backup_{TEAM}_{rep_week_iso}"):
        m = load_meta()
        m["last_backup_at"] = now_stamp()
        save_meta(m)
        clear_dirty()
        st.success("✅ Nice — backup timestamp recorded.")
        st.rerun()

    st.divider()

    st.subheader("📋 Report preview (copy/paste)")
    st.caption("Tip: click inside → Ctrl+A → Ctrl+C")
    st.text_area("Weekly report text", value=report_txt, height=520, key=f"preview_{TEAM}_{rep_week_iso}")

    st.divider()
    st.subheader("⬇️ Other downloads")
    st.download_button(
        "📝 Download weekly report (TXT)",
        data=report_txt.encode("utf-8"),
        file_name=f"{TEAM.lower()}_weekly_report_{rep_week_iso}.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.download_button(
        "📎 Download weekly data (TSV)",
        data=report_tsv.encode("utf-8"),
        file_name=f"{TEAM.lower()}_weekly_data_{rep_week_iso}.tsv",
        mime="text/tab-separated-values",
        use_container_width=True,
    )


# =========================================================
# TAB: BACKUP & EXPORT
# =========================================================
with tab_backup:
    st.subheader("💾 Backup & Export")
    st.caption("This is your safety net. Use it weekly (or after big edits).")

    st.divider()
    st.markdown("### 💾 Download backups")

    c1, c2, c3 = st.columns(3)
    with c1:
        b = build_team_backup("North")
        st.download_button(
            "⬇️ Download NORTH backup.json",
            data=json.dumps(b, indent=2).encode("utf-8"),
            file_name="north_backup.json",
            mime="application/json",
            use_container_width=True,
            key="dl_backup_north",
        )
    with c2:
        b = build_team_backup("South")
        st.download_button(
            "⬇️ Download SOUTH backup.json",
            data=json.dumps(b, indent=2).encode("utf-8"),
            file_name="south_backup.json",
            mime="application/json",
            use_container_width=True,
            key="dl_backup_south",
        )
    with c3:
        b = build_team_backup(TEAM)
        st.download_button(
            f"⬇️ Download CURRENT ({TEAM}) backup.json",
            data=json.dumps(b, indent=2).encode("utf-8"),
            file_name=f"{TEAM.lower()}_backup.json",
            mime="application/json",
            use_container_width=True,
            key="dl_backup_current",
        )

    st.info("📌 Backups include People, Baselines, Reports, and Upload metadata. Upload files themselves aren’t included.")

    st.divider()
    st.markdown("### ♻️ Restore from a backup")
    up = st.file_uploader("Upload backup JSON", type=["json"], key="restore_upload")
    target_team = st.radio("Restore into team", ["North", "South"], horizontal=True, key="restore_target_team")

    if st.button("♻️ Restore NOW (overwrite)", use_container_width=True, key="restore_now_btn"):
        if not up:
            st.error("Upload a backup JSON first.")
        else:
            try:
                data = json.loads(up.getvalue().decode("utf-8"))
                restore_team_backup(data, target_team)
                mark_dirty("restored from backup")
                st.success(f"✅ Restored backup into {target_team}.")
                st.info("Tip: use the sidebar to switch team and check it.")
            except Exception as e:
                st.error(f"Restore failed: {e}")

    st.divider()
    st.markdown("### 🧼 Clear backup reminder")
    if st.button("✅ I’ve backed up now", use_container_width=True, key="clear_dirty_btn"):
        clear_dirty()
        m = load_meta()
        m["last_backup_at"] = now_stamp()
        save_meta(m)
        st.success("✅ Backup reminder cleared + timestamp recorded.")

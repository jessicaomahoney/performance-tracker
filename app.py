import streamlit as st
from datetime import date, timedelta
from pathlib import Path
import json
import pandas as pd
import re
import numpy as np
from PIL import Image
import cv2
import easyocr
from rapidfuzz import process, fuzz
from io import BytesIO

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
EXEMPT_DEVIATIONS = {"Sick", "Annual leave", "Reward time"}  # baseline becomes 0

APP_PASSWORD = st.secrets["APP_PASSWORD"] if "APP_PASSWORD" in st.secrets else "password"

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
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
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

def extract_stacked_chart(image_bytes, known_people):
    """
    Strict extractor:
    - reads date labels (e.g., Jan 5)
    - reads legend names and samples their dot colours
    - reads numbers inside segments and assigns them by colour match
    If uncertain: skips (no guessing)
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    reader = get_ocr_reader()
    ocr = reader.readtext(np.array(img))  # [ (bbox, text, conf), ... ]

    # Find date labels like "Jan 5"
    date_re = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{1,2}$", re.I)
    dates = []
    for bbox, text, conf in ocr:
        t = text.strip()
        if conf >= 0.5 and date_re.match(t):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            cx = int(sum(xs) / 4)
            cy = int(sum(ys) / 4)
            dates.append((cx, cy, t.title()))

    dates = sorted(dates, key=lambda x: x[0])
    if len(dates) < 2:
        return {}, {"error": "Could not detect date labels reliably."}

    # Legend mapping: fuzzy-match names + sample dot colour just left of text
    legend = {}  # person -> sampled colour (BGR)
    for bbox, text, conf in ocr:
        t = text.strip()
        if conf < 0.45 or len(t) < 3:
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

        # dot is usually just left of the legend text
        sample_x = max(0, left - 18)
        color = _avg_color(img_bgr, sample_x, cy, r=8)
        legend[person] = color

    # Numeric labels inside segments
    num_re = re.compile(r"^\d+$")
    numeric = []
    for bbox, text, conf in ocr:
        t = text.strip()
        if conf < 0.45:
            continue
        if not num_re.match(t):
            continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = int(sum(xs) / 4)
        cy = int(sum(ys) / 4)
        val = int(t)

        col = _avg_color(img_bgr, cx, cy, r=6)
        numeric.append((cx, cy, val, conf, col))

    extracted = {d[2]: {} for d in dates}

    for cx, cy, val, conf, col in numeric:
        nearest_date = min(dates, key=lambda d: abs(cx - d[0]))[2]

        if not legend:
            continue

        best_person = None
        best_dist = 1e9
        for person, lcol in legend.items():
            dist = _color_distance(col, lcol)
            if dist < best_dist:
                best_dist = dist
                best_person = person

        # strict threshold — tune later if needed
        if best_person is None or best_dist > 70:
            continue

        extracted[nearest_date][best_person] = val

    debug = {
        "dates_detected": [d[2] for d in dates],
        "legend_people_detected": sorted(list(legend.keys())),
    }
    return extracted, debug

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

# =========================================================
# STORAGE PATHS (PER TEAM)
# =========================================================
BASE_DIR = Path("data") / TEAM
UPLOADS_DIR = BASE_DIR / "uploads"
INDEX_FILE = BASE_DIR / "index.json"          # uploads index
PEOPLE_FILE = BASE_DIR / "people.json"        # active/archived
BASELINES_FILE = BASE_DIR / "baselines.json"  # baselines per person/metric
REPORTS_FILE = BASE_DIR / "reports.json"      # weekly entries + deviations + actions

BASE_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# PEOPLE (ACTIVE/ARCHIVED) + MIGRATION
# =========================================================
def default_people_state():
    return {"active": ["Rebecca", "Nicole", "Sonia"], "archived": []}

def load_people_state():
    raw = load_json(PEOPLE_FILE, default_people_state())

    # migration if old format was a list
    if isinstance(raw, list):
        raw = {"active": raw, "archived": []}
        save_json(PEOPLE_FILE, raw)

    if not isinstance(raw, dict):
        raw = default_people_state()

    raw.setdefault("active", [])
    raw.setdefault("archived", [])

    def clean(lst):
        out = []
        seen = set()
        for x in lst:
            name = str(x).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    raw["active"] = clean(raw["active"])
    raw["archived"] = clean(raw["archived"])

    # prevent overlap (active wins)
    active_set = set(raw["active"])
    raw["archived"] = [x for x in raw["archived"] if x not in active_set]

    return raw

def save_people_state(state):
    state["active"] = sorted(state.get("active", []), key=lambda x: x.lower())
    state["archived"] = sorted(state.get("archived", []), key=lambda x: x.lower())
    save_json(PEOPLE_FILE, state)

# =========================================================
# REPORT STRUCTURE
# =========================================================
def load_reports():
    return load_json(REPORTS_FILE, {})

def save_reports(r):
    save_json(REPORTS_FILE, r)

def ensure_week_structure(reports: dict, week_iso: str, active_people: list):
    """
    - Create week if missing
    - Sync new ACTIVE people into week snapshot (never removes historical names)
    - Ensure metric/action/deviation structures exist
    """
    if week_iso not in reports:
        reports[week_iso] = {
            "people": list(active_people),
            "metrics": {},
            "actions": {},
            "deviations": {}
        }
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

# =========================================================
# UPLOADS INDEX
# =========================================================
def load_index():
    return load_json(INDEX_FILE, [])

def save_index(idx):
    save_json(INDEX_FILE, idx)

def week_items(idx, week_iso: str):
    return [x for x in idx if x.get("week_start") == week_iso]

def metric_items(items, metric: str):
    return [x for x in items if x.get("metric") == metric]

def file_exists(filename: str) -> bool:
    return (UPLOADS_DIR / filename).exists()

# =========================================================
# COMPLETENESS CHECKS
# =========================================================
def run_checks(block: dict, display_people: list, active_set: set, baselines: dict, week_iso: str):
    """
    Returns:
      status: "green" | "amber" | "red"
      issues: list[str]
    """
    issues_red = []
    issues_amber = []

    # Check: TL actions present
    for p in display_people:
        act = (block.get("actions", {}).get(p, "") or "").strip()
        if not act:
            issues_red.append(f"TL action missing for **{p}**.")

    deviations = block.get("deviations", {})
    metrics = block.get("metrics", {})

    # Check: baselines + daily entries
    for p in display_people:
        p_dev = deviations.get(p, {d: "Normal" for d in DAYS})

        for m in METRICS:
            md = metrics.get(m, {}).get(p, {"Baseline": "", **{d: "" for d in DAYS}})

            # baseline used: weekly override OR default baselines tab
            weekly_base = str(md.get("Baseline", "")).strip()
            default_base = str(baselines.get(p, {}).get(m, "")).strip()
            used_base = weekly_base if weekly_base != "" else default_base

            if used_base == "":
                issues_red.append(f"Baseline missing for **{p}** — **{m}**.")
            else:
                if to_float(used_base) is None:
                    issues_amber.append(f"Baseline is not a number for **{p}** — **{m}** (value: `{used_base}`).")

            # daily values required unless EXEMPT day
            for d in DAYS:
                dev = p_dev.get(d, "Normal")
                v = str(md.get(d, "")).strip()

                if dev in EXEMPT_DEVIATIONS:
                    # Optional flag: value entered on exempt day (not necessarily wrong)
                    if v != "":
                        issues_amber.append(f"Value entered on EXEMPT day for **{p}** — **{m}** — {d} ({dev}).")
                    continue

                # Non-exempt days must have a value
                if v == "":
                    issues_red.append(f"Missing value for **{p}** — **{m}** — {d} (deviation: {dev}).")

    # Check: uploads present for the week (helpful but not required)
    idx = load_index()
    week_uploads = week_items(idx, week_iso)
    for m in METRICS:
        cnt = sum(1 for x in metric_items(week_uploads, m) if x.get("filename") and file_exists(x["filename"]))
        if cnt == 0:
            issues_amber.append(f"No screenshot uploaded for **{m}** this week (optional, but recommended).")

    if issues_red:
        return "red", issues_red + issues_amber
    if issues_amber:
        return "amber", issues_amber
    return "green", []

# =========================================================
# UI
# =========================================================
st.title(f"📊 Team Performance Tracker — {TEAM}")

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

tab_uploads, tab_baselines, tab_deviations, tab_report, tab_history = st.tabs(
    ["✅ Uploads", "🎯 Baselines", "🗓️ Deviations", "📝 Weekly report", "🗂️ History"]
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
                (UPLOADS_DIR / safe_name).write_bytes(f.getbuffer())
                idx.append({"week_start": w_iso, "metric": metric, "filename": safe_name})
                saved += 1
            save_index(idx)
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
                p = UPLOADS_DIR / it["filename"]
                if p.exists():
                    st.image(str(p), width=900)
            st.divider()
    st.divider()
    st.subheader("Extract from screenshot (beta)")
    st.caption("Strict mode: if it can’t read confidently, it skips it (no guessing).")

    people_state = load_people_state()
    known_people = people_state["active"] + people_state["archived"]

    # Use the most recent upload for this week + currently selected metric
    idx = load_index()
    candidates = [x for x in idx if x.get("week_start") == w_iso and x.get("metric") == metric]
    candidates = list(reversed(candidates))

    if not candidates:
        st.info("Upload a screenshot for the selected metric first.")
    else:
        latest = candidates[0]
        img_path = UPLOADS_DIR / latest["filename"]

        if img_path.exists():
            st.image(str(img_path), width=900)

            if st.button("Attempt extract (beta)", use_container_width=True, key=f"extract_{TEAM}_{w_iso}_{metric}"):
                extracted, dbg = extract_stacked_chart(img_path.read_bytes(), known_people)
                if "error" in dbg:
                    st.error(dbg["error"])
                else:
                    st.write("Detected dates:", dbg["dates_detected"])
                    st.write("Detected legend names:", dbg["legend_people_detected"])
                    st.json(extracted)
        else:
            st.warning("Latest upload file not found on disk.")

# =========================================================
# TAB: BASELINES (people management minimised)
# =========================================================
with tab_baselines:
    st.subheader("Baselines (Active people only)")

    people_state = load_people_state()
    active = people_state["active"]
    archived = people_state["archived"]

    baselines = load_json(BASELINES_FILE, {})

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
            new_b = load_json(BASELINES_FILE, {})
            for _, r in edited.iterrows():
                person = str(r["Person"]).strip()
                if not person:
                    continue
                new_b[person] = {
                    "Calls": str(r.get("Calls", "")).strip(),
                    "EA Calls": str(r.get("EA Calls", "")).strip(),
                    "Things Done": str(r.get("Things Done", "")).strip(),
                }
            save_json(BASELINES_FILE, new_b)
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

        st.markdown("### Add person")
        new_name = st.text_input("Name", key=f"add_person_{TEAM}").strip()
        if st.button("Add to Active", use_container_width=True, key=f"add_person_btn_{TEAM}"):
            if not new_name:
                st.error("Enter a name first.")
            else:
                if new_name in archived:
                    archived.remove(new_name)
                if new_name not in active:
                    active.append(new_name)
                save_people_state({"active": active, "archived": archived})
                st.success(f"Added {new_name}.")
                st.rerun()

        st.divider()

        st.markdown("### Archive (removes from current views, keeps history)")
        to_archive = st.multiselect("Select active people", options=active, key=f"archive_select_{TEAM}")
        if st.button("Archive selected", use_container_width=True, key=f"archive_btn_{TEAM}"):
            if not to_archive:
                st.error("Select at least one person to archive.")
            else:
                for p in to_archive:
                    if p in active:
                        active.remove(p)
                    if p not in archived:
                        archived.append(p)
                save_people_state({"active": active, "archived": archived})
                st.success("Archived.")
                st.rerun()

        st.divider()

        st.markdown("### Restore")
        to_restore = st.multiselect("Select archived people", options=archived, key=f"restore_select_{TEAM}")
        if st.button("Restore selected", use_container_width=True, key=f"restore_btn_{TEAM}"):
            if not to_restore:
                st.error("Select at least one person to restore.")
            else:
                for p in to_restore:
                    if p in archived:
                        archived.remove(p)
                    if p not in active:
                        active.append(p)
                save_people_state({"active": active, "archived": archived})
                st.success("Restored.")
                st.rerun()

# =========================================================
# TAB: DEVIATIONS
# =========================================================
with tab_deviations:
    st.subheader("Deviations (sick / half day / reward time etc.)")

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
        st.success("Saved deviations.")

# =========================================================
# TAB: WEEKLY REPORT (checks + copy/paste preview)
# =========================================================
with tab_report:
    st.subheader("Weekly report (per person)")

    people_state = load_people_state()
    active = people_state["active"]
    active_set = set(active)

    default_week = monday_of(date.today())
    rep_week = monday_of(st.date_input("Report week starting (Monday)", value=default_week, key="rep_week"))
    rep_week_iso = iso(rep_week)

    show_archived = st.checkbox("Show archived people", value=False, key=f"rep_show_archived_{TEAM}_{rep_week_iso}")

    baselines = load_json(BASELINES_FILE, {})

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
                stored[person] = {
                    "Baseline": r.get("Baseline", ""),
                    **{d: r.get(d, "") for d in DAYS}
                }
            reports[rep_week_iso]["metrics"][metric] = stored
            save_reports(reports)
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
        st.success("Saved TL actions.")

    st.divider()

    # Refresh block after saves
    reports = load_reports()
    block = reports[rep_week_iso]

    report_txt = generate_report_txt(TEAM, rep_week, block, display_people)
    report_tsv = generate_report_tsv(block, display_people)

    # ✅ Checks
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

    # ✅ Copy/paste preview
    st.subheader("Report preview (copy/paste)")
    st.caption("Tip: click inside → Ctrl+A → Ctrl+C (Chromebook friendly).")
    st.text_area(
        "Weekly report text",
        value=report_txt,
        height=520,
        key=f"preview_{TEAM}_{rep_week_iso}_{'all' if show_archived else 'active'}"
    )

    st.divider()

    # Downloads (still useful for attaching / saving history)
    st.subheader("Downloads")
    st.download_button(
        "Download weekly report (TXT)",
        data=report_txt.encode("utf-8"),
        file_name=f"{TEAM.lower()}_weekly_report_{rep_week_iso}.txt",
        mime="text/plain",
        use_container_width=True,
        key=f"dl_txt_{TEAM}_{rep_week_iso}_{'all' if show_archived else 'active'}"
    )
    st.download_button(
        "Download weekly data (TSV)",
        data=report_tsv.encode("utf-8"),
        file_name=f"{TEAM.lower()}_weekly_data_{rep_week_iso}.tsv",
        mime="text/tab-separated-values",
        use_container_width=True,
        key=f"dl_tsv_{TEAM}_{rep_week_iso}_{'all' if show_archived else 'active'}"
    )

# =========================================================
# TAB: HISTORY (UPLOADS)
# =========================================================
with tab_history:
    st.subheader("Upload history")

    idx = load_index()
    if not idx:
        st.info("No uploads yet.")
        st.stop()

    weeks = sorted({x.get("week_start") for x in idx if x.get("week_start")}, reverse=True)
    selected_week_iso = st.selectbox("Select a week", weeks, key="history_week_select")
    selected_week_date = date.fromisoformat(selected_week_iso)

    st.caption(f"Selected week: **{week_label(selected_week_date)}**")

    items = week_items(idx, selected_week_iso)
    cols = st.columns(3)
    for i, m in enumerate(METRICS):
        c = sum(1 for x in metric_items(items, m) if x.get("filename") and file_exists(x["filename"]))
        cols[i].metric(m, f"{'✅' if c > 0 else '❌'}  {c} uploaded")

    st.divider()
    for m in METRICS:
        mi = metric_items(items, m)
        if not mi:
            continue
        st.markdown(f"### {m}")
        for it in reversed(mi):
            p = UPLOADS_DIR / it["filename"]
            if p.exists():
                st.image(str(p), width=900)
        st.divider()

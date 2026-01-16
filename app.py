import streamlit as st
from datetime import date, timedelta
from pathlib import Path
import json
import pandas as pd

# ----------------------------
# Auth
# ----------------------------
APP_PASSWORD = st.secrets["APP_PASSWORD"] if "APP_PASSWORD" in st.secrets else "password"

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 Login required")
    password = st.text_input("Enter password", type="password")

    if password == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.error("Incorrect password")

    return False

if not check_password():
    st.stop()

# ----------------------------
# Team selection (North / South)
# ----------------------------
def choose_team():
    if "team" not in st.session_state:
        st.session_state.team = None

    if st.session_state.team:
        return st.session_state.team

    st.title("Select team")
    team = st.radio("Where do you want to work?", ["North", "South"], horizontal=True)
    if st.button("Continue", use_container_width=True):
        st.session_state.team = team
        st.rerun()

    st.stop()

TEAM = choose_team()

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title=f"Team Performance Tracker — {TEAM}", layout="wide")

BASE_DATA_DIR = Path("data")
DATA_DIR = BASE_DATA_DIR / TEAM
UPLOADS_DIR = DATA_DIR / "uploads"
INDEX_FILE = DATA_DIR / "index.json"
PEOPLE_FILE = DATA_DIR / "people.json"
REPORTS_FILE = DATA_DIR / "reports.json"
BASELINES_FILE = DATA_DIR / "baselines.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["Calls", "EA Calls", "Things Done"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

DEVIATION_OPTIONS = ["Normal", "Half day", "Sick", "Annual leave", "Reward time", "Other"]

DEVIATION_MULTIPLIER = {
    "Normal": 1.0,
    "Half day": 0.5,
    "Sick": 0.0,
    "Annual leave": 0.0,
    "Reward time": 0.0,
    "Other": 1.0,  # we can later add custom % if needed
}

# ----------------------------
# Storage helpers
# ----------------------------
def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def load_index():
    return load_json(INDEX_FILE, [])

def save_index(items):
    save_json(INDEX_FILE, items)

def load_people():
    default_people = ["Rebecca", "Nicole", "Sonia"]
    return load_json(PEOPLE_FILE, default_people)

def save_people(people_list):
    save_json(PEOPLE_FILE, people_list)

def load_reports():
    return load_json(REPORTS_FILE, {})

def save_reports(reports_dict):
    save_json(REPORTS_FILE, reports_dict)

def load_baselines():
    # Structure: { "Person": { "Calls": 150, "EA Calls": 50, "Things Done": 20 }, ... }
    return load_json(BASELINES_FILE, {})

def save_baselines(b):
    save_json(BASELINES_FILE, b)

# ----------------------------
# Date helpers
# ----------------------------
def iso(d: date) -> str:
    return d.isoformat()

def monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())

def week_label(week_start: date) -> str:
    week_end = week_start + timedelta(days=6)
    return f"{week_start.strftime('%d %b %Y')} – {week_end.strftime('%d %b %Y')}"

def week_items(index, week_start_iso: str):
    return [x for x in index if x.get("week_start") == week_start_iso]

def metric_items(items, metric: str):
    return [x for x in items if x.get("metric") == metric]

def file_exists(filename: str) -> bool:
    return (UPLOADS_DIR / filename).exists()

# ----------------------------
# Report helpers
# ----------------------------
def ensure_week_report_structure(reports, week_iso, people):
    """
    Ensure the weekly report structure exists and
    ALWAYS syncs in any new people from the master people list.
    Never removes existing people from historical weeks.
    """
    if week_iso not in reports:
        reports[week_iso] = {
            "people": list(people),
            "metrics": {},
            "actions": {},
            "deviations": {}
        }
    else:
        # 🔑 SYNC STEP: add any new people to this existing week
        existing = reports[week_iso].get("people", [])
        for p in people:
            if p not in existing:
                existing.append(p)
        reports[week_iso]["people"] = existing

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

def df_from_saved_metric(metric_dict, people, default_baseline_value):
    rows = []
    for p in people:
        saved = metric_dict.get(p, {"Baseline": "", **{d: "" for d in DAYS}})
        baseline = saved.get("Baseline", "")
        # if user hasn't set a baseline in the week table, use baselines tab value
        if baseline == "" and default_baseline_value is not None:
            baseline = default_baseline_value
        row = {"Person": p, "Baseline": baseline}
        for d in DAYS:
            row[d] = saved.get(d, "")
        rows.append(row)
    return pd.DataFrame(rows)

def saved_metric_from_df(df):
    out = {}
    for _, r in df.iterrows():
        person = str(r["Person"]).strip()
        if not person:
            continue
        out[person] = {"Baseline": r.get("Baseline", "")}
        for d in DAYS:
            out[person][d] = r.get(d, "")
    return out

def deviations_df_from_saved(deviations_dict, people):
    rows = []
    for p in people:
        saved = deviations_dict.get(p, {d: "Normal" for d in DAYS})
        row = {"Person": p}
        for d in DAYS:
            row[d] = saved.get(d, "Normal")
        rows.append(row)
    return pd.DataFrame(rows)

def deviations_from_df(df):
    out = {}
    for _, r in df.iterrows():
        p = str(r["Person"]).strip()
        if not p:
            continue
        out[p] = {}
        for d in DAYS:
            v = r.get(d, "Normal")
            out[p][d] = v if v in DEVIATION_OPTIONS else "Normal"
    return out

def _to_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def generate_weekly_report_text(team: str, week_start: date, report_block: dict) -> str:
    week_iso = iso(week_start)
    label = week_label(week_start)
    people = report_block.get("people", [])
    metrics = report_block.get("metrics", {})
    actions = report_block.get("actions", {})
    deviations = report_block.get("deviations", {})

    lines = []
    lines.append(f"{team} — Weekly performance report — {label}")
    lines.append(f"Week start (ISO): {week_iso}")
    lines.append("")
    lines.append("This report is designed to stand alone (no spreadsheet context).")
    lines.append("Deviations reduce daily baseline fairly (e.g., sick day = baseline 0, half day = baseline 50%).")
    lines.append("TLs: include what you will do about performance gaps below.")
    lines.append("")
    lines.append("------------------------------------------------------------")
    lines.append("")

    for p in people:
        lines.append(f"{p}")
        lines.append("-" * len(p))

        p_dev = deviations.get(p, {d: "Normal" for d in DAYS})

        for m in METRICS:
            md = metrics.get(m, {}).get(p, {"Baseline": "", **{d: "" for d in DAYS}})
            base = _to_float(md.get("Baseline", ""))

            day_parts = []
            for d in DAYS:
                v_raw = md.get(d, "")
                v_num = _to_float(v_raw)
                dev = p_dev.get(d, "Normal")
                mult = DEVIATION_MULTIPLIER.get(dev, 1.0)

                # If no baseline set, just show the value.
                if base is None:
                    day_parts.append(f"{d} - {v_raw}")
                    continue

                adj = base * mult

                # If exempt day (adj baseline 0), show as EXEMPT
                if adj == 0:
                    day_parts.append(f"{d} - {v_raw}/EXEMPT ({dev})")
                else:
                    # show actual/adjusted baseline
                    day_parts.append(f"{d} - {v_raw}/{int(adj) if adj.is_integer() else adj:g} ({dev})")

            lines.append(f"{m}: " + " | ".join(day_parts))

        lines.append("")
        lines.append("TL action (what I will do about it):")
        act = (actions.get(p, "") or "").strip()
        lines.append(act if act else "[NOT FILLED IN]")
        lines.append("")
        lines.append("------------------------------------------------------------")
        lines.append("")

    return "\n".join(lines)

def generate_weekly_report_tsv(report_block: dict) -> str:
    people = report_block.get("people", [])
    metrics = report_block.get("metrics", {})
    rows = []
    for p in people:
        for m in METRICS:
            md = metrics.get(m, {}).get(p, {"Baseline": "", **{d: "" for d in DAYS}})
            row = [p, m, str(md.get("Baseline", ""))] + [str(md.get(d, "")) for d in DAYS]
            rows.append("\t".join(row))
    return "\n".join(rows)

# ----------------------------
# UI
# ----------------------------
st.title(f"📊 Team Performance Tracker — {TEAM}")

with st.sidebar:
    st.subheader("Team")
    st.write(f"**{TEAM}**")
    if st.button("Switch team"):
        st.session_state.team = None
        st.rerun()

tab_upload, tab_baselines, tab_deviations, tab_report, tab_history = st.tabs(
    ["✅ This week", "🎯 Baselines", "🗓️ Deviations", "📝 Weekly report", "🗂️ History"]
)

# ----------------------------
# Tab: This week (uploads)
# ----------------------------
with tab_upload:
    default_week = monday_of(date.today())
    week_start = st.date_input("Week starting (Monday)", value=default_week, key="upload_week")
    week_start = monday_of(week_start)
    st.caption(f"Selected week: **{week_label(week_start)}**")

    index = load_index()
    this_week_iso = iso(week_start)
    this_week = week_items(index, this_week_iso)

    st.divider()
    st.subheader("Status for this week")

    cols = st.columns(3)
    for i, metric in enumerate(METRICS):
        items = metric_items(this_week, metric)
        count = sum(1 for x in items if file_exists(x["filename"]))
        status = "✅" if count > 0 else "❌"
        cols[i].metric(label=metric, value=f"{status}  {count} uploaded")

    st.divider()
    st.subheader("Upload screenshots for this week")

    metric = st.selectbox("Metric type", METRICS, key="metric_select_upload")
    uploaded_files = st.file_uploader(
        "Upload chart screenshots (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="uploader"
    )

    if st.button("Save uploads", use_container_width=True):
        if not uploaded_files:
            st.error("Upload at least one screenshot.")
        else:
            index = load_index()
            saved = 0
            for f in uploaded_files:
                safe_name = f"{this_week_iso}_{metric}_{f.name}".replace(" ", "_")
                (UPLOADS_DIR / safe_name).write_bytes(f.getbuffer())
                index.append({"week_start": this_week_iso, "metric": metric, "filename": safe_name})
                saved += 1
            save_index(index)
            st.success(f"Saved {saved} file(s) for {metric}.")
            st.rerun()

    st.divider()
    st.subheader("This week’s uploads")

    index = load_index()
    this_week = week_items(index, this_week_iso)
    if not this_week:
        st.info("No uploads saved for this week yet.")
    else:
        for metric in METRICS:
            items = metric_items(this_week, metric)
            if not items:
                continue
            st.markdown(f"### {metric}")
            for item in reversed(items):
                img_path = UPLOADS_DIR / item["filename"]
                if img_path.exists():
                    st.image(str(img_path), width=900)
            st.divider()

# ----------------------------
# Tab: Baselines (per person)
# ----------------------------
with tab_baselines:
    st.subheader("Baselines (per person)")
    st.write("These are the default baselines used for reports. They are team-specific (North/South).")

    with st.expander("People list"):
        people = load_people()
        people_text = st.text_area("One person per line", value="\n".join(people), height=150, key="people_editor_baselines")
        if st.button("Save people list", key="save_people_baselines"):
            cleaned = [x.strip() for x in people_text.splitlines() if x.strip()]
            if not cleaned:
                st.error("People list cannot be empty.")
            else:
                save_people(cleaned)
                st.success("Saved people list.")
                st.rerun()

    people = load_people()
    baselines = load_baselines()

    # Build dataframe
    rows = []
    for p in people:
        rows.append({
            "Person": p,
            "Calls": baselines.get(p, {}).get("Calls", ""),
            "EA Calls": baselines.get(p, {}).get("EA Calls", ""),
            "Things Done": baselines.get(p, {}).get("Things Done", ""),
        })

    df = pd.DataFrame(rows)

    edited = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Person": st.column_config.TextColumn(disabled=True),
        },
        key="baseline_table"
    )

    if st.button("Save baselines", use_container_width=True):
        new_b = {}
        for _, r in edited.iterrows():
            p = str(r["Person"]).strip()
            if not p:
                continue
            new_b[p] = {
                "Calls": str(r.get("Calls", "")).strip(),
                "EA Calls": str(r.get("EA Calls", "")).strip(),
                "Things Done": str(r.get("Things Done", "")).strip(),
            }
        save_baselines(new_b)
        st.success("Saved baselines.")

# ----------------------------
# Tab: Deviations (per week, per person, per day)
# ----------------------------
with tab_deviations:
    st.subheader("Deviations (sick day / half day / reward time etc.)")
    st.write("These adjust the *daily baseline* for fairness. They do not change the actual numbers you enter.")

    default_week = monday_of(date.today())
    dev_week = st.date_input("Week starting (Monday)", value=default_week, key="dev_week")
    dev_week = monday_of(dev_week)
    dev_week_iso = iso(dev_week)
    st.caption(f"Selected week: **{week_label(dev_week)}**")

    people = load_people()
    reports = load_reports()
    ensure_week_report_structure(reports, dev_week_iso, people)
    save_reports(reports)

    block = load_reports()[dev_week_iso]
    dev_df = deviations_df_from_saved(block.get("deviations", {}), block["people"])

    edited_dev = st.data_editor(
        dev_df,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Person": st.column_config.TextColumn(disabled=True),
            "Mon": st.column_config.SelectboxColumn(options=DEVIATION_OPTIONS),
            "Tue": st.column_config.SelectboxColumn(options=DEVIATION_OPTIONS),
            "Wed": st.column_config.SelectboxColumn(options=DEVIATION_OPTIONS),
            "Thu": st.column_config.SelectboxColumn(options=DEVIATION_OPTIONS),
            "Fri": st.column_config.SelectboxColumn(options=DEVIATION_OPTIONS),
        },
        key=f"dev_table_{dev_week_iso}"
    )

    if st.button("Save deviations", use_container_width=True):
        reports = load_reports()
        ensure_week_report_structure(reports, dev_week_iso, load_people())
        reports[dev_week_iso]["deviations"] = deviations_from_df(edited_dev)
        save_reports(reports)
        st.success("Saved deviations.")

# ----------------------------
# Tab: Weekly report (per person)
# ----------------------------
with tab_report:
    st.subheader("Weekly report builder (per person)")

    default_week = monday_of(date.today())
    report_week = st.date_input("Report week starting (Monday)", value=default_week, key="report_week")
    report_week = monday_of(report_week)
    report_week_iso = iso(report_week)
    st.caption(f"Selected week: **{week_label(report_week)}**")

    people = load_people()
    baselines = load_baselines()

    reports = load_reports()
    ensure_week_report_structure(reports, report_week_iso, people)
    save_reports(reports)

    st.divider()
    st.write("Enter daily actuals. Baseline will auto-fill from the Baselines tab (you can override per week if needed).")

    for metric in METRICS:
        st.markdown(f"### {metric}")
        reports = load_reports()
        block = reports[report_week_iso]

        # Default baseline value per person for this metric
        metric_dict = block["metrics"].get(metric, {})
        rows = []
        for p in block["people"]:
            saved = metric_dict.get(p, {"Baseline": "", **{d: "" for d in DAYS}})
            default_b = baselines.get(p, {}).get(metric, "")
            df_row = {"Person": p}

            baseline = saved.get("Baseline", "")
            if baseline == "" and default_b != "":
                baseline = default_b

            df_row["Baseline"] = baseline
            for d in DAYS:
                df_row[d] = saved.get(d, "")
            rows.append(df_row)

        df = pd.DataFrame(rows)

        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={"Person": st.column_config.TextColumn(disabled=True)},
            key=f"editor_{metric}_{report_week_iso}"
        )

        if st.button(f"Save {metric}", key=f"save_{metric}_{report_week_iso}"):
            reports = load_reports()
            ensure_week_report_structure(reports, report_week_iso, load_people())
            reports[report_week_iso]["metrics"][metric] = saved_metric_from_df(edited)
            save_reports(reports)
            st.success(f"Saved {metric}.")
            st.rerun()

        st.divider()

    st.subheader("TL actions (required)")
    reports = load_reports()
    ensure_week_report_structure(reports, report_week_iso, load_people())
    block = reports[report_week_iso]

    for p in block["people"]:
        block["actions"][p] = st.text_area(
            f"{p} — TL action",
            value=block["actions"].get(p, ""),
            height=100,
            key=f"action_{p}_{report_week_iso}"
        )

    if st.button("Save TL actions"):
        reports = load_reports()
        ensure_week_report_structure(reports, report_week_iso, load_people())
        for p in reports[report_week_iso]["people"]:
            reports[report_week_iso]["actions"][p] = st.session_state.get(f"action_{p}_{report_week_iso}", "")
        save_reports(reports)
        st.success("Saved TL actions.")

    st.divider()

    reports = load_reports()
    block = reports[report_week_iso]

    report_txt = generate_weekly_report_text(TEAM, report_week, block)
    report_tsv = generate_weekly_report_tsv(block)

    st.subheader("Downloads")
    st.download_button(
        "Download weekly report (TXT)",
        data=report_txt.encode("utf-8"),
        file_name=f"{TEAM.lower()}_weekly_report_{report_week_iso}.txt",
        mime="text/plain",
        use_container_width=True
    )
    st.download_button(
        "Download weekly data (TSV)",
        data=report_tsv.encode("utf-8"),
        file_name=f"{TEAM.lower()}_weekly_data_{report_week_iso}.tsv",
        mime="text/tab-separated-values",
        use_container_width=True
    )

# ----------------------------
# Tab: History (uploads)
# ----------------------------
with tab_history:
    st.subheader("Browse previous weeks (uploads)")

    index = load_index()
    if not index:
        st.info("No upload history yet.")
    else:
        weeks = sorted({x["week_start"] for x in index if "week_start" in x}, reverse=True)
        selected_week_iso = st.selectbox("Select a week", weeks, key="history_week")
        selected_week_date = date.fromisoformat(selected_week_iso)
        st.caption(f"Selected week: **{week_label(selected_week_date)}**")

        items = week_items(index, selected_week_iso)

        cols = st.columns(3)
        for i, metric in enumerate(METRICS):
            mitems = metric_items(items, metric)
            count = sum(1 for x in mitems if file_exists(x["filename"]))
            status = "✅" if count > 0 else "❌"
            cols[i].metric(label=metric, value=f"{status}  {count} uploaded")

        st.divider()
        for metric in METRICS:
            mitems = metric_items(items, metric)
            if not mitems:
                continue
            st.markdown(f"### {metric}")
            for item in reversed(mitems):
                img_path = UPLOADS_DIR / item["filename"]
                if img_path.exists():
                    st.image(str(img_path), width=900)
            st.divider()

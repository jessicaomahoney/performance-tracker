import streamlit as st
from datetime import date, timedelta
from pathlib import Path
import json
import pandas as pd

# ============================
# Auth
# ============================
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

# ============================
# Team selection
# ============================
def choose_team():
    if "team" not in st.session_state:
        st.session_state.team = None

    if st.session_state.team:
        return st.session_state.team

    st.title("Select team")
    team = st.radio("Choose team", ["North", "South"], horizontal=True)
    if st.button("Continue", use_container_width=True):
        st.session_state.team = team
        st.rerun()

    st.stop()

TEAM = choose_team()

# ============================
# App config
# ============================
st.set_page_config(page_title=f"Team Performance Tracker — {TEAM}", layout="wide")

BASE = Path("data") / TEAM
UPLOADS = BASE / "uploads"
PEOPLE_FILE = BASE / "people.json"
BASELINES_FILE = BASE / "baselines.json"
REPORTS_FILE = BASE / "reports.json"

BASE.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)

METRICS = ["Calls", "EA Calls", "Things Done"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

DEVIATION_OPTIONS = ["Normal", "Half day", "Sick", "Annual leave", "Reward time"]
DEVIATION_MULT = {
    "Normal": 1,
    "Half day": 0.5,
    "Sick": 0,
    "Annual leave": 0,
    "Reward time": 0,
}

# ============================
# Storage helpers
# ============================
def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))

# ============================
# People (active / archived)
# ============================
def load_people():
    raw = load_json(PEOPLE_FILE, {"active": [], "archived": []})
    if isinstance(raw, list):  # migration
        raw = {"active": raw, "archived": []}
        save_json(PEOPLE_FILE, raw)
    raw.setdefault("active", [])
    raw.setdefault("archived", [])
    return raw

def save_people(p):
    save_json(PEOPLE_FILE, p)

# ============================
# Date helpers
# ============================
def monday(d):
    return d - timedelta(days=d.weekday())

def iso(d):
    return d.isoformat()

# ============================
# Reports helpers
# ============================
def ensure_week(reports, week_iso, active_people):
    if week_iso not in reports:
        reports[week_iso] = {
            "people": list(active_people),
            "metrics": {},
            "actions": {},
            "deviations": {}
        }
    else:
        for p in active_people:
            if p not in reports[week_iso]["people"]:
                reports[week_iso]["people"].append(p)

    for m in METRICS:
        reports[week_iso]["metrics"].setdefault(m, {})
        for p in reports[week_iso]["people"]:
            reports[week_iso]["metrics"][m].setdefault(p, {"Baseline": "", **{d: "" for d in DAYS}})
            reports[week_iso]["actions"].setdefault(p, "")
            reports[week_iso]["deviations"].setdefault(p, {d: "Normal" for d in DAYS})

# ============================
# UI
# ============================
st.title(f"📊 Team Performance Tracker — {TEAM}")

with st.sidebar:
    if st.button("Switch team"):
        st.session_state.team = None
        st.rerun()

tab_baselines, tab_deviations, tab_report = st.tabs(
    ["🎯 Baselines", "🗓️ Deviations", "📝 Weekly report"]
)

# ============================
# Baselines tab
# ============================
with tab_baselines:
    st.subheader("Baselines (active people)")

    people = load_people()
    active = people["active"]
    archived = people["archived"]
    baselines = load_json(BASELINES_FILE, {})

    if active:
        df = pd.DataFrame([
            {
                "Person": p,
                "Calls": baselines.get(p, {}).get("Calls", ""),
                "EA Calls": baselines.get(p, {}).get("EA Calls", ""),
                "Things Done": baselines.get(p, {}).get("Things Done", ""),
            }
            for p in active
        ])

        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={"Person": st.column_config.TextColumn(disabled=True)},
             key=f"baselines_editor_{TEAM}"
        )
        if st.button("Save baselines", key=f"save_baselines_{TEAM}"):
            for _, r in edited.iterrows():
                baselines[r["Person"]] = {
                    "Calls": r["Calls"],
                    "EA Calls": r["EA Calls"],
                    "Things Done": r["Things Done"],
                }
            save_json(BASELINES_FILE, baselines)
            st.success("Saved baselines.")
    else:
        st.info("No active people yet.")

    with st.expander("People management"):
        new = st.text_input("Add person")
        if st.button("Add"):
            if new and new not in active:
                active.append(new)
                active.sort()
                save_people({"active": active, "archived": archived})
                st.rerun()

        to_archive = st.multiselect("Archive", active)
        if st.button("Archive selected"):
            for p in to_archive:
                active.remove(p)
                archived.append(p)
            save_people({"active": active, "archived": archived})
            st.rerun()

        to_restore = st.multiselect("Restore archived", archived)
        if st.button("Restore selected"):
            for p in to_restore:
                archived.remove(p)
                active.append(p)
            active.sort()
            save_people({"active": active, "archived": archived})
            st.rerun()

# ============================
# Deviations tab
# ============================
with tab_deviations:
    st.subheader("Deviations")

    week = monday(st.date_input("Week starting", value=monday(date.today())))
    week_iso = iso(week)

    people = load_people()
    active = people["active"]

    reports = load_json(REPORTS_FILE, {})
    ensure_week(reports, week_iso, active)
    save_json(REPORTS_FILE, reports)

    block = reports[week_iso]
    df = pd.DataFrame([
        {"Person": p, **block["deviations"].get(p, {})}
        for p in block["people"]
        if p in active
    ])

    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Person": st.column_config.TextColumn(disabled=True),
            **{d: st.column_config.SelectboxColumn(options=DEVIATION_OPTIONS) for d in DAYS}
        },
        key=f"deviations_editor_{TEAM}_{week_iso}"
    )

 if st.button("Save deviations", key=f"save_dev_{TEAM}_{week_iso}"):
        for _, r in edited.iterrows():
            block["deviations"][r["Person"]] = {d: r[d] for d in DAYS}
        save_json(REPORTS_FILE, reports)
        st.success("Saved deviations.")

# ============================
# Weekly report tab
# ============================
with tab_report:
    st.subheader("Weekly report")

    week = monday(st.date_input("Report week", value=monday(date.today())))
    week_iso = iso(week)

    people = load_people()
    active = people["active"]
    baselines = load_json(BASELINES_FILE, {})
    reports = load_json(REPORTS_FILE, {})
    ensure_week(reports, week_iso, active)
    save_json(REPORTS_FILE, reports)

    block = reports[week_iso]

    for metric in METRICS:
        st.markdown(f"### {metric}")
        df = pd.DataFrame([
            {
                "Person": p,
                "Baseline": block["metrics"][metric][p]["Baseline"] or baselines.get(p, {}).get(metric, ""),
                **{d: block["metrics"][metric][p][d] for d in DAYS}
            }
            for p in block["people"]
            if p in active
        ])

        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={"Person": st.column_config.TextColumn(disabled=True)}
            key=f"report_editor_{TEAM}_{week_iso}_{metric}"
        )

        if st.button(f"Save {metric}"key=f"save_{TEAM}_{week_iso}_{metric}")):
            for _, r in edited.iterrows():
                block["metrics"][metric][r["Person"]] = {
                    "Baseline": r["Baseline"],
                    **{d: r[d] for d in DAYS}
                }
            save_json(REPORTS_FILE, reports)
            st.success(f"Saved {metric}.")

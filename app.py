import streamlit as st
from datetime import date, timedelta
from pathlib import Path
import json

# ----------------------------
# Auth
# ----------------------------
# Recommended: set APP_PASSWORD in Streamlit Secrets:
# APP_PASSWORD = st.secrets["APP_PASSWORD"]
# If you haven't set secrets yet, you can temporarily use:
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
# App config
# ----------------------------
st.set_page_config(page_title="Team Performance Tracker", layout="wide")

DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
INDEX_FILE = DATA_DIR / "index.json"

DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["Calls", "EA Calls", "Things Done"]

def load_index():
    if not INDEX_FILE.exists():
        return []
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))

def save_index(items):
    INDEX_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")

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
# UI
# ----------------------------
st.title("📊 Team Performance Tracker")

tab_upload, tab_history = st.tabs(["✅ This week", "🗂️ History"])

with tab_upload:
    # Week selector defaults to current Monday
    default_week = monday_of(date.today())
    week_start = st.date_input("Week starting (Monday)", value=default_week)
    week_start = monday_of(week_start)  # force to Monday
    st.caption(f"Selected week: **{week_label(week_start)}**")

    index = load_index()
    this_week_iso = iso(week_start)
    this_week = week_items(index, this_week_iso)

    st.divider()

    # Status panel
    st.subheader("Status for this week")
    cols = st.columns(3)
    for i, metric in enumerate(METRICS):
        items = metric_items(this_week, metric)
        count = sum(1 for x in items if file_exists(x["filename"]))
        status = "✅" if count > 0 else "❌"
        cols[i].metric(label=metric, value=f"{status}  {count} uploaded")

    st.divider()

    # Upload section
    st.subheader("Upload screenshots for this week")
    metric = st.selectbox("Metric type", METRICS, key="metric_select")

    uploaded_files = st.file_uploader(
        "Upload chart screenshots (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        save_clicked = st.button("Save uploads", use_container_width=True)

    if save_clicked:
        if not uploaded_files:
            st.error("Upload at least one screenshot.")
        else:
            index = load_index()  # reload in case
            saved = 0

            for f in uploaded_files:
                safe_name = f"{this_week_iso}_{metric}_{f.name}".replace(" ", "_")
                out_path = UPLOADS_DIR / safe_name
                out_path.write_bytes(f.getbuffer())
                index.append({
                    "week_start": this_week_iso,
                    "metric": metric,
                    "filename": safe_name
                })
                saved += 1

            save_index(index)
            st.success(f"Saved {saved} file(s) for {metric}.")
            st.rerun()

    st.divider()

    # This week's uploads gallery
    st.subheader("This week’s uploads")

    index = load_index()
    this_week = week_items(index, this_week_iso)

    if not this_week:
        st.info("No uploads saved for this week yet.")
    else:
        # group by metric
        for metric in METRICS:
            items = metric_items(this_week, metric)
            if not items:
                continue

            st.markdown(f"### {metric}")
            # show newest first
            items = list(reversed(items))

            for item in items:
                img_path = UPLOADS_DIR / item["filename"]
                if img_path.exists():
                    st.image(str(img_path), width=900)
                else:
                    st.warning(f"Missing file: {item['filename']}")
            st.divider()

with tab_history:
    st.subheader("Browse previous weeks")

    index = load_index()
    if not index:
        st.info("No history yet.")
    else:
        # Build unique weeks list (sorted newest first)
        weeks = sorted({x["week_start"] for x in index if "week_start" in x}, reverse=True)

        selected_week_iso = st.selectbox("Select a week", weeks)
        selected_week_date = date.fromisoformat(selected_week_iso)

        st.caption(f"Selected week: **{week_label(selected_week_date)}**")

        items = week_items(index, selected_week_iso)

        # status row
        cols = st.columns(3)
        for i, metric in enumerate(METRICS):
            mitems = metric_items(items, metric)
            count = sum(1 for x in mitems if file_exists(x["filename"]))
            status = "✅" if count > 0 else "❌"
            cols[i].metric(label=metric, value=f"{status}  {count} uploaded")

        st.divider()

        # show images grouped
        for metric in METRICS:
            mitems = metric_items(items, metric)
            if not mitems:
                continue

            st.markdown(f"### {metric}")
            mitems = list(reversed(mitems))
            for item in mitems:
                img_path = UPLOADS_DIR / item["filename"]
                if img_path.exists():
                    st.image(str(img_path), width=900)
                else:
                    st.warning(f"Missing file: {item['filename']}")
            st.divider()

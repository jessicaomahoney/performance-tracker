import streamlit as st
from datetime import date
from pathlib import Path
import json

APP_PASSWORD = st.secrets["APP_PASSWORD"]

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

# Stop the app here unless the password is correct
if not check_password():
    st.stop()

st.set_page_config(page_title="Team Performance Tracker", layout="wide")

DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
INDEX_FILE = DATA_DIR / "index.json"

DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def load_index():
    if not INDEX_FILE.exists():
        return []
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))

def save_index(items):
    INDEX_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")

st.title("📊 Team Performance Tracker (Free MVP)")

st.write("Upload screenshots by week + metric. This version stores files and shows history.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    week_start = st.date_input("Week starting (Monday)", value=date.today())
with col2:
    metric = st.selectbox("Metric", ["Calls", "EA Calls", "Things Done"])

uploaded_files = st.file_uploader(
    "Upload chart screenshots (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if st.button("Save uploads"):
    if not uploaded_files:
        st.error("Upload at least one screenshot.")
    else:
        index = load_index()

        for f in uploaded_files:
            safe_name = f"{week_start.isoformat()}_{metric}_{f.name}".replace(" ", "_")
            out_path = UPLOADS_DIR / safe_name
            out_path.write_bytes(f.getbuffer())

            index.append({
                "week_start": week_start.isoformat(),
                "metric": metric,
                "filename": safe_name
            })

        save_index(index)
        st.success(f"Saved {len(uploaded_files)} file(s).")

st.divider()
st.subheader("📁 History")

index = load_index()
if not index:
    st.info("No uploads yet.")
else:
    for item in reversed(index[-50:]):
        st.write(f"**Week:** {item['week_start']} | **Metric:** {item['metric']}")
        img_path = UPLOADS_DIR / item["filename"]
        if img_path.exists():
            st.image(str(img_path), width=700)
        st.divider()

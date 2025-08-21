import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Fantasy Draft War Room", layout="wide")

st.title("🏈 Fantasy Draft War Room")

# ---------- Helpers ----------
EXPECTED_COLS = [
    "Player","Position","Team","ByeWeek","ECR","ADP","Tier",
    "SOS_Score","ProjectedPoints","IsDrafted"
]

ALIASES = {
    "ByeWeek": ["Bye", "Bye Week", "Bye_Week"],
    "ProjectedPoints": ["Proj", "ProjPoints", "Projected", "Projection", "Projected_Points"],
    "SOS_Score": ["SOS", "StrengthOfSchedule", "Strength_of_Schedule", "SOS (Stars)", "SOS Stars"]
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Map common aliases to expected names
    for target, candidates in ALIASES.items():
        if target not in df.columns:
            for c in candidates:
                if c in df.columns:
                    df[target] = df[c]
                    break

    # Add any missing expected columns with sensible defaults
    for c in EXPECTED_COLS:
        if c not in df.columns:
            if c in ["ECR","ADP","Tier","ProjectedPoints","ByeWeek"]:
                df[c] = pd.NA
            elif c == "IsDrafted":
                df[c] = "N"
            else:
                df[c] = ""

    # Types
    for c in ["ByeWeek","ECR","ADP","Tier","ProjectedPoints"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # SOS stars → numeric (1–5)
    if "SOS_Score" in df.columns:
        def sos_to_num(x):
            if pd.isna(x): return np.nan
            s = str(x)
            if "★" in s:
                return s.count("★")
            # handle "3 out of 5 stars" or plain "3"
            for ch in s:
                if ch.isdigit():
                    return int(ch)
            return np.nan
        df["SOS_Num"] = df["SOS_Score"].apply(sos_to_num)
    else:
        df["SOS_Num"] = np.nan

    # Normalize IsDrafted to Y/N text
    df["IsDrafted"] = df["IsDrafted"].astype(str).str.upper().map(lambda v: "Y" if v in ["Y","YES","TRUE","1"] else "N")

    # Basic derived metric: value (lower is better for ADP)
    df["DraftValue"] = df["ECR"] - df["ADP"]
    return df

# ---------- Data input ----------
st.sidebar.header("📥 Data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel with your player pool", type=["csv","xlsx"])

if uploaded is None:
    st.info("Upload your rankings/ADP sheet to get started (CSV or XLSX).")
    # Sample starter table (so you can see the UI without data)
    sample = pd.DataFrame({
        "Player": ["Justin Jefferson","Christian McCaffrey","Ja'Marr Chase","Tyreek Hill","Travis Kelce"],
        "Position": ["WR","RB","WR","WR","TE"],
        "Team": ["MIN","SF","CIN","MIA","KC"],
        "ByeWeek": [13,9,12,10,6],
        "ECR": [1,2,3,4,5],
        "ADP": [2.2,1.3,4.1,4.0,11.0],
        "Tier": [1,1,1,1,1],
        "SOS_Score": ["★★★","★★","★★★★","★★★","★★"],
        "ProjectedPoints": [285,300,270,268,240],
        "IsDrafted": ["N","N","N","N","N"]
    })
    df = normalize_columns(sample.copy())
else:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    df = normalize_columns(df)

# ---------- Filters ----------
st.sidebar.header("🔎 Filters")
pos = st.sidebar.multiselect("Position", sorted(df["Position"].dropna().unique().tolist()))
tier = st.sidebar.multiselect("Tier", sorted([int(t) for t in df["Tier"].dropna().unique().tolist()]))
show_only_available = st.sidebar.checkbox("Show only available (IsDrafted = N)", value=False)
search = st.sidebar.text_input("Search player")

df_view = df.copy()
if pos: df_view = df_view[df_view["Position"].isin(pos)]
if tier: df_view = df_view[df_view["Tier"].isin(tier)]
if show_only_available: df_view = df_view[df_view["IsDrafted"] == "N"]
if search:
    s = search.lower()
    df_view = df_view[df_view["Player"].str.lower().str.contains(s)]

# ---------- Styling helpers ----------
def value_color(val):
    if pd.isna(val): return ""
    # negative = value (good); positive = reach (bad)
    if val <= -2: color = "#1f8a70"  # strong value
    elif val < 0: color = "#7fc8a9"   # mild value
    elif val < 2: color = "#f2f2f2"   # neutral
    else: color = "#ffb3b3"           # reach
    return f"background-color: {color}"

def drafted_style(is_drafted):
    return "opacity: 0.45;" if is_drafted == "Y" else ""

# ---------- Display metrics ----------
left, mid, right = st.columns(3)
with left:
    st.metric("Players in view", len(df_view))
with mid:
    st.metric("Available in view", int((df_view["IsDrafted"]=="N").sum()))
with right:
    st.metric("Avg Draft Value (lower is better)", round(df_view["DraftValue"].dropna().mean(),2) if df_view["DraftValue"].notna().any() else "—")

# ---------- Editable table (mark drafted here) ----------
st.subheader("Draft Board")

# Create an editable copy with a checkbox-like column
edit_df = df_view.copy()
edit_df.insert(0, "Drafted?", edit_df["IsDrafted"].map(lambda v: True if v=="Y" else False))

# Choose columns to show/edit
show_cols = ["Drafted?","Player","Position","Team","ByeWeek","Tier","ECR","ADP","DraftValue","SOS_Num","ProjectedPoints"]

edited = st.data_editor(
    edit_df[show_cols],
    hide_index=True,
    use_container_width=True,
    column_config={
        "Drafted?": st.column_config.CheckboxColumn("Drafted?", help="Mark as drafted"),
        "DraftValue": st.column_config.NumberColumn(format="%.1f"),
        "SOS_Num": st.column_config.NumberColumn("SOS (1-5)")
    }
)

# Push edits back into df (so downloads reflect changes)
# Match rows by Player+Team+Position to be safe
key_cols = ["Player","Team","Position"]
merge_key = edited[["Player","Team","Position","Drafted?"]].copy()
merge_key["IsDrafted"] = merge_key["Drafted?"].map(lambda b: "Y" if b else "N")
merge_key = merge_key.drop(columns=["Drafted?"])

df = df.merge(merge_key, on=key_cols, how="left", suffixes=("","_upd"))
df["IsDrafted"] = np.where(df["IsDrafted_upd"].notna(), df["IsDrafted_upd"], df["IsDrafted"])
df = df.drop(columns=[c for c in df.columns if c.endswith("_upd")])

# ---------- Downloads ----------
st.subheader("⬇️ Download Updated Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="draft_board_updated.csv", mime="text/csv")

st.caption("Tip: Use filters to focus by position/tier or toggle 'Show only available'. Colors = value (green) vs reach (red). Drafted rows appear faded.")

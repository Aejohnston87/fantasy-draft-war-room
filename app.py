# app.py
import io
import re
import csv
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fantasy Draft War Room", layout="wide")
st.title("🏈 Fantasy Draft War Room")

# -----------------------------
# Helpers: CSV reading & normalization
# -----------------------------
@st.cache_data(show_spinner=False)
def read_any_csv(uploaded_file) -> pd.DataFrame:
    """
    Read 'messy' CSVs safely:
    - auto-detect delimiter (comma/semicolon/tab/pipe)
    - skip banner lines until a header row with 'player' appears
    - ignore bad lines
    """
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")

    # If user saved the web page instead of CSV
    if "<html" in text.lower():
        st.error("This file looks like a web page, not a CSV. Re-download using the **Download CSV** button on FantasyPros.")
        return pd.DataFrame()

    # Detect delimiter
    sample = "\n".join(text.splitlines()[:25])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
    except Exception:
        sep = ","  # fallback

    # Find first plausible header line (contains 'player')
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines[:10]):
        if "player" in line.lower():
            start = i
            break

    clean_text = "\n".join(lines[start:])
    return pd.read_csv(
        io.StringIO(clean_text),
        sep=sep,
        engine="python",
        on_bad_lines="skip"
    )

def first_col(df: pd.DataFrame, *candidates):
    """Find first existing column (case-insensitive) matching candidates."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for name in cols:
            if name == cand.lower():
                return cols[name]
    return None

def find_like(df: pd.DataFrame, *substrings):
    """Find first column whose name contains all substrings (case-insensitive)."""
    for c in df.columns:
        n = c.lower()
        if all(s.lower() in n for s in substrings):
            return c
    return None

def strip_digits(pos: str):
    """Turn WR12 -> WR, RB3 -> RB."""
    if pd.isna(pos):
        return pos
    return re.sub(r"\d+", "", str(pos)).strip().upper()

def extract_team_bye(col: pd.Series):
    """
    From strings like 'MIN (7)' or 'DAL (10)' extract team & bye.
    If not present, return (team, NaN).
    """
    teams, byes = [], []
    pattern = re.compile(r"\b([A-Z]{2,3})\b(?:\s*\((\d{1,2})\))?")
    for val in col.astype(str).fillna(""):
        m = pattern.search(val)
        if m:
            teams.append(m.group(1))
            byes.append(float(m.group(2)) if m.group(2) else np.nan)
        else:
            teams.append(np.nan)
            byes.append(np.nan)
    return pd.Series(teams), pd.Series(byes)

def stars_to_num(x):
    """Convert '★★★' or '3 out of 5' -> 3."""
    if pd.isna(x): return np.nan
    s = str(x)
    if "★" in s:
        return s.count("★")
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan

def normalize_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Convert various FantasyPros exports into a unified schema:
      Player, Position, Team, ByeWeek, ECR, ADP, Tier, SOS_Score, ProjectedPoints, IsDrafted
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=[
            "Player","Position","Team","ByeWeek","ECR","ADP","Tier",
            "SOS_Score","ProjectedPoints","IsDrafted","SOS_Num","DraftValue"
        ])

    df = df_in.copy()
    # Standardize column names (trim & collapse spaces)
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]

    # --- Player ---
    c_player = first_col(df, "Player", "PLAYER")
    if not c_player:
        # Sometimes FantasyPros has 'Player Name'
        c_player = find_like(df, "player")
    if c_player:
        df["Player"] = df[c_player].astype(str).str.strip()
    else:
        st.warning("Could not find a Player column. Showing raw columns.")
        df["Player"] = np.nan

    # --- Position ---
    c_pos = first_col(df, "Position", "POS")
    if not c_pos:
        c_pos = find_like(df, "pos")
    df["Position"] = df[c_pos].apply(strip_digits) if c_pos else np.nan

    # --- Team & ByeWeek ---
    # Try explicit columns first
    c_team = first_col(df, "Team", "NFL Team", "Tm")
    c_bye  = first_col(df, "ByeWeek", "Bye", "Bye Week", "BYE")
    # If not present, try a combined 'Team (Bye)' style field
    if not c_team or not c_bye:
        c_combined = find_like(df, "team", "bye")
        if c_combined:
            t, b = extract_team_bye(df[c_combined])
            df["Team"] = t
            df["ByeWeek"] = b
        else:
            if c_team: df["Team"] = df[c_team]
            if c_bye:  df["ByeWeek"] = pd.to_numeric(df[c_bye], errors="coerce")
    else:
        df["Team"] = df[c_team]
        df["ByeWeek"] = pd.to_numeric(df[c_bye], errors="coerce")

    # --- ECR / Rank ---
    c_ecr = first_col(df, "ECR", "ECR Rank", "RK", "Rank")
    df["ECR"] = pd.to_numeric(df[c_ecr], errors="coerce") if c_ecr else np.nan

    # --- ADP ---
    c_adp = first_col(df, "ADP", "AVG ADP", "Average Draft Position", "AVG")
    df["ADP"] = pd.to_numeric(df[c_adp], errors="coerce") if c_adp else np.nan

    # --- Tier ---
    c_tier = first_col(df, "Tier", "TIER")
    df["Tier"] = pd.to_numeric(df[c_tier], errors="coerce") if c_tier else np.nan

    # --- Projected Points ---
    c_proj = first_col(df, "ProjectedPoints", "Proj", "Projection", "Projected Points")
    df["ProjectedPoints"] = pd.to_numeric(df[c_proj], errors="coerce") if c_proj else np.nan

    # --- SOS (stars or numbers) ---
    c_sos = first_col(df, "SOS_Score", "SOS", "Strength of Schedule", "SOS Stars")
    if c_sos:
        df["SOS_Score"] = df[c_sos]
        df["SOS_Num"] = df["SOS_Score"].apply(stars_to_num)
    else:
        df["SOS_Score"] = np.nan
        df["SOS_Num"] = np.nan

    # Drafted flag default
    c_drafted = first_col(df, "IsDrafted")
    if c_drafted:
        df["IsDrafted"] = (
            df[c_drafted].astype(str).str.upper().map(lambda v: "Y" if v in ["Y","YES","TRUE","1"] else "N")
        )
    else:
        df["IsDrafted"] = "N"

    # Derived metric: value (negative = value)
    df["DraftValue"] = df["ECR"] - df["ADP"]

    # Final column order (plus keep originals for debugging)
    for col in ["Player","Position","Team","ByeWeek","ECR","ADP","Tier","SOS_Score","ProjectedPoints","IsDrafted","SOS_Num","DraftValue"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

# -----------------------------
# Sidebar: file input
# -----------------------------
st.sidebar.header("📥 Data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel with your player pool", type=["csv","xlsx"])

# Sample fallback so the UI isn't empty before upload
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

# Load data
if uploaded is None:
    st.info("Upload your FantasyPros CSV/XLSX to see your full board. Showing a small sample for now.")
    df = normalize_columns(sample)
else:
    if uploaded.name.lower().endswith(".csv"):
        df = read_any_csv(uploaded)
    else:
        uploaded.seek(0)
        df = pd.read_excel(uploaded, engine="openpyxl")
    df = normalize_columns(df)

if df.empty:
    st.stop()

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("🔎 Filters")
pos = st.sidebar.multiselect("Position", sorted([p for p in df["Position"].dropna().unique()]))
tiers_available = sorted([int(t) for t in df["Tier"].dropna().unique()])
tier = st.sidebar.multiselect("Tier", tiers_available)
only_available = st.sidebar.checkbox("Show only available (IsDrafted = N)", value=False)
search = st.sidebar.text_input("Search player")

view = df.copy()
if pos: view = view[view["Position"].isin(pos)]
if tier: view = view[view["Tier"].isin(tier)]
if only_available: view = view[view["IsDrafted"] == "N"]
if search:
    s = search.lower().strip()
    view = view[view["Player"].str.lower().str.contains(s)]

# -----------------------------
# Summary metrics
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Players in view", len(view))
with c2: st.metric("Available in view", int((view["IsDrafted"]=="N").sum()))
with c3:
    dv = view["DraftValue"].dropna()
    st.metric("Avg DraftValue (lower=better)", round(dv.mean(),2) if not dv.empty else "—")
with c4:
    st.metric("Avg SOS (1–5)", round(view["SOS_Num"].dropna().mean(),2) if view["SOS_Num"].notna().any() else "—")

# -----------------------------
# Draft board (editable)
# -----------------------------
st.subheader("Draft Board")

editable = view.copy()
editable.insert(0, "Drafted?", editable["IsDrafted"].map(lambda v: v == "Y"))

show_cols = ["Drafted?","Player","Position","Team","ByeWeek","Tier","ECR","ADP","DraftValue","SOS_Num","ProjectedPoints"]
show_cols = [c for c in show_cols if c in editable.columns]

edited = st.data_editor(
    editable[show_cols],
    hide_index=True,
    use_container_width=True,
    column_config={
        "Drafted?": st.column_config.CheckboxColumn("Drafted?"),
        "DraftValue": st.column_config.NumberColumn("DraftValue", help="ECR - ADP (negative = value)", format="%.1f"),
        "SOS_Num": st.column_config.NumberColumn("SOS (1–5)")
    }
)

# Push the Drafted? edits back into the full df (match on Player+Team+Position)
key_cols = ["Player","Team","Position"]
if all(k in edited.columns for k in key_cols):
    update = edited[key_cols + ["Drafted?"]].copy()
    update["IsDrafted"] = update["Drafted?"].map(lambda b: "Y" if b else "N")
    update = update.drop(columns=["Drafted?"])

    df = df.merge(update, on=key_cols, how="left", suffixes=("","_upd"))
    df["IsDrafted"] = np.where(df["IsDrafted_upd"].notna(), df["IsDrafted_upd"], df["IsDrafted"])
    df = df.drop(columns=[c for c in df.columns if c.endswith("_upd")])

# -----------------------------
# Download updated data
# -----------------------------
st.subheader("⬇️ Download Updated CSV")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="draft_board_updated.csv", mime="text/csv")

st.caption("Tip: Use filters (Position, Tier, Available) and search. Negative DraftValue = potential value pick. Checked rows are faded in your head; the CSV preserves your Drafted flags.")

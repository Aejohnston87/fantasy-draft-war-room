# app.py
import io, re, csv
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fantasy Draft War Room", layout="wide")
st.title("🏈 Fantasy Draft War Room")

# -----------------------------
# Robust readers
# -----------------------------
@st.cache_data(show_spinner=False)
def read_any_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    if "<html" in text.lower():
        st.error("That looks like a web page, not a CSV. Please click **Download CSV** on FantasyPros and upload that file.")
        return pd.DataFrame()

    # Detect delimiter
    sample = "\n".join(text.splitlines()[:40])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
    except Exception:
        sep = ","

    # Find first header row (contains 'player')
    lines = text.splitlines()
    header_idx = 0
    for i, line in enumerate(lines[:15]):
        if "player" in line.lower():
            header_idx = i
            break
    text = "\n".join(lines[header_idx:])
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="skip")

@st.cache_data(show_spinner=False)
def read_any_excel(uploaded_file) -> pd.DataFrame:
    """
    Read FantasyPros-style XLSX:
    - detect the first sheet
    - find header row by scanning for 'player'
    - re-read with header at that row
    """
    uploaded_file.seek(0)
    # Read all sheets minimally to pick one
    x = pd.ExcelFile(uploaded_file)
    sheet = x.sheet_names[0]

    # Read a small sample to find header row
    uploaded_file.seek(0)
    sample = pd.read_excel(uploaded_file, sheet_name=sheet, header=None, nrows=30, engine="openpyxl")
    header_row = 0
    for i in range(min(15, len(sample))):
        row_text = " ".join([str(v) for v in list(sample.iloc[i].values)])
        if "player" in row_text.lower():
            header_row = i
            break

    # Re-read with proper header
    uploaded_file.seek(0)
    df = pd.read_excel(uploaded_file, sheet_name=sheet, header=header_row, engine="openpyxl")
    # Drop fully empty columns/rows
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

# -----------------------------
# Normalization helpers
# -----------------------------
def stars_to_num(x):
    if pd.isna(x): return np.nan
    s = str(x)
    if "★" in s: return s.count("★")
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan

def strip_digits(pos):
    if pd.isna(pos): return pos
    return re.sub(r"\d+", "", str(pos)).strip().upper()

def extract_team_bye(series):
    """
    From 'MIN (7)' or 'Player Team (10)' patterns, extract team and bye.
    Works if the column contains only 'Team (Bye)' or 'Player Team (Bye)' strings.
    """
    teams, byes = [], []
    # Team in parentheses number
    pat = re.compile(r"\b([A-Z]{2,3})\b(?:.*?\((\d{1,2})\))?")
    for val in series.astype(str).fillna(""):
        m = pat.search(val)
        if m:
            teams.append(m.group(1))
            byes.append(float(m.group(2)) if m.group(2) else np.nan)
        else:
            teams.append(np.nan)
            byes.append(np.nan)
    return pd.Series(teams), pd.Series(byes)

def first_col(df, *names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower: return lower[n.lower()]
    return None

def find_contains(df, *parts):
    for c in df.columns:
        nc = c.lower()
        if all(p.lower() in nc for p in parts):
            return c
    return None

def normalize_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to:
    Player, Position, Team, ByeWeek, ECR, ADP, Tier, SOS_Score, ProjectedPoints, IsDrafted, SOS_Num, DraftValue
    Handles FantasyPros ADP & Rankings exports.
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=["Player","Position","Team","ByeWeek","ECR","ADP","Tier",
                                     "SOS_Score","ProjectedPoints","IsDrafted","SOS_Num","DraftValue"])
    df = df_in.copy()
    # Tidy headers
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]

    # ---- Player ----
    c_player = first_col(df, "Player", "PLAYER")
    if not c_player:
        # FantasyPros often uses 'Player Team (Bye)'
        c_player_combo = find_contains(df, "player")
        if c_player_combo:
            # Split off player name before team token if present
            # e.g., 'Christian McCaffrey SF (9)' -> 'Christian McCaffrey'
            df["Player"] = df[c_player_combo].astype(str).str.replace(r"\b[A-Z]{2,3}\b.*", "", regex=True).str.strip()
            # Also derive team/bye from same field
            team, bye = extract_team_bye(df[c_player_combo])
            df["Team"] = team
            df["ByeWeek"] = bye
        else:
            df["Player"] = np.nan
    else:
        df["Player"] = df[c_player].astype(str).str.strip()

    # ---- Position ----
    c_pos = first_col(df, "Position", "POS")
    if not c_pos: c_pos = find_contains(df, "pos")
    df["Position"] = df[c_pos].apply(strip_digits) if c_pos else np.nan

    # ---- Team & Bye (if not derived above) ----
    if "Team" not in df.columns or "ByeWeek" not in df.columns:
        c_team = first_col(df, "Team", "NFL Team", "Tm")
        c_bye  = first_col(df, "ByeWeek", "Bye", "Bye Week", "BYE")
        if c_team: df["Team"] = df["Team"] if "Team" in df.columns else df[c_team]
        if c_bye:  df["ByeWeek"] = pd.to_numeric(df[c_bye], errors="coerce")

    # ---- ECR / Rank ----
    # FantasyPros ADP file often has 'ECR' or 'RK'
    c_ecr = first_col(df, "ECR", "ECR Rank", "RK", "Rank")
    df["ECR"] = pd.to_numeric(df[c_ecr], errors="coerce") if c_ecr else np.nan

    # ---- ADP ----
    c_adp = first_col(df, "ADP", "AVG ADP", "Average Draft Position", "AVG")
    if not c_adp:
        # Some files have 'ADP AVG'
        c_adp = find_contains(df, "adp", "avg") or find_contains(df, "avg")
    df["ADP"] = pd.to_numeric(df[c_adp], errors="coerce") if c_adp else np.nan

    # ---- Tier ----
    c_tier = first_col(df, "Tier", "TIER")
    df["Tier"] = pd.to_numeric(df[c_tier], errors="coerce") if c_tier else np.nan

    # ---- Projected Points ----
    c_proj = first_col(df, "ProjectedPoints", "Proj", "Projection", "Projected Points", "FPTS", "PPR")
    df["ProjectedPoints"] = pd.to_numeric(df[c_proj], errors="coerce") if c_proj else np.nan

    # ---- SOS ----
    c_sos = first_col(df, "SOS_Score", "SOS", "Strength of Schedule", "SOS Stars")
    if c_sos:
        df["SOS_Score"] = df[c_sos]
        df["SOS_Num"] = df["SOS_Score"].apply(stars_to_num)
    else:
        df["SOS_Score"] = np.nan
        df["SOS_Num"] = np.nan

    # ---- Draft flag ----
    c_drafted = first_col(df, "IsDrafted")
    if c_drafted:
        df["IsDrafted"] = df[c_drafted].astype(str).str.upper().map(lambda v: "Y" if v in ["Y","YES","TRUE","1"] else "N")
    else:
        df["IsDrafted"] = "N"

    # Derived metric
    df["DraftValue"] = df["ECR"] - df["ADP"]
    # Ensure all expected columns exist
    for col in ["Player","Position","Team","ByeWeek","ECR","ADP","Tier","SOS_Score","ProjectedPoints","IsDrafted","SOS_Num","DraftValue"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

# -----------------------------
# Sidebar: upload
# -----------------------------
st.sidebar.header("📥 Data")
uploaded = st.sidebar.file_uploader("Upload FantasyPros CSV or XLSX", type=["csv","xlsx"])

# Small fallback so UI isn’t empty
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

# Load
if uploaded is None:
    st.info("Upload your FantasyPros file to see your full player pool. Showing a small sample until then.")
    df_raw = sample
else:
    if uploaded.name.lower().endswith(".csv"):
        df_raw = read_any_csv(uploaded)
    else:
        df_raw = read_any_excel(uploaded)

# Debug info
with st.expander("🔧 Debug: what did we read?", expanded=False):
    if uploaded is not None:
        st.write("**Detected columns:**", list(df_raw.columns))
        st.dataframe(df_raw.head(10), use_container_width=True)

df = normalize_columns(df_raw)
if df.empty or df["Player"].isna().all():
    st.error("Could not detect player rows. If this was an XLSX, open it and verify a sheet contains a table with a 'Player' column or a 'Player Team (Bye)' column. Then re-export and upload.")
    st.stop()

# -----------------------------
# Filters & metrics
# -----------------------------
st.sidebar.header("🔎 Filters")
pos = st.sidebar.multiselect("Position", sorted([p for p in df["Position"].dropna().unique()]))
tiers_avail = sorted([int(t) for t in df["Tier"].dropna().unique()]) if df["Tier"].notna().any() else []
tier = st.sidebar.multiselect("Tier", tiers_avail)
only_avail = st.sidebar.checkbox("Show only available (IsDrafted = N)", value=False)
search = st.sidebar.text_input("Search player")

view = df.copy()
if pos: view = view[view["Position"].isin(pos)]
if tier: view = view[view["Tier"].isin(tier)]
if only_avail: view = view[view["IsDrafted"]=="N"]
if search:
    s = search.lower().strip()
    view = view[view["Player"].str.lower().str.contains(s)]

c1,c2,c3,c4 = st.columns(4)
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
editable.insert(0, "Drafted?", editable["IsDrafted"].map(lambda v: v=="Y"))

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

# Push edits back (match on Player + Team + Position)
key_cols = ["Player","Team","Position"]
if all(k in edited.columns for k in key_cols):
    update = edited[key_cols + ["Drafted?"]].copy()
    update["IsDrafted"] = update["Drafted?"].map(lambda b: "Y" if b else "N")
    update = update.drop(columns=["Drafted?"])

    df = df.merge(update, on=key_cols, how="left", suffixes=("","_upd"))
    df["IsDrafted"] = np.where(df["IsDrafted_upd"].notna(), df["IsDrafted_upd"], df["IsDrafted"])
    df = df.drop(columns=[c for c in df.columns if c.endswith("_upd")])

# -----------------------------
# Download
# -----------------------------
st.subheader("⬇️ Download Updated CSV")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="draft_board_updated.csv", mime="text/csv")

# app.py
import io, re, csv
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fantasy Draft War Room", layout="wide")
st.title("🏈 Fantasy Draft War Room")

# =========================
# Robust readers (CSV/XLSX)
# =========================
@st.cache_data(show_spinner=False)
def read_any_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    if "<html" in text.lower():
        st.error("This looks like a web page, not a CSV. Please use FantasyPros' **Download CSV** and upload that file.")
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
    for i, line in enumerate(lines[:20]):
        if "player" in line.lower():
            header_idx = i
            break

    text = "\n".join(lines[header_idx:])
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="skip")

@st.cache_data(show_spinner=False)
def read_any_excel(uploaded_file) -> pd.DataFrame:
    """Handle FantasyPros Excel with banner rows by scanning for header row."""
    uploaded_file.seek(0)
    x = pd.ExcelFile(uploaded_file)
    sheet = x.sheet_names[0]

    uploaded_file.seek(0)
    sample = pd.read_excel(uploaded_file, sheet_name=sheet, header=None, nrows=30, engine="openpyxl")
    header_row = 0
    for i in range(min(20, len(sample))):
        row_text = " ".join([str(v) for v in list(sample.iloc[i].values)])
        if "player" in row_text.lower():
            header_row = i
            break

    uploaded_file.seek(0)
    df = pd.read_excel(uploaded_file, sheet_name=sheet, header=header_row, engine="openpyxl")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

def read_any(uploaded):
    if uploaded is None:
        return None
    if uploaded.name.lower().endswith(".csv"):
        return read_any_csv(uploaded)
    return read_any_excel(uploaded)

# =========================
# Normalization helpers
# =========================
def strip_digits(pos):
    if pd.isna(pos): return pos
    return re.sub(r"\d+", "", str(pos)).strip().upper()

def extract_team_bye(series):
    """Extract TEAM and BYE from strings like 'MIN (7)' or 'Christian McCaffrey SF (9)'."""
    teams, byes = [], []
    pat = re.compile(r"\b([A-Z]{2,3})\b(?:.*?\((\d{1,2})\))?")
    for val in series.astype(str).fillna(""):
        m = pat.search(val)
        if m:
            teams.append(m.group(1))
            byes.append(float(m.group(2)) if m.group(2) else np.nan)
        else:
            teams.append(np.nan); byes.append(np.nan)
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
    Standardize to: Player, Position, Team, ByeWeek, ECR, ADP, IsDrafted, DraftValue
    (We'll display Drafted? boolean derived from IsDrafted.)
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=["Player","Position","Team","ByeWeek","ECR","ADP","IsDrafted","DraftValue"])
    df = df_in.copy()
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]

    # Player
    c_player = first_col(df, "Player", "PLAYER")
    if not c_player:
        c_player_combo = find_contains(df, "player")
        if c_player_combo:
            df["Player"] = df[c_player_combo].astype(str).str.replace(r"\b[A-Z]{2,3}\b.*", "", regex=True).str.strip()
            team, bye = extract_team_bye(df[c_player_combo])
            df["Team"] = team; df["ByeWeek"] = bye
        else:
            df["Player"] = np.nan
    else:
        df["Player"] = df[c_player].astype(str).str.strip()

    # Position
    c_pos = first_col(df, "Position", "POS")
    if not c_pos: c_pos = find_contains(df, "pos")
    df["Position"] = df[c_pos].apply(strip_digits) if c_pos else np.nan

    # Team / Bye
    if "Team" not in df.columns or "ByeWeek" not in df.columns:
        c_team = first_col(df, "Team", "NFL Team", "Tm")
        c_bye  = first_col(df, "ByeWeek", "Bye", "Bye Week", "BYE")
        if c_team: df["Team"] = df[c_team]
        if c_bye:  df["ByeWeek"] = pd.to_numeric(df[c_bye], errors="coerce")

    # ECR / Rank
    c_ecr = first_col(df, "ECR", "ECR Rank", "RK", "Rank")
    df["ECR"] = pd.to_numeric(df[c_ecr], errors="coerce") if c_ecr else np.nan

    # ADP
    c_adp = first_col(df, "ADP", "AVG ADP", "Average Draft Position", "AVG")
    if not c_adp:
        c_adp = find_contains(df, "adp", "avg") or find_contains(df, "avg")
    df["ADP"] = pd.to_numeric(df[c_adp], errors="coerce") if c_adp else np.nan

    # IsDrafted flag
    c_drafted = first_col(df, "IsDrafted")
    if c_drafted:
        df["IsDrafted"] = (
            df[c_drafted].astype(str).str.upper().map(lambda v: "Y" if v in ["Y","YES","TRUE","1"] else "N")
        )
    else:
        df["IsDrafted"] = "N"

    # DraftValue
    df["DraftValue"] = df["ECR"] - df["ADP"]

    # Keep only the fields we want
    keep_cols = ["Player","Position","Team","ByeWeek","ECR","ADP","DraftValue","IsDrafted"]
    for col in keep_cols:
        if col not in df.columns: df[col] = np.nan
    df = df[keep_cols]

    # Clean types
    for c in ["ByeWeek","ECR","ADP","DraftValue"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# =========================
# Sidebar: TWO uploaders + merge
# =========================
st.sidebar.header("📥 Data")
ranks_file = st.sidebar.file_uploader("Upload Rankings (CSV/XLSX)", type=["csv","xlsx"], key="rankings")
adp_file   = st.sidebar.file_uploader("Upload ADP (CSV/XLSX)",      type=["csv","xlsx"], key="adp")

# Sample fallback so UI isn't empty
sample = pd.DataFrame({
    "Player": ["Justin Jefferson","Christian McCaffrey","Ja'Marr Chase","Tyreek Hill","Travis Kelce"],
    "Position": ["WR","RB","WR","WR","TE"],
    "Team": ["MIN","SF","CIN","MIA","KC"],
    "ByeWeek": [13,9,12,10,6],
    "ECR": [1,2,3,4,5],
    "ADP": [2.2,1.3,4.1,4.0,11.0],
    "DraftValue": [1-2.2, 2-1.3, 3-4.1, 4-4.0, 5-11.0],
    "IsDrafted": ["N","N","N","N","N"]
})

def safe_normalize(uploaded):
    raw = read_any(uploaded)
    return normalize_columns(raw) if raw is not None else None

df_ranks = safe_normalize(ranks_file)
df_adp   = safe_normalize(adp_file)

# Merge:
# - If both present: base=ranks, left-join ADP on Player+Position (fallback Player)
# - Prefer ADP from ADP file when present
if df_ranks is not None and not df_ranks.empty:
    df = df_ranks.copy()
    if df_adp is not None and not df_adp.empty:
        key_cols = [c for c in ["Player","Position"] if c in df.columns and c in df_adp.columns]
        if not key_cols: key_cols = ["Player"]
        cols_from_adp = [c for c in ["ADP","Team","ByeWeek","ECR"] if c in df_adp.columns]
        merged = df.merge(df_adp[key_cols + cols_from_adp], on=key_cols, how="left", suffixes=("","_adp"))
        if "ADP_adp" in merged.columns:
            merged["ADP"] = np.where(merged["ADP_adp"].notna(), merged["ADP_adp"], merged["ADP"])
        # Recompute DraftValue after pulling ADP
        merged["DraftValue"] = merged["ECR"] - merged["ADP"]
        # Drop merge suffix cols
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_adp")], errors="ignore")
        df = merged
elif df_adp is not None and not df_adp.empty:
    df = df_adp.copy()
else:
    st.info("Upload at least one file (Rankings and/or ADP). Showing a tiny sample until then.")
    df = sample.copy()

# Final safety: keep only requested columns (and drop empty cols)
keep_cols_final = ["Player","Position","Team","ByeWeek","ECR","ADP","DraftValue","IsDrafted"]
df = df[[c for c in keep_cols_final if c in df.columns]].dropna(how="all", axis=1)

if df.empty or df["Player"].isna().all():
    st.error("Could not detect player rows. Check your uploads have a 'Player' column (or a combined 'Player Team (Bye)' field).")
    st.stop()

# =========================
# Filters (reflect kept columns)
# =========================
st.sidebar.header("🔎 Filters")
pos_opts  = sorted([p for p in df["Position"].dropna().unique()]) if "Position" in df.columns else []
team_opts = sorted([t for t in df["Team"].dropna().unique()]) if "Team" in df.columns else []
bye_opts  = sorted([int(b) for b in df["ByeWeek"].dropna().unique()]) if "ByeWeek" in df.columns else []

pos_sel  = st.sidebar.multiselect("Position", pos_opts)
team_sel = st.sidebar.multiselect("Team", team_opts)
bye_sel  = st.sidebar.multiselect("ByeWeek", bye_opts)
only_avail = st.sidebar.checkbox("Only show available", value=True)
search = st.sidebar.text_input("Search player")

# Apply filters
view = df.copy()
if pos_sel:  view = view[view["Position"].isin(pos_sel)]
if team_sel: view = view[view["Team"].isin(team_sel)]
if bye_sel:  view = view[view["ByeWeek"].isin(bye_sel)]
if only_avail: view = view[view["IsDrafted"] != "Y"]
if search:
    s = search.lower().strip()
    view = view[view["Player"].str.lower().str.contains(s)]

# =========================
# Summary
# =========================
c1, c2, c3 = st.columns(3)
with c1: st.metric("Players in view", len(view))
with c2: st.metric("Available in view", int((view["IsDrafted"]!="Y").sum()))
with c3:
    dv = view["DraftValue"].dropna()
    st.metric("Avg DraftValue (lower=better)", round(dv.mean(),2) if not dv.empty else "—")

# =========================
# Draft board (editable)
# =========================
st.subheader("Draft Board")

editable = view.copy()
editable.insert(0, "Drafted?", editable["IsDrafted"].map(lambda v: v == "Y"))

show_cols = ["Drafted?","Player","Position","Team","ByeWeek","ECR","ADP","DraftValue"]
edited = st.data_editor(
    editable[show_cols],
    hide_index=True,
    use_container_width=True,
    column_config={
        "Drafted?": st.column_config.CheckboxColumn("Drafted?", help="Mark drafted to hide when 'Only show available' is on"),
        "ByeWeek": st.column_config.NumberColumn("ByeWeek", step=1),
        "ECR": st.column_config.NumberColumn("ECR", format="%.0f"),
        "ADP": st.column_config.NumberColumn("ADP", format="%.1f"),
        "DraftValue": st.column_config.NumberColumn("DraftValue", help="ECR - ADP (negative = value)", format="%.1f")
    }
)

# Push edits back into the full df (match on Player+Team+Position when possible)
key_cols = [c for c in ["Player","Team","Position"] if c in edited.columns and c in df.columns]
if not key_cols: key_cols = ["Player"]

update = edited[key_cols + ["Drafted?"]].copy()
update["IsDrafted"] = update["Drafted?"].map(lambda b: "Y" if b else "N")
update = update.drop(columns=["Drafted?"])

df = df.merge(update, on=key_cols, how="left", suffixes=("","_upd"))
df["IsDrafted"] = np.where(df["IsDrafted_upd"].notna(), df["IsDrafted_upd"], df["IsDrafted"])
df = df.drop(columns=[c for c in df.columns if c.endswith("_upd")], errors="ignore")

st.caption("Tip: Turn on **Only show available** in the sidebar. Checking **Drafted?** will immediately hide that player on the next refresh (Streamlit reruns automatically after edits).")

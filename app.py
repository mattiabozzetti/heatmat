from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CONFIG BASE
# ============================================================

st.set_page_config(page_title="Dual Role Radar", page_icon="📊", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
PLAYERS_FILE = DATA_DIR / "players_enriched_with_clusters.csv.gz"
GK_FILE = DATA_DIR / "gk_enriched_with_clusters.csv.gz"

AXIS_ORDER = ["Offensive", "Defensive", "Possession", "Passing"]
BIG_FIVE_LEAGUES = {"Serie A", "Premier League", "La Liga", "Bundesliga", "Ligue 1"}

ROLE_TO_BUCKET = {
    "CB": "CB",
    "FB/WB": "FB",
    "MF": "MF",
    "AM": "AM",
    "W/RML": "W",
    "FW": "FW",
}

ROLE_HELP = {
    "CB": "Centre-backs: CB, LCB, RCB",
    "FB/WB": "Full-backs / wing-backs: LB, RB, LWB/RWB logic via FB role bucket",
    "MF": "Central / defensive midfielders",
    "AM": "Attacking midfielders",
    "W/RML": "Wide players: LW, RW, LM, RM, LAM, RAM",
    "FW": "Forwards / strikers",
    "GK": "Goalkeepers",
}

# ============================================================
# METRIC HELPERS
# ============================================================

def m(column: str, higher_is_better: bool = True, adjustment: str | None = None) -> dict[str, Any]:
    """Metric shorthand.

    adjustment:
    - none: no possession adjustment
    - on_ball: volume boosted downward for high-possession teams, upward for low-possession teams
    - off_ball: defensive/off-ball volume boosted upward for high-possession teams, downward for low-possession teams
    """
    if adjustment is None:
        adjustment = infer_adjustment(column, higher_is_better)
    return {"column": column, "higher_is_better": higher_is_better, "adjustment": adjustment}


def infer_adjustment(column: str, higher_is_better: bool) -> str:
    lower = column.lower()
    quality_tokens = [", %", "%", "xgc", "xgps", "xg per", "goals prevented, %", "cross claim rate"]
    off_ball_tokens = [
        "defensive", "tackle", "interception", "recover", "air challenge", "challenge",
        "shots faced", "shots on target faced", "opponent", "cross and pass interception", "sweeping"
    ]
    if any(tok in lower for tok in quality_tokens):
        return "none"
    if any(tok in lower for tok in off_ball_tokens):
        return "off_ball"
    return "on_ball"


def clean_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"-": np.nan, "–": np.nan, "—": np.nan, "nan": np.nan, "None": np.nan, "": np.nan}),
        errors="coerce",
    )


def possession_adjust(raw: pd.Series, possession: pd.Series, adjustment: str, k: float = 8.0, gamma: float = 0.35) -> pd.Series:
    raw = pd.to_numeric(raw, errors="coerce")
    if adjustment == "none":
        return raw
    possession = pd.to_numeric(possession, errors="coerce")
    # Team possession is stored as 0-1 in the processed files.
    s = 2 / (1 + np.exp(-k * (possession - 0.50))) - 1
    if adjustment == "on_ball":
        return raw * (1 - gamma * s)
    if adjustment == "off_ball":
        return raw * (1 + gamma * s)
    return raw


def metric_series(df: pd.DataFrame, metric: dict[str, Any], mode: str) -> pd.Series:
    col = metric["column"]
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    raw = clean_numeric(df[col])
    if mode == "Possession-adjusted":
        possession = clean_numeric(df.get("Ball possession, %", pd.Series(np.nan, index=df.index)))
        return possession_adjust(raw, possession, metric.get("adjustment", "none"))
    return raw


def percentile_rank(value: float, reference_values: pd.Series, higher_is_better: bool = True) -> float:
    values = pd.to_numeric(reference_values, errors="coerce").dropna()
    if pd.isna(value) or len(values) < 3:
        return float("nan")
    pct = 100 * ((values < value).sum() + 0.5 * (values == value).sum()) / len(values)
    if not higher_is_better:
        pct = 100 - pct
    return float(np.clip(pct, 0, 100))


def axis_score(player_row: pd.Series, reference_df: pd.DataFrame, metrics: list[dict[str, Any]], mode: str) -> tuple[float, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    pcts: list[float] = []
    player_df = player_row.to_frame().T

    for metric in metrics:
        col = metric["column"]
        if col not in reference_df.columns or col not in player_row.index:
            details.append({"metric": col, "raw": np.nan, "percentile": np.nan, "used": False})
            continue
        raw_value = metric_series(player_df, metric, mode).iloc[0]
        ref_values = metric_series(reference_df, metric, mode)
        pct = percentile_rank(raw_value, ref_values, metric.get("higher_is_better", True))
        details.append({"metric": col, "raw": raw_value, "percentile": pct, "used": not math.isnan(pct)})
        if not math.isnan(pct):
            pcts.append(pct)

    return (float(np.mean(pcts)) if pcts else float("nan")), details


def compute_radar_scores(player_row: pd.Series, reference_df: pd.DataFrame, template: dict[str, Any], mode: str) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]]]:
    style_scores: dict[str, float] = {}
    perf_scores: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for axis in AXIS_ORDER:
        style_score, style_details = axis_score(player_row, reference_df, template["style"][axis], mode)
        perf_score, perf_details = axis_score(player_row, reference_df, template["performance"][axis], mode)

        style_scores[axis] = style_score
        perf_scores[axis] = perf_score

        rows.append({
            "Axis": axis,
            "Tactical meaning": template["axis_labels"][axis],
            "Style score": style_score,
            "Performance score": perf_score,
            "Style metrics used": ", ".join(d["metric"] for d in style_details if d["used"]),
            "Performance metrics used": ", ".join(d["metric"] for d in perf_details if d["used"]),
            "Missing style metrics": ", ".join(d["metric"] for d in style_details if not d["used"]),
            "Missing performance metrics": ", ".join(d["metric"] for d in perf_details if not d["used"]),
        })

    return style_scores, perf_scores, rows


def fmt_pct(value: float) -> str:
    return "—" if pd.isna(value) or math.isnan(value) else f"{value:.0f}"


def fmt_num(value: Any, digits: int = 0) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "—"

# ============================================================
# ROLE-SPECIFIC TEMPLATES
# ============================================================

RADAR_TEMPLATES: dict[str, dict[str, Any]] = {
    "CB": {
        "axis_labels": {
            "Offensive": "Set-piece / box threat",
            "Defensive": "Stopping & aerial command",
            "Possession": "Ball security under pressure",
            "Passing": "Build-up range",
        },
        "style": {
            "Offensive": [m("Actions in opponent's box"), m("Headers"), m("Goals by head"), m("xG (expected goals)")],
            "Defensive": [m("Defensive challenges", adjustment="off_ball"), m("Tackles", adjustment="off_ball"), m("Interceptions", adjustment="off_ball"), m("Air challenges", adjustment="off_ball"), m("Ball recoveries", adjustment="off_ball")],
            "Possession": [m("Actions"), m("Carry"), m("Open passes received"), m("Lost balls", False)],
            "Passing": [m("Passes"), m("Long passes"), m("Progressive passes"), m("Passes forward to the final third")],
        },
        "performance": {
            "Offensive": [m("Headers on target, %", adjustment="none"), m("Goals"), m("xGC (xG conversion)", adjustment="none"), m("Actions in opponent's box successful, %", adjustment="none")],
            "Defensive": [m("Defensive challenges won, %", adjustment="none"), m("Tackles successful, %", adjustment="none"), m("Air challenges won, %", adjustment="none"), m("Mistakes leading to chances", False, "none")],
            "Possession": [m("Actions successful, %", adjustment="none"), m("Bad ball control", False), m("Individual ball losses", False), m("Lost balls", False)],
            "Passing": [m("Passes accurate, %", adjustment="none"), m("Long passes accurate, %", adjustment="none"), m("Progressive passes accurate, %", adjustment="none"), m("Passes forward to the final third accurate, %", adjustment="none")],
        },
    },
    "FB/WB": {
        "axis_labels": {
            "Offensive": "Wide threat / box delivery",
            "Defensive": "1v1 defending & recovery",
            "Possession": "Carry & escape",
            "Passing": "Progressive supply",
        },
        "style": {
            "Offensive": [m("Crosses"), m("Passes into the penalty box"), m("Final third entries"), m("Actions in opponent's box"), m("xA")],
            "Defensive": [m("Defensive challenges", adjustment="off_ball"), m("Tackles", adjustment="off_ball"), m("Interceptions", adjustment="off_ball"), m("Ball recoveries", adjustment="off_ball"), m("Air challenges", adjustment="off_ball")],
            "Possession": [m("Carry"), m("Final third entries through carry"), m("Dribbles"), m("Open passes received in the final third")],
            "Passing": [m("Progressive passes"), m("Passes forward to the final third"), m("Key passes"), m("Passes for a shot")],
        },
        "performance": {
            "Offensive": [m("Crosses accurate, %", adjustment="none"), m("Passes into the penalty box accurate, %", adjustment="none"), m("Chances created"), m("xA")],
            "Defensive": [m("Defensive challenges won, %", adjustment="none"), m("Tackles successful, %", adjustment="none"), m("Air challenges won, %", adjustment="none"), m("Mistakes leading to chances", False, "none")],
            "Possession": [m("Dribbles successful, %", adjustment="none"), m("Actions successful, %", adjustment="none"), m("Lost balls", False), m("Bad ball control", False)],
            "Passing": [m("Progressive passes accurate, %", adjustment="none"), m("Passes forward to the final third accurate, %", adjustment="none"), m("Key passes accurate, %", adjustment="none"), m("Passes accurate, %", adjustment="none")],
        },
    },
    "MF": {
        "axis_labels": {
            "Offensive": "Chance support / final-third influence",
            "Defensive": "Ball winning & coverage",
            "Possession": "Availability & carrying",
            "Passing": "Tempo & progression",
        },
        "style": {
            "Offensive": [m("Key passes"), m("Passes for a shot"), m("Chances created"), m("xA"), m("Final third entries")],
            "Defensive": [m("Defensive challenges", adjustment="off_ball"), m("Tackles", adjustment="off_ball"), m("Interceptions", adjustment="off_ball"), m("Ball recoveries", adjustment="off_ball"), m("Loose ball recoveries", adjustment="off_ball")],
            "Possession": [m("Open passes received"), m("Open passes received in the central third"), m("Carry"), m("Final third entries through carry"), m("Actions")],
            "Passing": [m("Passes"), m("Short passes"), m("Progressive passes"), m("Passes forward to the final third"), m("Long passes")],
        },
        "performance": {
            "Offensive": [m("Key passes accurate, %", adjustment="none"), m("Chances successful, %", adjustment="none"), m("xA"), m("Goals + Assists")],
            "Defensive": [m("Defensive challenges won, %", adjustment="none"), m("Tackles successful, %", adjustment="none"), m("Challenges won, %", adjustment="none"), m("Mistakes leading to chances", False, "none")],
            "Possession": [m("Actions successful, %", adjustment="none"), m("Lost balls", False), m("Individual ball losses", False), m("Bad ball control", False)],
            "Passing": [m("Passes accurate, %", adjustment="none"), m("Short passes accurate, %", adjustment="none"), m("Progressive passes accurate, %", adjustment="none"), m("Long passes accurate, %", adjustment="none")],
        },
    },
    "AM": {
        "axis_labels": {
            "Offensive": "Final-third damage",
            "Defensive": "Counterpressing / high recovery",
            "Possession": "Between-lines receiving & 1v1",
            "Passing": "Creative passing",
        },
        "style": {
            "Offensive": [m("Goals"), m("xG (expected goals)"), m("xA"), m("Actions in opponent's box"), m("Chances created")],
            "Defensive": [m("Defensive challenges", adjustment="off_ball"), m("Tackles", adjustment="off_ball"), m("Ball recoveries in opponent's half", adjustment="off_ball"), m("Interceptions", adjustment="off_ball")],
            "Possession": [m("Open passes received in the final third"), m("Open passes received in the opponent's box"), m("Dribbles"), m("Dribbling in the final third"), m("Carry")],
            "Passing": [m("Key passes"), m("Passes for a shot"), m("Progressive passes"), m("Passes into the penalty box"), m("Passes")],
        },
        "performance": {
            "Offensive": [m("Goals + Assists"), m("xGC (xG conversion)", adjustment="none"), m("Chances successful, %", adjustment="none"), m("Actions in opponent's box successful, %", adjustment="none")],
            "Defensive": [m("Defensive challenges won, %", adjustment="none"), m("Tackles successful, %", adjustment="none"), m("Challenges won, %", adjustment="none")],
            "Possession": [m("Dribbles successful, %", adjustment="none"), m("Dribbling in the final third successful, %", adjustment="none"), m("Actions successful, %", adjustment="none"), m("Lost balls", False)],
            "Passing": [m("Key passes accurate, %", adjustment="none"), m("Progressive passes accurate, %", adjustment="none"), m("Passes into the penalty box accurate, %", adjustment="none"), m("Passes accurate, %", adjustment="none")],
        },
    },
    "W/RML": {
        "axis_labels": {
            "Offensive": "Wide-to-box threat",
            "Defensive": "Wide work rate",
            "Possession": "1v1 / carrying threat",
            "Passing": "Delivery & chance creation",
        },
        "style": {
            "Offensive": [m("Shots"), m("xG (expected goals)"), m("Actions in opponent's box"), m("Open passes received in the opponent's box"), m("Shots from the penalty area")],
            "Defensive": [m("Defensive challenges", adjustment="off_ball"), m("Tackles", adjustment="off_ball"), m("Ball recoveries", adjustment="off_ball"), m("Ball recoveries in opponent's half", adjustment="off_ball")],
            "Possession": [m("Dribbles"), m("Dribbling in the final third"), m("Carry"), m("Final third entries through carry"), m("Open passes received in the final third")],
            "Passing": [m("Crosses"), m("Key passes"), m("Passes into the penalty box"), m("Passes for a shot"), m("Progressive passes")],
        },
        "performance": {
            "Offensive": [m("Goals + Assists"), m("Shots on target, %", adjustment="none"), m("xGPS (xG per shot)", adjustment="none"), m("xGC (xG conversion)", adjustment="none")],
            "Defensive": [m("Defensive challenges won, %", adjustment="none"), m("Tackles successful, %", adjustment="none"), m("Challenges won, %", adjustment="none")],
            "Possession": [m("Dribbles successful, %", adjustment="none"), m("Dribbling in the final third successful, %", adjustment="none"), m("Actions successful, %", adjustment="none"), m("Lost balls", False)],
            "Passing": [m("Crosses accurate, %", adjustment="none"), m("Key passes accurate, %", adjustment="none"), m("Passes into the penalty box accurate, %", adjustment="none"), m("Progressive passes accurate, %", adjustment="none")],
        },
    },
    "FW": {
        "axis_labels": {
            "Offensive": "Box threat / finishing",
            "Defensive": "Pressing & physical duels",
            "Possession": "Hold-up / carrying",
            "Passing": "Link play / chance creation",
        },
        "style": {
            "Offensive": [m("Shots"), m("xG (expected goals)"), m("Shots from the penalty area"), m("Actions in opponent's box"), m("Open passes received in the opponent's box")],
            "Defensive": [m("Attacking challenges", adjustment="off_ball"), m("Defensive challenges", adjustment="off_ball"), m("Ball recoveries in opponent's half", adjustment="off_ball"), m("Air challenges", adjustment="off_ball")],
            "Possession": [m("Open passes received"), m("Open passes received in the final third"), m("Attacking challenges"), m("Dribbles"), m("Carry")],
            "Passing": [m("Passes"), m("Key passes"), m("Passes for a shot"), m("Progressive passes"), m("xA")],
        },
        "performance": {
            "Offensive": [m("Goals"), m("Shots on target, %", adjustment="none"), m("xGPS (xG per shot)", adjustment="none"), m("xGC (xG conversion)", adjustment="none"), m("Chances successful, %", adjustment="none")],
            "Defensive": [m("Attacking challenges won, %", adjustment="none"), m("Defensive challenges won, %", adjustment="none"), m("Air challenges won, %", adjustment="none"), m("Tackles successful, %", adjustment="none")],
            "Possession": [m("Actions successful, %", adjustment="none"), m("Dribbles successful, %", adjustment="none"), m("Lost balls", False), m("Bad ball control", False)],
            "Passing": [m("Passes accurate, %", adjustment="none"), m("Key passes accurate, %", adjustment="none"), m("Progressive passes accurate, %", adjustment="none"), m("xA")],
        },
    },
    "GK": {
        "axis_labels": {
            "Offensive": "Launch / transition starter",
            "Defensive": "Shot stopping & box command",
            "Possession": "Security under pressure",
            "Passing": "Build-up distribution",
        },
        "style": {
            "Offensive": [m("Progressive open passes"), m("Long passes"), m("Goal kicks long (40+ m)"), m("Throws")],
            "Defensive": [m("Shots on target faced", adjustment="off_ball"), m("Opponent's shots xG", adjustment="off_ball"), m("Opponent's crosses", adjustment="off_ball"), m("Cross and pass interception attempts", adjustment="off_ball"), m("Sweeping actions", adjustment="off_ball")],
            "Possession": [m("Actions"), m("Open play passes"), m("Sweeping actions"), m("Throws")],
            "Passing": [m("Passes"), m("Open play passes"), m("Short passes"), m("Medium passes"), m("Progressive open passes")],
        },
        "performance": {
            "Offensive": [m("Long passes accurate, %", adjustment="none"), m("Goal kicks long (40+ m) accurate, %", adjustment="none"), m("Throws accurate, %", adjustment="none")],
            "Defensive": [m("Goals prevented"), m("Goals prevented, %", adjustment="none"), m("Shots saved, %", adjustment="none"), m("Cross claim rate", adjustment="none"), m("Sweeping actions successful, %", adjustment="none")],
            "Possession": [m("Actions successful, %", adjustment="none"), m("Mistakes leading to chances", False, "none"), m("Mistakes leading to goals", False, "none"), m("Open play passes accurate, %", adjustment="none")],
            "Passing": [m("Passes accurate, %", adjustment="none"), m("Open play passes accurate, %", adjustment="none"), m("Short passes accurate, %", adjustment="none"), m("Medium passes accurate, %", adjustment="none")],
        },
    },
}

# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_outfield() -> pd.DataFrame:
    df = pd.read_csv(PLAYERS_FILE, compression="gzip")
    return standardize_base_columns(df)


@st.cache_data(show_spinner=False)
def load_gk() -> pd.DataFrame:
    df = pd.read_csv(GK_FILE, compression="gzip")
    return standardize_base_columns(df)


def standardize_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Season", "Player", "Team", "League", "Nation", "Position", "Role bucket", "GK role"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", np.nan)
    if "Minutes played" in df.columns:
        df["Minutes played"] = clean_numeric(df["Minutes played"])
    if "style_cluster_confidence" in df.columns:
        df["style_cluster_confidence"] = clean_numeric(df["style_cluster_confidence"])
    return df

# ============================================================
# REFERENCE + PLOT
# ============================================================

def build_reference(
    df: pd.DataFrame,
    role: str,
    season: str,
    player_league: str | None,
    reference_scope: str,
    custom_leagues: list[str],
    min_minutes: int,
) -> pd.DataFrame:
    ref = df[df["Season"].astype(str).eq(str(season))].copy()
    ref = ref[clean_numeric(ref["Minutes played"]).fillna(0) >= min_minutes]

    if role != "GK" and "Role bucket" in ref.columns:
        ref = ref[ref["Role bucket"].astype(str).eq(ROLE_TO_BUCKET[role])]

    if reference_scope == "Player league" and player_league:
        ref = ref[ref["League"].astype(str).eq(str(player_league))]
    elif reference_scope == "Big Five":
        ref = ref[ref["League"].isin(BIG_FIVE_LEAGUES)]
    elif reference_scope == "Custom leagues" and custom_leagues:
        ref = ref[ref["League"].isin(custom_leagues)]
    elif reference_scope == "All leagues":
        pass

    return ref


def radar_values(scores: dict[str, float], template: dict[str, Any]) -> tuple[list[str], list[float]]:
    theta = [f"{axis}<br><span style='font-size:11px'>{template['axis_labels'][axis]}</span>" for axis in AXIS_ORDER]
    values = [scores.get(axis, np.nan) for axis in AXIS_ORDER]
    theta.append(theta[0])
    values.append(values[0])
    return theta, values


def make_radar(title: str, scores: dict[str, float], template: dict[str, Any]) -> go.Figure:
    theta, values = radar_values(scores, template)
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=theta,
            fill="toself",
            mode="lines+markers",
            name=title,
            hovertemplate="%{theta}<br>Percentile: %{r:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=70, b=40),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[20, 40, 60, 80, 100]),
            angularaxis=dict(direction="clockwise", rotation=90),
        ),
        height=520,
    )
    return fig


def player_label(row: pd.Series, role: str, reference_scope: str, reference_n: int) -> None:
    name = row.get("Player", "—")
    team = row.get("Team", "—")
    league = row.get("League", "—")
    season = row.get("Season", "—")
    position = row.get("Position", row.get("GK role", "GK"))
    minutes = fmt_num(row.get("Minutes played"), 0)
    cluster = row.get("style_cluster_short_label", row.get("style_cluster_name", "—"))
    cluster_name = row.get("style_cluster_name", "—")
    conf = row.get("style_cluster_confidence", np.nan)
    conf_txt = "—" if pd.isna(conf) else f"{float(conf):.0f}%"

    st.header(str(name))
    st.caption(
        f"{team} · {league} · {season} · Pos: {position} · {minutes} minutes · "
        f"Compared as: {role} · Reference: {reference_scope} · n = {reference_n}"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Team", str(team))
    c2.metric("Minutes", minutes)
    c3.metric("Cluster", str(cluster))
    c4.metric("Confidence", conf_txt)
    if pd.notna(cluster_name) and str(cluster_name) != str(cluster):
        st.caption(f"Cluster full label: {cluster_name}")


def select_one(options: list[str], label: str, default: str | None = None) -> str:
    if not options:
        return ""
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index)

# ============================================================
# UI
# ============================================================

st.title("Dual Role Radar")
st.caption("Una app minimale: Style a sinistra, Performance a destra. Quattro assi fissi, contenuto ruolo-specifico.")

with st.sidebar:
    st.subheader("Selezione")
    role = st.selectbox("Role template", list(RADAR_TEMPLATES.keys()), help="Scegli il ruolo usato per template e percentili.")
    st.caption(ROLE_HELP[role])

    df = load_gk() if role == "GK" else load_outfield()

    seasons = sorted(df["Season"].dropna().astype(str).unique().tolist(), reverse=True)
    default_season = "2025-2026" if "2025-2026" in seasons else (seasons[0] if seasons else "")
    season = select_one(seasons, "Season", default=default_season)

    season_df = df[df["Season"].astype(str).eq(str(season))].copy()
    if role != "GK" and "Role bucket" in season_df.columns:
        # Player selection remains broad enough to allow role-forced comparison,
        # but default list is filtered to the selected role bucket.
        season_df_for_player = season_df[season_df["Role bucket"].astype(str).eq(ROLE_TO_BUCKET[role])].copy()
        if season_df_for_player.empty:
            season_df_for_player = season_df.copy()
    else:
        season_df_for_player = season_df.copy()

    leagues = ["All"] + sorted(season_df_for_player["League"].dropna().astype(str).unique().tolist())
    league_filter = st.selectbox("Filter teams by league", leagues)
    team_pool = season_df_for_player.copy()
    if league_filter != "All":
        team_pool = team_pool[team_pool["League"].astype(str).eq(league_filter)]

    teams = sorted(team_pool["Team"].dropna().astype(str).unique().tolist())
    team = select_one(teams, "Team")

    player_pool = team_pool[team_pool["Team"].astype(str).eq(str(team))].copy()
    players = sorted(player_pool["Player"].dropna().astype(str).unique().tolist())
    player = select_one(players, "Player")

    st.divider()
    st.subheader("Percentili")
    reference_scope = st.selectbox("Reference scope", ["Player league", "Big Five", "All leagues", "Custom leagues"], index=1)
    all_leagues = sorted(df["League"].dropna().astype(str).unique().tolist())
    custom_leagues: list[str] = []
    if reference_scope == "Custom leagues":
        custom_leagues = st.multiselect("Custom leagues", all_leagues, default=[l for l in all_leagues if l in BIG_FIVE_LEAGUES])
    min_minutes = st.slider("Minimum minutes", min_value=0, max_value=2500, value=600, step=100)
    mode = st.radio("Metric mode", ["Raw", "Possession-adjusted"], horizontal=False)

if not season or not team or not player:
    st.warning("Seleziona stagione, team e giocatore.")
    st.stop()

selected_rows = player_pool[player_pool["Player"].astype(str).eq(str(player))].copy()
if selected_rows.empty:
    st.error("Giocatore non trovato con i filtri selezionati.")
    st.stop()

selected_rows = selected_rows.sort_values("Minutes played", ascending=False)
player_row = selected_rows.iloc[0]
player_league = player_row.get("League", None)
reference_df = build_reference(df, role, season, player_league, reference_scope, custom_leagues, min_minutes)

template = RADAR_TEMPLATES[role]
style_scores, perf_scores, detail_rows = compute_radar_scores(player_row, reference_df, template, mode)

player_label(player_row, role, reference_scope, len(reference_df))

if len(reference_df) < 15:
    st.warning(f"Reference group piccolo: n = {len(reference_df)}. I percentili potrebbero essere instabili.")

left, right = st.columns(2)
with left:
    st.plotly_chart(make_radar("Player Style", style_scores, template), use_container_width=True)
with right:
    st.plotly_chart(make_radar("Performance", perf_scores, template), use_container_width=True)

score_df = pd.DataFrame({
    "Axis": AXIS_ORDER,
    "Meaning": [template["axis_labels"][a] for a in AXIS_ORDER],
    "Style": [style_scores[a] for a in AXIS_ORDER],
    "Performance": [perf_scores[a] for a in AXIS_ORDER],
})
score_df["Style"] = score_df["Style"].round(0).astype("Int64")
score_df["Performance"] = score_df["Performance"].round(0).astype("Int64")

st.subheader("Axis scores")
st.dataframe(score_df, hide_index=True, use_container_width=True)

with st.expander("Metriche usate per ogni asse"):
    details_df = pd.DataFrame(detail_rows)
    for col in ["Style score", "Performance score"]:
        details_df[col] = details_df[col].round(1)
    st.dataframe(details_df, hide_index=True, use_container_width=True)

with st.expander("Reference group"):
    cols = [c for c in ["Player", "Team", "League", "Season", "Position", "Role bucket", "Minutes played", "style_cluster_short_label"] if c in reference_df.columns]
    st.dataframe(reference_df[cols].sort_values("Minutes played", ascending=False), hide_index=True, use_container_width=True)

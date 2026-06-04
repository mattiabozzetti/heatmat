from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from mplsoccer import PyPizza
except ImportError as exc:  # pragma: no cover
    st.error("Manca mplsoccer. Installa le dipendenze con: pip install -r requirements.txt")
    raise exc

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(page_title="Dual Role Pizza Radar", page_icon="📊", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
PLAYERS_FILE = DATA_DIR / "players_enriched_with_clusters.csv.gz"
GK_FILE = DATA_DIR / "gk_enriched_with_clusters.csv.gz"

BIG_FIVE_LEAGUES = {"Serie A", "Premier League", "La Liga", "Bundesliga", "Ligue 1"}
BIG_FIVE_COMPETITIONS = {
    ("Serie A", "Italy"),
    ("Premier League", "England"),
    ("La Liga", "Spain"),
    ("Bundesliga", "Germany"),
    ("Ligue 1", "France"),
}
LEAGUE_DISPLAY_COL = "League display"

# Stile grafico coerente con il template caricato.
FIG_BG = "#FFFFFF"   # richiesto: background grafico white
LINE_BLACK = "#000000"
CIRCLE_GREY = "#A6A6A6"
GREEN = "#3CB37C"   # passing / creation
YELLOW = "#F2D529"  # possession / carrying
BLUE = "#2A86B1"    # attack / scoring
RED = "#EF5A5A"     # defense / duel / inverse security

FAMILY_COLORS = {
    "attack": BLUE,
    "possession": YELLOW,
    "passing": GREEN,
    "defense": RED,
}

ROLE_TO_BUCKET = {
    "CB": "CB",
    "FB/WB": "FB",
    "MF": "MF",
    "AM": "AM",
    "W/RML": "W",
    "FW": "FW",
}

ROLE_HELP = {
    "CB": "Difensori centrali: CB, LCB, RCB",
    "FB/WB": "Terzini/quinti: LB, RB e role bucket FB",
    "MF": "Centrocampisti centrali / mediani",
    "AM": "Trequartisti / attacking midfielders",
    "W/RML": "Esterni: LW, RW, LM, RM, LAM, RAM",
    "FW": "Attaccanti / prime punte",
    "GK": "Portieri",
}

# ============================================================
# METRICHE
# ============================================================

def metric(label: str, column: str, family: str, reverse: bool = False, adjustment: str | None = None) -> dict[str, Any]:
    """Definisce una singola metrica del pizza/radar.

    family: attack, possession, passing, defense.
    reverse=True quando valori più bassi sono migliori.
    adjustment: none, on_ball, off_ball. Se None viene inferito.
    """
    if adjustment is None:
        adjustment = infer_adjustment(column, reverse)
    return {
        "label": label,
        "column": column,
        "family": family,
        "reverse": reverse,
        "adjustment": adjustment,
    }


def infer_adjustment(column: str, reverse: bool = False) -> str:
    lower = column.lower()
    quality_tokens = [", %", "%", "xgc", "xgps", "xg per", "goals prevented", "cross claim rate"]
    off_ball_tokens = [
        "defensive", "tackle", "interception", "recover", "air challenge", "challenge",
        "shots faced", "shots on target faced", "opponent", "cross and pass interception", "sweeping",
    ]
    if any(tok in lower for tok in quality_tokens):
        return "none"
    if any(tok in lower for tok in off_ball_tokens):
        return "off_ball"
    if reverse:
        return "none"
    return "on_ball"


m = metric

# ============================================================
# ROLE-SPECIFIC TEMPLATES
# Ordine identico al template caricato: ATTACK -> POSSESSION -> PASSING -> DEFENSE
# Ogni ruolo mantiene la stessa struttura visuale ma cambia metriche.
# ============================================================

ROLE_TEMPLATES: dict[str, dict[str, Any]] = {
    "FW": {
        "display": "Forward / Striker",
        "description": "Box threat, hold-up, link play and pressing/duels.",
        "style": [
            # ATTACK
            m("Shots p90", "Shots", "attack"),
            m("xG p90", "xG (expected goals)", "attack"),
            m("Box shots\np90", "Shots from the penalty area", "attack"),
            m("Box actions\np90", "Actions in opponent's box", "attack"),
            # POSSESSION
            m("Open passes\nreceived p90", "Open passes received", "possession"),
            m("Final 3rd\nreceiving p90", "Open passes received in the final third", "possession"),
            m("Box\nreceptions p90", "Open passes received in the opponent's box", "possession"),
            m("Attacking\nduels p90", "Attacking challenges", "possession"),
            # PASSING
            m("Passes p90", "Passes", "passing"),
            m("Key passes\np90", "Key passes", "passing"),
            m("Passes for\nshot p90", "Passes for a shot", "passing"),
            m("xA p90", "xA", "passing"),
            # DEFENSE
            m("Def. duels\np90", "Defensive challenges", "defense", adjustment="off_ball"),
            m("Opp. half\nrecoveries p90", "Ball recoveries in opponent's half", "defense", adjustment="off_ball"),
            m("Aerial duels\np90", "Air challenges", "defense", adjustment="off_ball"),
            m("Tackles p90", "Tackles", "defense", adjustment="off_ball"),
        ],
        "performance": [
            # ATTACK
            m("Goals p90", "Goals", "attack"),
            m("Shots on\ntarget %", "Shots on target, %", "attack", adjustment="none"),
            m("xG per\nshot", "xGPS (xG per shot)", "attack", adjustment="none"),
            m("xG\nconversion", "xGC (xG conversion)", "attack", adjustment="none"),
            # POSSESSION
            m("Actions\nsuccess %", "Actions successful, %", "possession", adjustment="none"),
            m("Dribble\nsuccess %", "Dribbles successful, %", "possession", adjustment="none"),
            m("Attacking\nduel win %", "Attacking challenges won, %", "possession", adjustment="none"),
            m("Lost balls\ninverted", "Lost balls", "possession", reverse=True),
            # PASSING
            m("Passes\naccuracy %", "Passes accurate, %", "passing", adjustment="none"),
            m("Key pass\naccuracy %", "Key passes accurate, %", "passing", adjustment="none"),
            m("Assists p90", "Assists", "passing"),
            m("xA p90", "xA", "passing"),
            # DEFENSE
            m("Def. duel\nwin %", "Defensive challenges won, %", "defense", adjustment="none"),
            m("Aerial\nwin %", "Air challenges won, %", "defense", adjustment="none"),
            m("Tackle\nsuccess %", "Tackles successful, %", "defense", adjustment="none"),
            m("Challenges\nwon %", "Challenges won, %", "defense", adjustment="none"),
        ],
    },
    "W/RML": {
        "display": "Winger / Wide Midfielder",
        "description": "Wide-to-box threat, 1v1, delivery and wide work rate.",
        "style": [
            m("Shots p90", "Shots", "attack"),
            m("xG p90", "xG (expected goals)", "attack"),
            m("Box shots\np90", "Shots from the penalty area", "attack"),
            m("Box actions\np90", "Actions in opponent's box", "attack"),
            m("Dribbles p90", "Dribbles", "possession"),
            m("Final 3rd\ndribbles p90", "Dribbling in the final third", "possession"),
            m("Carries p90", "Carry", "possession"),
            m("Final 3rd\ncarries p90", "Final third entries through carry", "possession"),
            m("Crosses p90", "Crosses", "passing"),
            m("Key passes\np90", "Key passes", "passing"),
            m("Box passes\np90", "Passes into the penalty box", "passing"),
            m("Passes for\nshot p90", "Passes for a shot", "passing"),
            m("Def. duels\np90", "Defensive challenges", "defense", adjustment="off_ball"),
            m("Tackles p90", "Tackles", "defense", adjustment="off_ball"),
            m("Recoveries p90", "Ball recoveries", "defense", adjustment="off_ball"),
            m("Opp. half\nrecoveries p90", "Ball recoveries in opponent's half", "defense", adjustment="off_ball"),
        ],
        "performance": [
            m("Goals+Assists\np90", "Goals + Assists", "attack"),
            m("Shots on\ntarget %", "Shots on target, %", "attack", adjustment="none"),
            m("xG per\nshot", "xGPS (xG per shot)", "attack", adjustment="none"),
            m("xG\nconversion", "xGC (xG conversion)", "attack", adjustment="none"),
            m("Dribble\nsuccess %", "Dribbles successful, %", "possession", adjustment="none"),
            m("Final 3rd\ndribble success %", "Dribbling in the final third successful, %", "possession", adjustment="none"),
            m("Actions\nsuccess %", "Actions successful, %", "possession", adjustment="none"),
            m("Lost balls\ninverted", "Lost balls", "possession", reverse=True),
            m("Cross\naccuracy %", "Crosses accurate, %", "passing", adjustment="none"),
            m("Key pass\naccuracy %", "Key passes accurate, %", "passing", adjustment="none"),
            m("Box pass\naccuracy %", "Passes into the penalty box accurate, %", "passing", adjustment="none"),
            m("Progressive pass\naccuracy %", "Progressive passes accurate, %", "passing", adjustment="none"),
            m("Def. duel\nwin %", "Defensive challenges won, %", "defense", adjustment="none"),
            m("Tackle\nsuccess %", "Tackles successful, %", "defense", adjustment="none"),
            m("Challenges\nwon %", "Challenges won, %", "defense", adjustment="none"),
            m("Mistakes\ninverted", "Mistakes leading to chances", "defense", reverse=True, adjustment="none"),
        ],
    },
    "AM": {
        "display": "Attacking Midfielder",
        "description": "Final-third damage, between-lines receiving, creative passing and counterpressing.",
        "style": [
            m("Shots p90", "Shots", "attack"),
            m("xG p90", "xG (expected goals)", "attack"),
            m("xA p90", "xA", "attack"),
            m("Box actions\np90", "Actions in opponent's box", "attack"),
            m("Final 3rd\nreceiving p90", "Open passes received in the final third", "possession"),
            m("Box\nreceptions p90", "Open passes received in the opponent's box", "possession"),
            m("Dribbles p90", "Dribbles", "possession"),
            m("Carries p90", "Carry", "possession"),
            m("Key passes\np90", "Key passes", "passing"),
            m("Passes for\nshot p90", "Passes for a shot", "passing"),
            m("Progressive\npasses p90", "Progressive passes", "passing"),
            m("Box passes\np90", "Passes into the penalty box", "passing"),
            m("Def. duels\np90", "Defensive challenges", "defense", adjustment="off_ball"),
            m("Tackles p90", "Tackles", "defense", adjustment="off_ball"),
            m("Opp. half\nrecoveries p90", "Ball recoveries in opponent's half", "defense", adjustment="off_ball"),
            m("Interceptions\np90", "Interceptions", "defense", adjustment="off_ball"),
        ],
        "performance": [
            m("Goals+Assists\np90", "Goals + Assists", "attack"),
            m("xG\nconversion", "xGC (xG conversion)", "attack", adjustment="none"),
            m("xG per\nshot", "xGPS (xG per shot)", "attack", adjustment="none"),
            m("Chances\nsuccess %", "Chances successful, %", "attack", adjustment="none"),
            m("Dribble\nsuccess %", "Dribbles successful, %", "possession", adjustment="none"),
            m("Final 3rd\ndribble success %", "Dribbling in the final third successful, %", "possession", adjustment="none"),
            m("Actions\nsuccess %", "Actions successful, %", "possession", adjustment="none"),
            m("Lost balls\ninverted", "Lost balls", "possession", reverse=True),
            m("Key pass\naccuracy %", "Key passes accurate, %", "passing", adjustment="none"),
            m("Progressive pass\naccuracy %", "Progressive passes accurate, %", "passing", adjustment="none"),
            m("Box pass\naccuracy %", "Passes into the penalty box accurate, %", "passing", adjustment="none"),
            m("Passes\naccuracy %", "Passes accurate, %", "passing", adjustment="none"),
            m("Def. duel\nwin %", "Defensive challenges won, %", "defense", adjustment="none"),
            m("Tackle\nsuccess %", "Tackles successful, %", "defense", adjustment="none"),
            m("Challenges\nwon %", "Challenges won, %", "defense", adjustment="none"),
            m("Mistakes\ninverted", "Mistakes leading to chances", "defense", reverse=True, adjustment="none"),
        ],
    },
    "MF": {
        "display": "Midfielder",
        "description": "Tempo, progression, availability and ball winning.",
        "style": [
            m("Key passes\np90", "Key passes", "attack"),
            m("Passes for\nshot p90", "Passes for a shot", "attack"),
            m("Chances\ncreated p90", "Chances created", "attack"),
            m("xA p90", "xA", "attack"),
            m("Open passes\nreceived p90", "Open passes received", "possession"),
            m("Central 3rd\nreceiving p90", "Open passes received in the central third", "possession"),
            m("Carries p90", "Carry", "possession"),
            m("Actions p90", "Actions", "possession"),
            m("Passes p90", "Passes", "passing"),
            m("Short passes\np90", "Short passes", "passing"),
            m("Progressive\npasses p90", "Progressive passes", "passing"),
            m("Long passes\np90", "Long passes", "passing"),
            m("Def. duels\np90", "Defensive challenges", "defense", adjustment="off_ball"),
            m("Tackles p90", "Tackles", "defense", adjustment="off_ball"),
            m("Interceptions\np90", "Interceptions", "defense", adjustment="off_ball"),
            m("Recoveries p90", "Ball recoveries", "defense", adjustment="off_ball"),
        ],
        "performance": [
            m("Key pass\naccuracy %", "Key passes accurate, %", "attack", adjustment="none"),
            m("Chances\nsuccess %", "Chances successful, %", "attack", adjustment="none"),
            m("xA p90", "xA", "attack"),
            m("Goals+Assists\np90", "Goals + Assists", "attack"),
            m("Actions\nsuccess %", "Actions successful, %", "possession", adjustment="none"),
            m("Lost balls\ninverted", "Lost balls", "possession", reverse=True),
            m("Individual losses\ninverted", "Individual ball losses", "possession", reverse=True),
            m("Bad control\ninverted", "Bad ball control", "possession", reverse=True),
            m("Passes\naccuracy %", "Passes accurate, %", "passing", adjustment="none"),
            m("Short pass\naccuracy %", "Short passes accurate, %", "passing", adjustment="none"),
            m("Progressive pass\naccuracy %", "Progressive passes accurate, %", "passing", adjustment="none"),
            m("Long pass\naccuracy %", "Long passes accurate, %", "passing", adjustment="none"),
            m("Def. duel\nwin %", "Defensive challenges won, %", "defense", adjustment="none"),
            m("Tackle\nsuccess %", "Tackles successful, %", "defense", adjustment="none"),
            m("Challenges\nwon %", "Challenges won, %", "defense", adjustment="none"),
            m("Mistakes\ninverted", "Mistakes leading to chances", "defense", reverse=True, adjustment="none"),
        ],
    },
    "FB/WB": {
        "display": "Full-back / Wing-back",
        "description": "Wide delivery, progression, carrying and 1v1 defending.",
        "style": [
            m("Crosses p90", "Crosses", "attack"),
            m("Box passes\np90", "Passes into the penalty box", "attack"),
            m("Final 3rd\nentries p90", "Final third entries", "attack"),
            m("Box actions\np90", "Actions in opponent's box", "attack"),
            m("Carries p90", "Carry", "possession"),
            m("Final 3rd\ncarries p90", "Final third entries through carry", "possession"),
            m("Dribbles p90", "Dribbles", "possession"),
            m("Final 3rd\nreceiving p90", "Open passes received in the final third", "possession"),
            m("Progressive\npasses p90", "Progressive passes", "passing"),
            m("Final 3rd\npasses p90", "Passes forward to the final third", "passing"),
            m("Key passes\np90", "Key passes", "passing"),
            m("Passes for\nshot p90", "Passes for a shot", "passing"),
            m("Def. duels\np90", "Defensive challenges", "defense", adjustment="off_ball"),
            m("Tackles p90", "Tackles", "defense", adjustment="off_ball"),
            m("Interceptions\np90", "Interceptions", "defense", adjustment="off_ball"),
            m("Recoveries p90", "Ball recoveries", "defense", adjustment="off_ball"),
        ],
        "performance": [
            m("Cross\naccuracy %", "Crosses accurate, %", "attack", adjustment="none"),
            m("Box pass\naccuracy %", "Passes into the penalty box accurate, %", "attack", adjustment="none"),
            m("xA p90", "xA", "attack"),
            m("Box action\nsuccess %", "Actions in opponent's box successful, %", "attack", adjustment="none"),
            m("Dribble\nsuccess %", "Dribbles successful, %", "possession", adjustment="none"),
            m("Actions\nsuccess %", "Actions successful, %", "possession", adjustment="none"),
            m("Lost balls\ninverted", "Lost balls", "possession", reverse=True),
            m("Bad control\ninverted", "Bad ball control", "possession", reverse=True),
            m("Progressive pass\naccuracy %", "Progressive passes accurate, %", "passing", adjustment="none"),
            m("Final 3rd pass\naccuracy %", "Passes forward to the final third accurate, %", "passing", adjustment="none"),
            m("Key pass\naccuracy %", "Key passes accurate, %", "passing", adjustment="none"),
            m("Passes\naccuracy %", "Passes accurate, %", "passing", adjustment="none"),
            m("Def. duel\nwin %", "Defensive challenges won, %", "defense", adjustment="none"),
            m("Tackle\nsuccess %", "Tackles successful, %", "defense", adjustment="none"),
            m("Challenges\nwon %", "Challenges won, %", "defense", adjustment="none"),
            m("Mistakes\ninverted", "Mistakes leading to chances", "defense", reverse=True, adjustment="none"),
        ],
    },
    "CB": {
        "display": "Centre-back",
        "description": "Stopping, aerial command, build-up range and security.",
        "style": [
            m("Box actions\np90", "Actions in opponent's box", "attack"),
            m("Headers p90", "Headers", "attack"),
            m("Head goals\np90", "Goals by head", "attack"),
            m("xG p90", "xG (expected goals)", "attack"),
            m("Actions p90", "Actions", "possession"),
            m("Carries p90", "Carry", "possession"),
            m("Open passes\nreceived p90", "Open passes received", "possession"),
            m("Final 3rd\ncarries p90", "Final third entries through carry", "possession"),
            m("Passes p90", "Passes", "passing"),
            m("Long passes\np90", "Long passes", "passing"),
            m("Progressive\npasses p90", "Progressive passes", "passing"),
            m("Final 3rd\npasses p90", "Passes forward to the final third", "passing"),
            m("Def. duels\np90", "Defensive challenges", "defense", adjustment="off_ball"),
            m("Tackles p90", "Tackles", "defense", adjustment="off_ball"),
            m("Interceptions\np90", "Interceptions", "defense", adjustment="off_ball"),
            m("Aerial duels\np90", "Air challenges", "defense", adjustment="off_ball"),
        ],
        "performance": [
            m("Header target %", "Headers on target, %", "attack", adjustment="none"),
            m("Goals p90", "Goals", "attack"),
            m("xG per\nshot", "xGPS (xG per shot)", "attack", adjustment="none"),
            m("xG\nconversion", "xGC (xG conversion)", "attack", adjustment="none"),
            m("Actions\nsuccess %", "Actions successful, %", "possession", adjustment="none"),
            m("Bad control\ninverted", "Bad ball control", "possession", reverse=True),
            m("Individual losses\ninverted", "Individual ball losses", "possession", reverse=True),
            m("Lost balls\ninverted", "Lost balls", "possession", reverse=True),
            m("Passes\naccuracy %", "Passes accurate, %", "passing", adjustment="none"),
            m("Long pass\naccuracy %", "Long passes accurate, %", "passing", adjustment="none"),
            m("Progressive pass\naccuracy %", "Progressive passes accurate, %", "passing", adjustment="none"),
            m("Final 3rd pass\naccuracy %", "Passes forward to the final third accurate, %", "passing", adjustment="none"),
            m("Def. duel\nwin %", "Defensive challenges won, %", "defense", adjustment="none"),
            m("Tackle\nsuccess %", "Tackles successful, %", "defense", adjustment="none"),
            m("Aerial\nwin %", "Air challenges won, %", "defense", adjustment="none"),
            m("Mistakes\ninverted", "Mistakes leading to chances", "defense", reverse=True, adjustment="none"),
        ],
    },
    "GK": {
        "display": "Goalkeeper",
        "description": "Shot stopping, box command, build-up distribution and launch profile.",
        "style": [
            m("Progressive\nopen passes", "Progressive open passes", "attack"),
            m("Long passes", "Long passes", "attack"),
            m("Long goal\nkicks", "Goal kicks long (40+ m)", "attack"),
            m("Throws", "Throws", "attack"),
            m("Actions", "Actions", "possession"),
            m("Open play\npasses", "Open play passes", "possession"),
            m("Sweeping\nactions", "Sweeping actions", "possession", adjustment="off_ball"),
            m("Set-piece\npasses", "Passes from set pieces", "possession"),
            m("Passes", "Passes", "passing"),
            m("Short passes", "Short passes", "passing"),
            m("Medium passes", "Medium passes", "passing"),
            m("Long passes", "Long passes", "passing"),
            m("Shots on target\nfaced", "Shots on target faced", "defense", adjustment="off_ball"),
            m("Opp. shots\nxG", "Opponent's shots xG", "defense", adjustment="off_ball"),
            m("Opp. crosses", "Opponent's crosses", "defense", adjustment="off_ball"),
            m("Cross/pass\ninterceptions", "Cross and pass interception attempts", "defense", adjustment="off_ball"),
        ],
        "performance": [
            m("Long pass\naccuracy %", "Long passes accurate, %", "attack", adjustment="none"),
            m("Long goal kick\naccuracy %", "Goal kicks long (40+ m) accurate, %", "attack", adjustment="none"),
            m("Throws\naccuracy %", "Throws accurate, %", "attack", adjustment="none"),
            m("Set-piece pass\naccuracy %", "Set-piece passes accurate, %", "attack", adjustment="none"),
            m("Actions\nsuccess %", "Actions successful, %", "possession", adjustment="none"),
            m("Open play pass\naccuracy %", "Open play passes accurate, %", "possession", adjustment="none"),
            m("Sweeping\nsuccess %", "Sweeping actions successful, %", "possession", adjustment="none"),
            m("Mistakes\ninverted", "Mistakes leading to chances", "possession", reverse=True, adjustment="none"),
            m("Passes\naccuracy %", "Passes accurate, %", "passing", adjustment="none"),
            m("Short pass\naccuracy %", "Short passes accurate, %", "passing", adjustment="none"),
            m("Medium pass\naccuracy %", "Medium passes accurate, %", "passing", adjustment="none"),
            m("Long pass\naccuracy %", "Long passes accurate, %", "passing", adjustment="none"),
            m("Goals\nprevented", "Goals prevented", "defense", adjustment="none"),
            m("Goals\nprevented %", "Goals prevented, %", "defense", adjustment="none"),
            m("Shots\nsaved %", "Shots saved, %", "defense", adjustment="none"),
            m("Cross claim\nrate", "Cross claim rate", "defense", adjustment="none"),
        ],
    },
}

# ============================================================
# DATA
# ============================================================

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


@st.cache_data(show_spinner=False)
def load_outfield() -> pd.DataFrame:
    df = pd.read_csv(PLAYERS_FILE, compression="gzip", low_memory=False)
    return standardize_base_columns(df)


@st.cache_data(show_spinner=False)
def load_gk() -> pd.DataFrame:
    df = pd.read_csv(GK_FILE, compression="gzip", low_memory=False)
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
    if "League" in df.columns:
        df[LEAGUE_DISPLAY_COL] = build_league_display(df)
    return df


def build_league_display(df: pd.DataFrame) -> pd.Series:
    league = df["League"].astype(str).replace("nan", np.nan) if "League" in df.columns else pd.Series("", index=df.index)
    if "Nation" not in df.columns:
        return league
    nation = df["Nation"].astype(str).replace("nan", np.nan)
    has_nation = nation.notna() & nation.str.strip().ne("")
    return league.where(~has_nation, league + " (" + nation + ")")


def league_display_values(df: pd.DataFrame) -> list[str]:
    if LEAGUE_DISPLAY_COL not in df.columns and "League" in df.columns:
        df = df.copy()
        df[LEAGUE_DISPLAY_COL] = build_league_display(df)
    if LEAGUE_DISPLAY_COL not in df.columns:
        return []
    cols = [c for c in ["Nation", "League", LEAGUE_DISPLAY_COL] if c in df.columns]
    values = (
        df[cols]
        .dropna(subset=[LEAGUE_DISPLAY_COL])
        .drop_duplicates()
        .sort_values([c for c in ["Nation", "League", LEAGUE_DISPLAY_COL] if c in cols])
    )
    return values[LEAGUE_DISPLAY_COL].astype(str).tolist()


def big_five_mask(df: pd.DataFrame) -> pd.Series:
    if "League" not in df.columns:
        return pd.Series(False, index=df.index)
    league = df["League"].astype(str)
    if "Nation" in df.columns:
        nation = df["Nation"].astype(str)
        mask = pd.Series(False, index=df.index)
        for league_name, country in BIG_FIVE_COMPETITIONS:
            mask = mask | (league.eq(league_name) & nation.eq(country))
        return mask
    return league.isin(BIG_FIVE_LEAGUES)


def big_five_league_display_values(df: pd.DataFrame) -> list[str]:
    if "League" not in df.columns:
        return []
    return league_display_values(df[big_five_mask(df)].copy())

# ============================================================
# CALCOLO PERCENTILI

# ============================================================

def possession_adjust(raw: pd.Series, possession: pd.Series, adjustment: str, k: float = 8.0, gamma: float = 0.35) -> pd.Series:
    raw = pd.to_numeric(raw, errors="coerce")
    if adjustment == "none":
        return raw
    possession = pd.to_numeric(possession, errors="coerce")
    # Ball possession è 0-1 nei file processati.
    s = 2 / (1 + np.exp(-k * (possession - 0.50))) - 1
    if adjustment == "on_ball":
        return raw * (1 - gamma * s)
    if adjustment == "off_ball":
        return raw * (1 + gamma * s)
    return raw


# Metriche che NON vanno mai corrette per possesso: sono output/qualità,
# non semplici volumi di coinvolgimento. Questo evita di normalizzare xG, xA,
# Goals, Assists, percentuali e conversioni.
NO_POSSESSION_ADJUST_TOKENS = [
    "%",
    "accurate",
    "accuracy",
    "success",
    "successful",
    "won",
    "conversion",
    "xg",
    "xa",
    "goal",
    "assist",
    "shots on target",
    "goals prevented",
    "saved",
    "cross claim rate",
    "mistakes",
    "lost balls",
    "bad ball control",
    "individual ball losses",
]


def effective_adjustment(spec: dict[str, Any], mode: str, section: str) -> str:
    """Restituisce la correzione realmente applicata.

    La modalità possession-adjusted corregge SOLO le metriche di Player Style
    che sono volumi/opportunità. Le metriche di Performance e gli output
    offensivi tipo xG/xA/goals/assists restano raw, come nella Player Card.
    """
    requested = spec.get("adjustment", "none")
    if mode != "Possession-adjusted":
        return "none"
    if section.lower() != "style":
        return "none"
    if requested == "none":
        return "none"

    col = str(spec.get("column", "")).lower()
    label = str(spec.get("label", "")).lower()
    combined = f"{col} {label}"
    if any(tok in combined for tok in NO_POSSESSION_ADJUST_TOKENS):
        return "none"
    return requested


def metric_series(df: pd.DataFrame, spec: dict[str, Any], mode: str, section: str = "style") -> pd.Series:
    col = spec["column"]
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    raw = clean_numeric(df[col])
    adj = effective_adjustment(spec, mode, section)
    if adj != "none":
        possession = clean_numeric(df.get("Ball possession, %", pd.Series(np.nan, index=df.index)))
        return possession_adjust(raw, possession, adj)
    return raw


def pct_rank(value: float, reference_values: pd.Series, reverse: bool = False) -> float:
    values = pd.to_numeric(reference_values, errors="coerce").dropna()
    if pd.isna(value) or len(values) < 3:
        return np.nan
    pct = 100 * ((values < value).sum() + 0.5 * (values == value).sum()) / len(values)
    if reverse:
        pct = 100 - pct
    return float(np.clip(pct, 0, 100))


def compute_metric_values(
    player_row: pd.Series,
    reference_df: pd.DataFrame,
    specs: list[dict[str, Any]],
    mode: str,
    section: str,
) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    labels: list[str] = []
    values: list[int] = []
    records: list[dict[str, Any]] = []
    player_df = player_row.to_frame().T

    for spec in specs:
        col = spec["column"]
        if col not in reference_df.columns or col not in player_row.index:
            records.append({
                "Metric label": spec["label"].replace("\n", " "),
                "Column": col,
                "Family": spec["family"],
                "Raw value": np.nan,
                "Percentile": np.nan,
                "Used": False,
            })
            continue

        raw_value = metric_series(player_df, spec, mode, section=section).iloc[0]
        ref_values = metric_series(reference_df, spec, mode, section=section)
        percentile = pct_rank(raw_value, ref_values, reverse=spec.get("reverse", False))

        records.append({
            "Metric label": spec["label"].replace("\n", " "),
            "Column": col,
            "Family": spec["family"],
            "Raw value": raw_value,
            "Percentile": percentile,
            "Reverse": bool(spec.get("reverse", False)),
            "Requested adjustment": spec.get("adjustment", "none"),
            "Applied adjustment": effective_adjustment(spec, mode, section),
            "Used": not pd.isna(percentile),
        })

        if not pd.isna(percentile):
            labels.append(spec["label"])
            values.append(int(round(percentile)))

    return labels, values, records


def colors_for(specs: list[dict[str, Any]], used_records: list[dict[str, Any]]) -> list[str]:
    # I colori seguono l'ordine delle metriche effettivamente usate.
    out: list[str] = []
    for spec, rec in zip(specs, used_records):
        if rec.get("Used", False):
            out.append(FAMILY_COLORS[spec["family"]])
    return out

# ============================================================
# REFERENCE GROUP
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
        if LEAGUE_DISPLAY_COL in ref.columns:
            ref = ref[ref[LEAGUE_DISPLAY_COL].astype(str).eq(str(player_league))]
        else:
            ref = ref[ref["League"].astype(str).eq(str(player_league))]
    elif reference_scope == "Big Five":
        ref = ref[big_five_mask(ref)]
    elif reference_scope == "Custom leagues" and custom_leagues:
        if LEAGUE_DISPLAY_COL in ref.columns:
            ref = ref[ref[LEAGUE_DISPLAY_COL].astype(str).isin(custom_leagues)]
        else:
            ref = ref[ref["League"].astype(str).isin(custom_leagues)]
    elif reference_scope == "All leagues":
        pass

    return ref

# ============================================================
# PLOT — STESSA STRUTTURA DEL TEMPLATE CARICATO
# ============================================================

def make_dual_pizza_figure(
    player_row: pd.Series,
    role: str,
    reference_df: pd.DataFrame,
    style_params: list[str],
    style_values: list[int],
    style_colors: list[str],
    perf_params: list[str],
    perf_values: list[int],
    perf_colors: list[str],
    mode: str,
    reference_scope: str,
) -> plt.Figure:
    fig = plt.figure(figsize=(20, 14), facecolor=FIG_BG)
    ax1 = fig.add_axes([0.05, 0.10, 0.40, 0.63], projection="polar")
    ax2 = fig.add_axes([0.55, 0.10, 0.40, 0.63], projection="polar")

    common_pizza_kwargs = dict(
        background_color=FIG_BG,
        straight_line_color=LINE_BLACK,
        straight_line_lw=1.1,
        last_circle_color=LINE_BLACK,
        last_circle_lw=2.4,
        other_circle_color=CIRCLE_GREY,
        other_circle_lw=0.8,
    )

    pizza1 = PyPizza(params=style_params, **common_pizza_kwargs)
    pizza1.make_pizza(
        style_values,
        ax=ax1,
        color_blank_space="same",
        slice_colors=style_colors[:len(style_values)],
        value_bck_colors=style_colors[:len(style_values)],
        blank_alpha=0.15,
        kwargs_slices=dict(edgecolor="black", linewidth=0.8),
        kwargs_params=dict(color="black", fontsize=11),
        kwargs_values=dict(
            color="black", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=1),
        ),
    )

    pizza2 = PyPizza(params=perf_params, **common_pizza_kwargs)
    pizza2.make_pizza(
        perf_values,
        ax=ax2,
        color_blank_space="same",
        slice_colors=perf_colors[:len(perf_values)],
        value_bck_colors=perf_colors[:len(perf_values)],
        blank_alpha=0.15,
        kwargs_slices=dict(edgecolor="black", linewidth=0.8),
        kwargs_params=dict(color="black", fontsize=11),
        kwargs_values=dict(
            color="black", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=1),
        ),
    )

    name = str(player_row.get("Player", "—"))
    age = player_row.get("Age", np.nan)
    age_txt = f" ({int(age)})" if pd.notna(age) else ""
    team = str(player_row.get("Team", "—"))
    league = str(player_row.get(LEAGUE_DISPLAY_COL, player_row.get("League", "—")))
    season = str(player_row.get("Season", "—"))
    minutes = player_row.get("Minutes played", np.nan)
    minutes_txt = f"{int(minutes):,}" if pd.notna(minutes) else "NA"
    position = str(player_row.get("Position", player_row.get("GK role", "GK")))
    cluster = str(player_row.get("style_cluster_short_label", player_row.get("style_cluster_name", "—")))
    cohort_txt = f"{reference_scope} {role} cohort"

    fig.text(0.5, 0.93, f"{name}{age_txt} - {team}", ha="center", va="center", fontsize=40, fontweight="bold")
    fig.text(
        0.5, 0.875,
        f"{league} | Season {season} | Pos {position} | {minutes_txt} minutes | Compared as {role} | n = {len(reference_df)} | {mode}",
        ha="center", va="center", fontsize=16,
    )
    fig.text(0.5, 0.845, f"Cluster: {cluster}", ha="center", va="center", fontsize=15)

    fig.text(0.25, 0.81, "Player Style", ha="center", va="center", fontsize=26, fontweight="bold")
    fig.text(0.75, 0.81, "Performance", ha="center", va="center", fontsize=26, fontweight="bold")

    # Legenda compatta centrata, nello stile del template originale.
    fig.text(0.44, 0.785, "Passing", fontsize=16, ha="right")
    fig.text(0.445, 0.785, "■", fontsize=20, color=GREEN, ha="left")
    fig.text(0.56, 0.785, "Possession", fontsize=16, ha="right")
    fig.text(0.565, 0.785, "■", fontsize=20, color=YELLOW, ha="left")
    fig.text(0.44, 0.755, "Attack", fontsize=16, ha="right")
    fig.text(0.445, 0.755, "■", fontsize=20, color=BLUE, ha="left")
    fig.text(0.56, 0.755, "Defense", fontsize=16, ha="right")
    fig.text(0.565, 0.755, "■", fontsize=20, color=RED, ha="left")

    fig.text(
        0.02, 0.03,
        f"Percentile rank vs. {cohort_txt} | possession-adjusted only on Player Style volume metrics | inverted metrics: lower is better",
        ha="left", va="bottom", fontsize=11,
    )

    return fig


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 300) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

# ============================================================
# UI HELPERS
# ============================================================

def fmt_num(value: Any, digits: int = 0) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "—"


def select_one(options: list[str], label: str, default: str | None = None) -> str:
    if not options:
        return ""
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index)


def player_header(row: pd.Series, role: str, reference_scope: str, reference_n: int) -> None:
    name = row.get("Player", "—")
    team = row.get("Team", "—")
    league = row.get(LEAGUE_DISPLAY_COL, row.get("League", "—"))
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

# ============================================================
# STREAMLIT APP
# ============================================================


st.title("Dual Role Pizza Radar")
st.caption(
    "Un solo grafico, stessa struttura del template: Player Style a sinistra, "
    "Performance a destra, metriche ruolo-specifiche. La scelta del template è indipendente dal giocatore."
)

BUCKET_TO_TEMPLATE = {
    "CB": "CB",
    "FB": "FB/WB",
    "MF": "MF",
    "AM": "AM",
    "W": "W/RML",
    "FW": "FW",
}

with st.sidebar:
    st.subheader("Selezione giocatore")

    player_database = st.radio(
        "Player database",
        ["Outfield players", "Goalkeepers"],
        index=0,
        help="Scegli da quale dataset pescare il giocatore. Il template di confronto si sceglie dopo ed è indipendente.",
    )

    df = load_gk() if player_database == "Goalkeepers" else load_outfield()

    seasons = sorted(df["Season"].dropna().astype(str).unique().tolist(), reverse=True)
    default_season = "2025-2026" if "2025-2026" in seasons else (seasons[0] if seasons else "")
    season = select_one(seasons, "Season", default=default_season)

    season_df = df[df["Season"].astype(str).eq(str(season))].copy()

    leagues = ["All"] + league_display_values(season_df)
    league_filter = st.selectbox("Filter teams by league", leagues)

    team_pool = season_df.copy()
    if league_filter != "All":
        team_pool = team_pool[team_pool[LEAGUE_DISPLAY_COL].astype(str).eq(league_filter)]

    teams = sorted(team_pool["Team"].dropna().astype(str).unique().tolist())
    team = select_one(teams, "Team")

    player_pool = team_pool[team_pool["Team"].astype(str).eq(str(team))].copy()
    players = sorted(player_pool["Player"].dropna().astype(str).unique().tolist())
    player = select_one(players, "Player")

    preview_rows = player_pool[player_pool["Player"].astype(str).eq(str(player))].copy()
    preview_row = preview_rows.sort_values("Minutes played", ascending=False).iloc[0] if not preview_rows.empty else None

    st.divider()
    st.subheader("Template e percentili")

    if player_database == "Goalkeepers":
        role_options = ["GK"]
        default_role = "GK"
    else:
        role_options = ["CB", "FB/WB", "MF", "AM", "W/RML", "FW"]
        actual_bucket = str(preview_row.get("Role bucket", "")) if preview_row is not None else ""
        default_role = BUCKET_TO_TEMPLATE.get(actual_bucket, "MF")

    role_index = role_options.index(default_role) if default_role in role_options else 0
    role = st.selectbox(
        "Compare/template as",
        role_options,
        index=role_index,
        help=(
            "Questo NON filtra il giocatore selezionato. Decide solo template grafico, metriche e cohort di percentili. "
            "Così puoi vedere un LCM nei percentili AM, un winger nei percentili FW, ecc."
        ),
    )
    st.caption(ROLE_HELP[role])

    if preview_row is not None and player_database != "Goalkeepers":
        actual_bucket = preview_row.get("Role bucket", "—")
        actual_pos = preview_row.get("Position", "—")
        st.caption(f"Actual player role: {actual_pos} · bucket {actual_bucket}. Compared as: {role}.")

    reference_scope = st.selectbox("Reference scope", ["Player league", "Big Five", "All leagues", "Custom leagues"], index=1)
    all_leagues = league_display_values(df)
    custom_leagues: list[str] = []
    if reference_scope == "Custom leagues":
        custom_leagues = st.multiselect(
            "Custom leagues",
            all_leagues,
            default=big_five_league_display_values(df),
        )
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
player_league = player_row.get(LEAGUE_DISPLAY_COL, player_row.get("League", None))
reference_df = build_reference(df, role, season, player_league, reference_scope, custom_leagues, min_minutes)

template = ROLE_TEMPLATES[role]
style_params, style_values, style_records = compute_metric_values(player_row, reference_df, template["style"], mode, section="style")
perf_params, perf_values, perf_records = compute_metric_values(player_row, reference_df, template["performance"], mode, section="performance")
style_colors = colors_for(template["style"], style_records)
perf_colors = colors_for(template["performance"], perf_records)

player_header(player_row, role, reference_scope, len(reference_df))
st.caption(f"Template: {template['display']} · {template['description']}")

if player_database != "Goalkeepers":
    actual_bucket = player_row.get("Role bucket", "—")
    actual_pos = player_row.get("Position", "—")
    if str(BUCKET_TO_TEMPLATE.get(str(actual_bucket), actual_bucket)) != str(role):
        st.info(
            f"Questo giocatore è classificato come {actual_pos} / bucket {actual_bucket}, "
            f"ma lo stai valutando con template e percentili {role}."
        )

if len(reference_df) < 15:
    st.warning(f"Reference group piccolo: n = {len(reference_df)}. I percentili potrebbero essere instabili.")

if len(style_params) < 6 or len(perf_params) < 6:
    st.error("Troppe poche metriche disponibili per disegnare il grafico. Controlla template e colonne del dataset.")
    st.stop()

fig = make_dual_pizza_figure(
    player_row=player_row,
    role=role,
    reference_df=reference_df,
    style_params=style_params,
    style_values=style_values,
    style_colors=style_colors,
    perf_params=perf_params,
    perf_values=perf_values,
    perf_colors=perf_colors,
    mode=mode,
    reference_scope=reference_scope,
)

st.pyplot(fig, use_container_width=True, clear_figure=False)

png = fig_to_png_bytes(fig)
st.download_button(
    "Download PNG",
    data=png,
    file_name=f"{str(player_row.get('Player','player')).replace(' ', '_')}_{role}_dual_pizza.png",
    mime="image/png",
)
plt.close(fig)

st.subheader("Metriche del grafico")
metric_rows = []
for section, records in [("Player Style", style_records), ("Performance", perf_records)]:
    for rec in records:
        if rec.get("Used", False):
            metric_rows.append({
                "Section": section,
                "Family": rec.get("Family"),
                "Metric label": rec.get("Metric label"),
                "Column": rec.get("Column"),
                "Raw value": rec.get("Raw value"),
                "Percentile": rec.get("Percentile"),
                "Reverse": rec.get("Reverse", False),
                "Requested adjustment": rec.get("Requested adjustment", "none"),
                "Applied adjustment": rec.get("Applied adjustment", "none"),
            })

metric_df = pd.DataFrame(metric_rows)
if not metric_df.empty:
    metric_df["Raw value"] = pd.to_numeric(metric_df["Raw value"], errors="coerce").round(2)
    metric_df["Percentile"] = pd.to_numeric(metric_df["Percentile"], errors="coerce").round(0).astype("Int64")
    st.dataframe(metric_df, hide_index=True, use_container_width=True)

with st.expander("Metriche escluse perché non disponibili / non calcolabili"):
    missing_rows = []
    for section, records in [("Player Style", style_records), ("Performance", perf_records)]:
        for rec in records:
            if not rec.get("Used", False):
                missing_rows.append({"Section": section, **rec})
    if missing_rows:
        st.dataframe(pd.DataFrame(missing_rows), hide_index=True, use_container_width=True)
    else:
        st.write("Nessuna metrica esclusa.")

with st.expander("Reference group"):
    cols = [
        c for c in [
            "Player", "Team", "League display", "League", "Nation", "Season", "Position", "Role bucket", "Minutes played", "style_cluster_short_label"
        ] if c in reference_df.columns
    ]
    st.dataframe(reference_df[cols].sort_values("Minutes played", ascending=False), hide_index=True, use_container_width=True)

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable
import colorsys
import hashlib
import re

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
PLAYERS_FILE = DATA_DIR / "players_enriched_with_clusters.csv.gz"
GK_FILE = DATA_DIR / "gk_enriched_with_clusters.csv.gz"
TEAM_FILE = DATA_DIR / "team_league_base.csv.gz"

BIG_FIVE_LEAGUES = {"Serie A", "Premier League", "La Liga", "Bundesliga", "Ligue 1"}
# True Big Five definition. This avoids including other leagues that share the
# same generic name, especially Premier League (Russia).
BIG_FIVE_COMPETITIONS = {
    ("Serie A", "Italy"),
    ("Premier League", "England"),
    ("La Liga", "Spain"),
    ("Bundesliga", "Germany"),
    ("Ligue 1", "France"),
}
LEAGUE_DISPLAY_COL = "League display"

FIG_BG = "#FFFFFF"
AXIS_COLOR = "#111111"
GRID_COLOR = "#D6D6D6"
TEXT_MUTED = "#4A4A4A"

# Club colours used when the exact team appears in the processed datasets.
# Unknown teams are handled by a deterministic fallback, so every club still gets
# a stable primary fill and secondary border.
TEAM_COLORS: dict[str, tuple[str, str]] = {
    # Serie A / Italy
    "Inter Milan": ("#0057B8", "#000000"),
    "AC Milan": ("#FB090B", "#000000"),
    "Juventus FC": ("#FFFFFF", "#000000"),
    "SSC Napoli": ("#12A0D7", "#003B79"),
    "AS Roma": ("#8E1F2F", "#F0BC42"),
    "SS Lazio": ("#87D8F7", "#FFFFFF"),
    "Atalanta BC": ("#1B75BC", "#000000"),
    "ACF Fiorentina": ("#4B2482", "#FFFFFF"),
    "Bologna FC 1909": ("#1D428A", "#C8102E"),
    "Torino FC": ("#8A1538", "#FFFFFF"),
    "Genoa CFC": ("#002D62", "#C8102E"),
    "Udinese Calcio": ("#FFFFFF", "#000000"),
    "US Lecce": ("#E31B23", "#FFD200"),
    "Cagliari Calcio": ("#00205B", "#D50032"),
    "Hellas Verona": ("#003DA5", "#FFD100"),
    "Como 1907": ("#0057B8", "#FFFFFF"),
    "Parma Calcio 1913": ("#FFCC00", "#0033A0"),
    "Sassuolo": ("#00A650", "#000000"),
    "Venezia FC": ("#000000", "#F58220"),
    "AC Monza": ("#E30613", "#FFFFFF"),
    "Cremonese": ("#A6192E", "#707372"),
    "Pisa Sporting Club": ("#0057B8", "#000000"),
    # Premier League / England
    "Arsenal FC": ("#EF0107", "#063672"),
    "Aston Villa": ("#95BFE5", "#670E36"),
    "AFC Bournemouth": ("#DA291C", "#000000"),
    "Brentford FC": ("#E30613", "#FFFFFF"),
    "Brighton & Hove Albion": ("#0057B8", "#FFFFFF"),
    "Burnley FC": ("#6C1D45", "#99D6EA"),
    "Chelsea FC": ("#034694", "#FFFFFF"),
    "Crystal Palace": ("#1B458F", "#C4122E"),
    "Everton FC": ("#003399", "#FFFFFF"),
    "Fulham": ("#FFFFFF", "#CC0000"),
    "Leeds United": ("#FFCD00", "#1D428A"),
    "Leicester City": ("#003090", "#FDBE11"),
    "Liverpool FC": ("#C8102E", "#00B2A9"),
    "Manchester City": ("#6CABDD", "#1C2C5B"),
    "Manchester United": ("#DA291C", "#FBE122"),
    "Newcastle United": ("#FFFFFF", "#000000"),
    "Nottingham Forest": ("#DD0000", "#FFFFFF"),
    "Southampton FC": ("#D71920", "#000000"),
    "Sunderland AFC": ("#EB172B", "#FFFFFF"),
    "Tottenham Hotspur": ("#FFFFFF", "#132257"),
    "West Ham United": ("#7A263A", "#1BB1E7"),
    "Wolverhampton Wanderers": ("#FDB913", "#231F20"),
    # La Liga / Spain
    "Real Madrid": ("#FFFFFF", "#FEBE10"),
    "FC Barcelona": ("#004D98", "#A50044"),
    "Atlético de Madrid": ("#CB3524", "#FFFFFF"),
    "Athletic Bilbao": ("#EE2523", "#FFFFFF"),
    "Real Sociedad": ("#0067B1", "#FFFFFF"),
    "Real Betis": ("#00954C", "#FFFFFF"),
    "Sevilla FC": ("#D71920", "#FFFFFF"),
    "Valencia CF": ("#F18E00", "#000000"),
    "Villarreal CF": ("#FFE667", "#005187"),
    "Getafe": ("#005BBB", "#FFFFFF"),
    "Girona FC": ("#E4002B", "#FFFFFF"),
    "Celta de Vigo": ("#8AD1F5", "#FFFFFF"),
    "CA Osasuna": ("#0A346F", "#D91A32"),
    "Rayo Vallecano": ("#FFFFFF", "#E53027"),
    "RCD Mallorca": ("#E30613", "#000000"),
    "RCD Espanyol Barcelona": ("#0070C0", "#FFFFFF"),
    # Bundesliga / Germany
    "FC Bayern Munchen": ("#DC052D", "#0066B2"),
    "Borussia Dortmund": ("#FDE100", "#000000"),
    "Bayer 04 Leverkusen": ("#E32221", "#000000"),
    "RB Leipzig": ("#FFFFFF", "#DD0741"),
    "Eintracht Frankfurt": ("#E1000F", "#000000"),
    "VfB Stuttgart": ("#E32219", "#FFFFFF"),
    "SC Freiburg": ("#E30613", "#000000"),
    "SV Werder Bremen": ("#1D9053", "#FFFFFF"),
    "Borussia M'gladbach": ("#FFFFFF", "#000000"),
    "VfL Wolfsburg": ("#65B32E", "#FFFFFF"),
    "TSG 1899 Hoffenheim": ("#005DAA", "#FFFFFF"),
    "1.FC Union Berlin": ("#E30613", "#FFD100"),
    "1.FSV Mainz 05": ("#C8102E", "#FFFFFF"),
    "FC St. Pauli": ("#5B3A29", "#FFFFFF"),
    "Hamburger SV": ("#0057B8", "#FFFFFF"),
    # Ligue 1 / France
    "Paris Saint-Germain": ("#004170", "#DA291C"),
    "Olympique Marseille": ("#2FAEE0", "#FFFFFF"),
    "Olympique Lyon": ("#004C99", "#E30613"),
    "AS Monaco": ("#E51B23", "#FFFFFF"),
    "LOSC Lille": ("#E01E3C", "#00205B"),
    "OGC Nice": ("#C8102E", "#000000"),
    "RC Lens": ("#D71920", "#FFCD00"),
    "Stade Rennais FC": ("#E30613", "#000000"),
    "FC Nantes": ("#FFF200", "#00843D"),
    "RC Strasbourg Alsace": ("#0055A4", "#FFFFFF"),
    "FC Toulouse": ("#5E2A84", "#FFFFFF"),
    # Saudi Pro League common names in this dataset
    "Al-Hilal SFC": ("#005BAC", "#FFFFFF"),
    "Al-Nassr FC": ("#FFDD00", "#0057B8"),
    "Al-Ittihad Club": ("#F7C600", "#000000"),
    "Al-Ahli SFC": ("#00843D", "#FFFFFF"),
    "Al-Shabab FC": ("#FFFFFF", "#000000"),
    "Al-Ettifaq FC": ("#C8102E", "#00843D"),
    "Al-Taawoun FC": ("#F8E71C", "#0047AB"),
    "Al-Fateh SC": ("#006837", "#0057B8"),
    "Al-Qadsiah FC": ("#C8102E", "#FFD100"),
    "NEOM SC": ("#111111", "#00C2A8"),
}

NON_METRIC_EXACT = {
    "№", "Index", "Age", "Height", "Weight", "Season_key", "Season_fallback_key",
    "League display",
    "style_cluster_id", "style_cluster_x", "style_cluster_y", "style_cluster_distance",
    "style_cluster_confidence", "style_cluster_min_minutes",
}
NON_METRIC_TOKENS = ["_merge", "team_context", "key"]


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


@st.cache_data(show_spinner=False)
def load_team_base() -> pd.DataFrame:
    df = pd.read_csv(TEAM_FILE, compression="gzip", low_memory=False)
    return standardize_base_columns(df)


def standardize_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Season", "Player", "Team", "League", "Nation", "Position", "Role bucket", "GK role", "style_cluster_short_label", "style_cluster_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", np.nan)
    if "Minutes played" in df.columns:
        df["Minutes played"] = clean_numeric(df["Minutes played"])
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


def big_five_league_display_values(df: pd.DataFrame) -> list[str]:
    if "League" not in df.columns:
        return []
    sub = df[big_five_mask(df)].copy()
    return league_display_values(sub)

def numeric_metric_columns(df: pd.DataFrame, min_non_null: int = 8) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if col in NON_METRIC_EXACT:
            continue
        lower = str(col).lower()
        if any(tok in lower for tok in NON_METRIC_TOKENS):
            continue
        values = clean_numeric(df[col])
        if values.notna().sum() >= min_non_null and values.nunique(dropna=True) > 1:
            out.append(str(col))
    preferred_order = [
        "Goals", "Assists", "Goals + Assists", "xG (expected goals)", "xA", "xG + xA",
        "Shots", "Shots on target, %", "Passes", "Passes accurate, %", "Key passes", "Key passes accurate, %",
        "Progressive passes", "Progressive passes accurate, %", "Carry", "Dribbles", "Dribbles successful, %",
        "Defensive challenges", "Defensive challenges won, %", "Air challenges", "Air challenges won, %",
        "Ball recoveries", "Interceptions", "Tackles", "Tackles successful, %", "Ball possession, %",
        "Goals prevented", "Shots saved, %", "Cross claim rate", "Passes accurate, %",
    ]
    ranked = [c for c in preferred_order if c in out]
    ranked.extend([c for c in out if c not in ranked])
    return ranked


def big_five_mask(df: pd.DataFrame) -> pd.Series:
    """True Big Five helper: league-name + nation, not league-name alone."""
    if "League" not in df.columns:
        return pd.Series(False, index=df.index)
    league = df["League"].astype(str)
    if "Nation" in df.columns:
        nation = df["Nation"].astype(str)
        mask = pd.Series(False, index=df.index)
        for league_name, country in BIG_FIVE_COMPETITIONS:
            mask = mask | (league.eq(league_name) & nation.eq(country))
        return mask
    # Fallback only for datasets without Nation: this is less precise but keeps older files usable.
    return league.isin(BIG_FIVE_LEAGUES)


def apply_base_filters(
    df: pd.DataFrame,
    season: str | None,
    league_mode: str,
    selected_leagues: list[str] | None = None,
    min_minutes: int = 0,
) -> pd.DataFrame:
    out = df.copy()
    if LEAGUE_DISPLAY_COL not in out.columns and "League" in out.columns:
        out[LEAGUE_DISPLAY_COL] = build_league_display(out)
    if season and "Season" in out.columns:
        out = out[out["Season"].astype(str).eq(str(season))]
    if min_minutes > 0 and "Minutes played" in out.columns:
        out = out[clean_numeric(out["Minutes played"]).fillna(0) >= min_minutes]
    if league_mode == "Big Five" and "League" in out.columns:
        out = out[big_five_mask(out)]
    elif league_mode == "Custom leagues" and selected_leagues:
        if LEAGUE_DISPLAY_COL in out.columns:
            out = out[out[LEAGUE_DISPLAY_COL].astype(str).isin(selected_leagues)]
        elif "League" in out.columns:
            out = out[out["League"].astype(str).isin(selected_leagues)]
    elif league_mode not in {"All leagues", "Big Five", "Custom leagues"}:
        if LEAGUE_DISPLAY_COL in out.columns:
            out = out[out[LEAGUE_DISPLAY_COL].astype(str).eq(str(league_mode))]
        elif "League" in out.columns:
            out = out[out["League"].astype(str).eq(str(league_mode))]
    return out.copy()

def _valid_hex(value: str) -> bool:
    return bool(re.fullmatch(r"#[0-9A-Fa-f]{6}", str(value).strip()))


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.strip().lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(np.clip(c, 0, 1) * 255):02X}" for c in rgb)


def luminance(hex_color: str) -> float:
    r, g, b = _hex_to_rgb(hex_color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def adjust_lightness(hex_color: str, factor: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = np.clip(l * factor, 0, 1)
    return _rgb_to_hex(colorsys.hls_to_rgb(h, l, s))


def fallback_team_colors(team: str) -> tuple[str, str]:
    digest = hashlib.md5(str(team).encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) / 0xFFFFFFFF
    sat = 0.63 + 0.20 * (int(digest[8:10], 16) / 255)
    light = 0.42 + 0.12 * (int(digest[10:12], 16) / 255)
    primary = _rgb_to_hex(colorsys.hls_to_rgb(hue, light, sat))
    secondary = adjust_lightness(primary, 0.48 if luminance(primary) > 0.45 else 1.55)
    return primary, secondary


def parse_color_overrides(text: str) -> dict[str, tuple[str, str]]:
    overrides: dict[str, tuple[str, str]] = {}
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        team = parts[0]
        primary = parts[1]
        secondary = parts[2] if len(parts) >= 3 else ""
        if _valid_hex(primary):
            if not _valid_hex(secondary):
                secondary = adjust_lightness(primary, 0.50 if luminance(primary) > 0.45 else 1.50)
            overrides[team] = (primary.upper(), secondary.upper())
    return overrides


def team_colors(team: str, overrides: dict[str, tuple[str, str]] | None = None) -> tuple[str, str]:
    team = str(team)
    if overrides and team in overrides:
        return overrides[team]
    if team in TEAM_COLORS:
        return TEAM_COLORS[team]
    return fallback_team_colors(team)




def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    value = str(hex_color).strip().lstrip('#')
    if len(value) != 6:
        return (0.0, 0.0, 0.0)
    try:
        r = int(value[0:2], 16) / 255.0
        g = int(value[2:4], 16) / 255.0
        b = int(value[4:6], 16) / 255.0
        return (r, g, b)
    except ValueError:
        return (0.0, 0.0, 0.0)


def perceived_luminance(hex_color: str) -> float:
    r, g, b = hex_to_rgb01(hex_color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_text_color(fill_hex: str) -> str:
    return "#111111" if perceived_luminance(fill_hex) >= 0.62 else "#FFFFFF"


_GENERIC_TEAM_TOKENS = {
    "ac", "acf", "afc", "as", "bc", "ca", "calcio", "cf", "club", "fc", "sc", "sfc",
    "sporting", "ss", "ssc", "us", "usd", "u.s.", "1907", "1908", "1909", "1912", "1913", "1919"
}


def team_abbreviation(team: str) -> str:
    name = str(team).strip()
    if not name:
        return "TEAM"
    cleaned = re.sub(r"[^A-Za-z0-9 -]", " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = [t for t in re.split(r"[\s-]+", cleaned) if t]
    if not tokens:
        return cleaned[:3].upper()

    lower_tokens = [t.lower() for t in tokens]

    # Common Saudi / Arabic form: Al + club name => use the next token.
    if lower_tokens[0] == "al" and len(tokens) >= 2:
        core = tokens[1]
        return core[:3].upper()

    significant = [t for t in tokens if t.lower() not in _GENERIC_TEAM_TOKENS and not t.isdigit()]
    if not significant:
        significant = tokens

    # If there is a clear first club word, prefer its first three letters.
    if len(significant) == 1:
        return significant[0][:3].upper()

    first = significant[0]
    if len(first) >= 3:
        return first[:3].upper()

    joined = "".join(t[0] for t in significant[:3]).upper()
    return (joined or cleaned[:3].upper())[:3]

def add_color_columns(df: pd.DataFrame, team_col: str = "Team", overrides: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    out = df.copy()
    colors = out[team_col].astype(str).apply(lambda t: team_colors(t, overrides))
    out["_fill_color"] = [c[0] for c in colors]
    out["_edge_color"] = [c[1] for c in colors]
    return out


def aggregate_teams(df: pd.DataFrame, x_metric: str, y_metric: str, extra_metrics: Iterable[str] = ()) -> pd.DataFrame:
    metrics = list(dict.fromkeys([x_metric, y_metric, *list(extra_metrics)]))
    work = df.copy()
    work["_minutes"] = clean_numeric(work.get("Minutes played", pd.Series(1, index=work.index))).fillna(0)
    group_cols = [c for c in ["Season", "League", "Team"] if c in work.columns]
    rows: list[dict[str, object]] = []
    for keys, g in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = dict(zip(group_cols, keys))
        minutes = clean_numeric(g.get("Minutes played", pd.Series(np.nan, index=g.index))).sum()
        row["Players"] = int(g["Player"].nunique()) if "Player" in g.columns else int(len(g))
        row["Total minutes"] = float(minutes) if pd.notna(minutes) else np.nan
        weights = clean_numeric(g.get("Minutes played", pd.Series(1, index=g.index))).fillna(0)
        if weights.sum() <= 0:
            weights = pd.Series(1.0, index=g.index)
        for metric in metrics:
            if metric not in g.columns:
                row[metric] = np.nan
                continue
            values = clean_numeric(g[metric])
            valid = values.notna()
            if valid.sum() == 0:
                row[metric] = np.nan
            else:
                w = weights[valid]
                if w.sum() <= 0:
                    row[metric] = float(values[valid].mean())
                else:
                    row[metric] = float(np.average(values[valid], weights=w))
        rows.append(row)
    return pd.DataFrame(rows)


def nice_metric_label(metric: str) -> str:
    return str(metric).replace("%", "%").replace("xG (expected goals)", "xG")


def fig_to_png_bytes(fig, dpi: int = 300) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "scatter"

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scatter_utils import (  # noqa: E402
    AXIS_COLOR,
    BIG_FIVE_LEAGUES,
    FIG_BG,
    GRID_COLOR,
    TEXT_MUTED,
    add_color_columns,
    aggregate_teams,
    apply_base_filters,
    clean_numeric,
    fig_to_png_bytes,
    load_outfield,
    nice_metric_label,
    numeric_metric_columns,
    parse_color_overrides,
    safe_filename,
)

st.set_page_config(page_title="Team Scatter Lab", page_icon="🟢", layout="wide")

st.title("Team Scatter Lab")
st.caption(
    "Scatter personalizzabili sulle squadre. I valori squadra sono aggregati dai giocatori outfield con media pesata per minuti; "
    "il riempimento del punto è il colore primario della squadra e il bordo è il colore secondario."
)

players = load_outfield()
metric_options = numeric_metric_columns(players)
if not metric_options:
    st.error("Non trovo colonne numeriche utilizzabili per costruire gli scatter.")
    st.stop()

with st.sidebar:
    st.subheader("Filtri")
    seasons = sorted(players["Season"].dropna().astype(str).unique().tolist(), reverse=True)
    default_season = "2025-2026" if "2025-2026" in seasons else (seasons[0] if seasons else None)
    season = st.selectbox("Season", seasons, index=seasons.index(default_season) if default_season in seasons else 0)

    leagues_available = sorted(players.loc[players["Season"].astype(str).eq(str(season)), "League"].dropna().astype(str).unique().tolist())
    league_choices = ["All leagues", "Big Five", "Custom leagues", *leagues_available]
    league_mode = st.selectbox("League scope", league_choices, index=1 if "Big Five" in league_choices else 0)
    selected_leagues: list[str] = []
    if league_mode == "Custom leagues":
        default_custom = [l for l in leagues_available if l in BIG_FIVE_LEAGUES]
        selected_leagues = st.multiselect("Custom leagues", leagues_available, default=default_custom)

    min_player_minutes = st.slider("Minimum player minutes", 0, 3000, 600, 100)
    min_team_players = st.slider("Minimum plotted players per team", 1, 30, 8, 1)
    min_team_minutes = st.slider("Minimum total team minutes", 0, 45000, 8000, 500)

    st.divider()
    st.subheader("Metriche")
    default_x = "xG (expected goals)" if "xG (expected goals)" in metric_options else metric_options[0]
    default_y = "Goals" if "Goals" in metric_options else metric_options[min(1, len(metric_options) - 1)]
    x_metric = st.selectbox("X-axis", metric_options, index=metric_options.index(default_x))
    y_metric = st.selectbox("Y-axis", metric_options, index=metric_options.index(default_y))

    st.divider()
    st.subheader("Aspetto")
    label_mode = st.radio("Team labels", ["Highlighted only", "All teams", "No labels"], index=0)
    size_by_minutes = st.checkbox("Size by total minutes", value=True)
    reference_lines = st.checkbox("Median reference lines", value=True)
    overrides_text = st.text_area(
        "Optional colour overrides",
        value="",
        help="Una riga per squadra: Team,#PRIMARY,#SECONDARY. Il bordo usa il secondo colore.",
        height=90,
    )

filtered_players = apply_base_filters(players, season, league_mode, selected_leagues, min_player_minutes)
if filtered_players.empty:
    st.warning("Nessun giocatore disponibile con questi filtri.")
    st.stop()

team_df = aggregate_teams(filtered_players, x_metric, y_metric)
team_df = team_df[
    (clean_numeric(team_df["Players"]).fillna(0) >= min_team_players)
    & (clean_numeric(team_df["Total minutes"]).fillna(0) >= min_team_minutes)
].copy()
team_df[x_metric] = clean_numeric(team_df[x_metric])
team_df[y_metric] = clean_numeric(team_df[y_metric])
team_df = team_df.dropna(subset=[x_metric, y_metric])

if team_df.empty:
    st.warning("Nessuna squadra plottabile dopo filtri e metriche selezionate.")
    st.stop()

all_teams = sorted(team_df["Team"].dropna().astype(str).unique().tolist())
with st.sidebar:
    highlighted_teams = st.multiselect("Highlight teams", all_teams, default=all_teams[:1] if all_teams else [])
    point_scale = st.slider("Point size", 60, 500, 220, 20)

color_overrides = parse_color_overrides(overrides_text)
plot_df = add_color_columns(team_df, overrides=color_overrides)
plot_df["_highlight"] = plot_df["Team"].astype(str).isin(highlighted_teams)

x = clean_numeric(plot_df[x_metric])
y = clean_numeric(plot_df[y_metric])

fig = plt.figure(figsize=(10, 10), dpi=220, facecolor=FIG_BG)
ax = fig.add_subplot(111)
ax.set_facecolor(FIG_BG)

if size_by_minutes:
    minutes = clean_numeric(plot_df["Total minutes"]).fillna(0)
    if minutes.max() > minutes.min():
        sizes = point_scale * (0.70 + 0.90 * (minutes - minutes.min()) / (minutes.max() - minutes.min()))
    else:
        sizes = pd.Series(point_scale, index=plot_df.index)
else:
    sizes = pd.Series(point_scale, index=plot_df.index)

# Draw non-highlighted first and highlighted second to keep focus clubs on top.
for is_highlight, alpha, lw, z in [(False, 0.72, 1.6, 3), (True, 0.98, 2.6, 5)]:
    sub = plot_df[plot_df["_highlight"].eq(is_highlight)]
    if sub.empty:
        continue
    ax.scatter(
        clean_numeric(sub[x_metric]),
        clean_numeric(sub[y_metric]),
        s=sizes.loc[sub.index],
        c=sub["_fill_color"].tolist(),
        edgecolors=sub["_edge_color"].tolist(),
        linewidths=lw,
        alpha=alpha,
        zorder=z,
    )

if reference_lines:
    ax.axvline(x.median(), color=GRID_COLOR, linestyle="--", linewidth=0.9, zorder=1)
    ax.axhline(y.median(), color=GRID_COLOR, linestyle="--", linewidth=0.9, zorder=1)

ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.35, color="lightgrey")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(AXIS_COLOR)
    ax.spines[spine].set_linewidth(1.2)
ax.tick_params(axis="both", which="both", length=0, labelsize=10, colors=AXIS_COLOR)
ax.set_xlabel(nice_metric_label(x_metric), fontsize=14, fontweight="bold", color=AXIS_COLOR)
ax.set_ylabel(nice_metric_label(y_metric), fontsize=14, fontweight="bold", color=AXIS_COLOR)

x_pad = (x.max() - x.min()) * 0.08 if x.max() > x.min() else 1
y_pad = (y.max() - y.min()) * 0.08 if y.max() > y.min() else 1
ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
ax.set_ylim(y.min() - y_pad, y.max() + y_pad)

label_df = plot_df.copy()
if label_mode == "Highlighted only":
    label_df = label_df[label_df["_highlight"]]
elif label_mode == "No labels":
    label_df = label_df.iloc[0:0]

for _, row in label_df.iterrows():
    ax.annotate(
        str(row["Team"]),
        (row[x_metric], row[y_metric]),
        xytext=(7, 5),
        textcoords="offset points",
        fontsize=8.5 if label_mode == "All teams" else 10.5,
        fontweight="bold" if row.get("_highlight", False) else "normal",
        color=AXIS_COLOR,
        zorder=7,
    )

scope_txt = league_mode if league_mode != "Custom leagues" else ", ".join(selected_leagues) or "Custom leagues"
fig.text(0.08, 0.965, "Team Scatter Lab", ha="left", va="top", fontsize=22, fontweight="bold", color=AXIS_COLOR)
fig.text(
    0.08,
    0.932,
    f"{season} | {scope_txt} | {len(plot_df)} teams | weighted by player minutes",
    ha="left",
    va="top",
    fontsize=10.5,
    color=TEXT_MUTED,
)
fig.text(
    0.02,
    0.018,
    "Point fill = team primary colour; point border = team secondary colour. Team values are minutes-weighted means from outfield players.",
    ha="left",
    va="bottom",
    fontsize=7,
    color=TEXT_MUTED,
)
plt.tight_layout(rect=[0.06, 0.06, 0.98, 0.89])

st.pyplot(fig, use_container_width=True, clear_figure=False)

st.download_button(
    "Download PNG",
    data=fig_to_png_bytes(fig),
    file_name=f"team_scatter_{safe_filename(x_metric)}_vs_{safe_filename(y_metric)}.png",
    mime="image/png",
)
plt.close(fig)

st.subheader("Dati plottati")
show_cols = ["Season", "League", "Team", "Players", "Total minutes", x_metric, y_metric, "_fill_color", "_edge_color"]
show_cols = [c for c in show_cols if c in plot_df.columns]
display_df = plot_df[show_cols].sort_values(y_metric, ascending=False).copy()
for col in ["Total minutes", x_metric, y_metric]:
    if col in display_df.columns:
        display_df[col] = clean_numeric(display_df[col]).round(2)
st.dataframe(display_df, hide_index=True, use_container_width=True)

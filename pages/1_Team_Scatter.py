from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scatter_utils import (  # noqa: E402
    AXIS_COLOR,
    FIG_BG,
    GRID_COLOR,
    TEXT_MUTED,
    add_color_columns,
    big_five_league_display_values,
    big_five_mask,
    clean_numeric,
    contrast_text_color,
    fig_to_png_bytes,
    league_display_values,
    load_team_base,
    nice_metric_label,
    numeric_metric_columns,
    parse_color_overrides,
    safe_filename,
    team_abbreviation,
)

st.set_page_config(page_title="Team Scatter Lab", page_icon="🟢", layout="wide")

st.title("Team Scatter Lab")
st.caption(
    "Scatter personalizzabili sulle squadre usando direttamente il dataset team-level `team_league_base.csv.gz`. "
    "Ogni punto è una squadra: dimensione uniforme, riempimento nel colore primario del club, bordo nel colore secondario e sigla a tre lettere al centro."
)

teams_base = load_team_base()
metric_options = numeric_metric_columns(teams_base)
if not metric_options:
    st.error("Non trovo colonne numeriche utilizzabili per costruire gli scatter.")
    st.stop()

with st.sidebar:
    st.subheader("Filtri")
    seasons = sorted(teams_base["Season"].dropna().astype(str).unique().tolist(), reverse=True)
    default_season = "2025-2026" if "2025-2026" in seasons else (seasons[0] if seasons else None)
    season = st.selectbox("Season", seasons, index=seasons.index(default_season) if default_season in seasons else 0)

    season_df = teams_base[teams_base["Season"].astype(str).eq(str(season))].copy()
    leagues_available = league_display_values(season_df)
    league_choices = ["All leagues", "Big Five", "Custom leagues", *leagues_available]
    league_mode = st.selectbox("League scope", league_choices, index=1 if "Big Five" in league_choices else 0)

    selected_leagues: list[str] = []
    if league_mode == "Custom leagues":
        default_custom = big_five_league_display_values(season_df)
        selected_leagues = st.multiselect("Custom leagues", leagues_available, default=default_custom)

    filtered = season_df.copy()
    if league_mode == "Big Five":
        filtered = filtered[big_five_mask(filtered)].copy()
    elif league_mode == "Custom leagues":
        if selected_leagues:
            filtered = filtered[filtered["League display"].astype(str).isin(selected_leagues)].copy()
        else:
            filtered = filtered.iloc[0:0].copy()
    elif league_mode != "All leagues":
        filtered = filtered[filtered["League display"].astype(str).eq(str(league_mode))].copy()

    if "Matches estimated" in filtered.columns:
        min_matches = st.slider("Minimum estimated matches", 0, 40, 0, 1)
        filtered = filtered[clean_numeric(filtered["Matches estimated"]).fillna(0) >= min_matches].copy()

    if "Player minutes total" in filtered.columns:
        min_player_coverage = st.slider("Minimum player-data coverage minutes", 0, 45000, 0, 500)
        if min_player_coverage > 0:
            filtered = filtered[clean_numeric(filtered["Player minutes total"]).fillna(0) >= min_player_coverage].copy()

    st.divider()
    st.subheader("Metriche")
    default_x_candidates = ["xG/team derived", "xG total derived", "xGA per match weighted", "xGD total derived", "Goals - xG total derived"]
    default_y_candidates = ["Goals", "Goals for total", "Goals total derived", "Goals/team derived"]
    default_x = next((m for m in default_x_candidates if m in metric_options), metric_options[0])
    default_y = next((m for m in default_y_candidates if m in metric_options), metric_options[min(1, len(metric_options) - 1)])
    x_metric = st.selectbox("X-axis", metric_options, index=metric_options.index(default_x))
    y_metric = st.selectbox("Y-axis", metric_options, index=metric_options.index(default_y))

    st.divider()
    st.subheader("Aspetto")
    label_mode = st.radio("Outside labels", ["Highlighted only", "All teams", "No labels"], index=0)
    reference_lines = st.checkbox("Median reference lines", value=True)
    show_initials = st.checkbox("Show 3-letter team codes inside points", value=True)
    point_scale = st.slider("Point size", 320, 900, 600, 10)
    overrides_text = st.text_area(
        "Optional colour overrides",
        value="",
        help="Una riga per squadra: Team,#PRIMARY,#SECONDARY. Il bordo usa il secondo colore.",
        height=90,
    )

if filtered.empty:
    st.warning("Nessuna squadra disponibile con questi filtri.")
    st.stop()

filtered[x_metric] = clean_numeric(filtered[x_metric])
filtered[y_metric] = clean_numeric(filtered[y_metric])
plot_df = filtered.dropna(subset=[x_metric, y_metric]).copy()

if plot_df.empty:
    st.warning("Nessuna squadra plottabile dopo filtri e metriche selezionate.")
    st.stop()

all_teams = sorted(plot_df["Team"].dropna().astype(str).unique().tolist())
with st.sidebar:
    highlighted_teams = st.multiselect("Highlight teams", all_teams, default=all_teams[:1] if all_teams else [])

color_overrides = parse_color_overrides(overrides_text)
plot_df = add_color_columns(plot_df, overrides=color_overrides)
plot_df["_highlight"] = plot_df["Team"].astype(str).isin(highlighted_teams)
plot_df["_abbr"] = plot_df["Team"].astype(str).apply(team_abbreviation)
plot_df["_text_color"] = plot_df["_fill_color"].astype(str).apply(contrast_text_color)

x = clean_numeric(plot_df[x_metric])
y = clean_numeric(plot_df[y_metric])

fig = plt.figure(figsize=(10.5, 10), dpi=220, facecolor=FIG_BG)
ax = fig.add_subplot(111)
ax.set_facecolor(FIG_BG)

sizes = pd.Series(float(point_scale), index=plot_df.index)

# Draw non-highlighted first and highlighted second to keep focus clubs on top.
for is_highlight, alpha, lw, z in [(False, 0.90, 1.8, 3), (True, 1.00, 2.8, 5)]:
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

if show_initials:
    for _, row in plot_df.iterrows():
        ax.text(
            row[x_metric],
            row[y_metric],
            str(row["_abbr"]),
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color=row["_text_color"],
            zorder=6,
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
        xytext=(10, 7),
        textcoords="offset points",
        fontsize=8.5 if label_mode == "All teams" else 10.5,
        fontweight="bold" if row.get("_highlight", False) else "normal",
        color=AXIS_COLOR,
        zorder=7,
    )

if league_mode == "Big Five":
    scope_txt = "Big Five (England, France, Germany, Italy, Spain)"
elif league_mode == "Custom leagues":
    scope_txt = ", ".join(selected_leagues) or "Custom leagues"
else:
    scope_txt = league_mode
fig.text(0.08, 0.965, "Team Scatter Lab", ha="left", va="top", fontsize=22, fontweight="bold", color=AXIS_COLOR)
fig.text(
    0.08,
    0.932,
    f"{season} | {scope_txt} | {len(plot_df)} teams | team-level source",
    ha="left",
    va="top",
    fontsize=10.5,
    color=TEXT_MUTED,
)
fig.text(
    0.02,
    0.018,
    "Source = team_league_base.csv.gz. Point fill = team primary colour; point border = team secondary colour; each point displays a 3-letter team code.",
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
show_cols = [
    "Season", "League display", "League", "Nation", "Competition", "Team", "Matches estimated", "Players in player dataset", "Player minutes total",
    x_metric, y_metric, "_abbr", "_fill_color", "_edge_color",
]
show_cols = [c for c in show_cols if c in plot_df.columns]
display_df = plot_df[show_cols].sort_values(y_metric, ascending=False).copy()
for col in ["Matches estimated", "Players in player dataset", "Player minutes total", x_metric, y_metric]:
    if col in display_df.columns:
        display_df[col] = clean_numeric(display_df[col]).round(2)
st.dataframe(display_df, hide_index=True, use_container_width=True)

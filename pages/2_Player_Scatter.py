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
    FIG_BG,
    GRID_COLOR,
    TEXT_MUTED,
    add_color_columns,
    apply_base_filters,
    big_five_league_display_values,
    clean_numeric,
    fig_to_png_bytes,
    league_display_values,
    load_gk,
    load_outfield,
    nice_metric_label,
    numeric_metric_columns,
    parse_color_overrides,
    safe_filename,
)

st.set_page_config(page_title="Player Scatter Lab", page_icon="🔵", layout="wide")

st.title("Player Scatter Lab")
st.caption(
    "Scatter personalizzabili sui giocatori. Ogni punto usa i colori della squadra: riempimento = colore primario, bordo = colore secondario, senza testo dentro i punti."
)

with st.sidebar:
    st.subheader("Dataset")
    database = st.radio("Player database", ["Outfield players", "Goalkeepers"], index=0)

df = load_gk() if database == "Goalkeepers" else load_outfield()
metric_options = numeric_metric_columns(df)
if not metric_options:
    st.error("Non trovo colonne numeriche utilizzabili per costruire gli scatter.")
    st.stop()

role_col = "GK role" if database == "Goalkeepers" else "Role bucket"
position_col = "GK role" if database == "Goalkeepers" else "Position"

with st.sidebar:
    st.subheader("Filtri")
    seasons = sorted(df["Season"].dropna().astype(str).unique().tolist(), reverse=True)
    default_season = "2025-2026" if "2025-2026" in seasons else (seasons[0] if seasons else None)
    season = st.selectbox("Season", seasons, index=seasons.index(default_season) if default_season in seasons else 0)

    season_df = df[df["Season"].astype(str).eq(str(season))].copy()
    leagues_available = league_display_values(season_df)
    league_choices = ["All leagues", "Big Five", "Custom leagues", *leagues_available]
    league_mode = st.selectbox("League scope", league_choices, index=1 if "Big Five" in league_choices else 0)
    selected_leagues: list[str] = []
    if league_mode == "Custom leagues":
        default_custom = big_five_league_display_values(season_df)
        selected_leagues = st.multiselect("Custom leagues", leagues_available, default=default_custom)

    min_minutes = st.slider("Minimum minutes", 0, 3000, 600, 100)

filtered = apply_base_filters(df, season, league_mode, selected_leagues, min_minutes)
if filtered.empty:
    st.warning("Nessun giocatore disponibile con questi filtri.")
    st.stop()

with st.sidebar:
    if role_col in filtered.columns:
        roles = sorted(filtered[role_col].dropna().astype(str).unique().tolist())
        selected_roles = st.multiselect("Role buckets", roles, default=roles)
        if selected_roles:
            filtered = filtered[filtered[role_col].astype(str).isin(selected_roles)].copy()

    teams = sorted(filtered["Team"].dropna().astype(str).unique().tolist())
    selected_teams = st.multiselect("Teams", teams, default=[])
    if selected_teams:
        filtered = filtered[filtered["Team"].astype(str).isin(selected_teams)].copy()

    clusters_available = []
    if "style_cluster_short_label" in filtered.columns:
        clusters_available = sorted(filtered["style_cluster_short_label"].dropna().astype(str).unique().tolist())
    selected_clusters: list[str] = []
    if clusters_available:
        selected_clusters = st.multiselect("Style clusters", clusters_available, default=[])
        if selected_clusters:
            filtered = filtered[filtered["style_cluster_short_label"].astype(str).isin(selected_clusters)].copy()

    st.divider()
    st.subheader("Metriche")
    if database == "Goalkeepers":
        default_x = "Shots saved, %" if "Shots saved, %" in metric_options else metric_options[0]
        default_y = "Goals prevented" if "Goals prevented" in metric_options else metric_options[min(1, len(metric_options) - 1)]
    else:
        default_x = "xG (expected goals)" if "xG (expected goals)" in metric_options else metric_options[0]
        default_y = "Goals" if "Goals" in metric_options else metric_options[min(1, len(metric_options) - 1)]
    x_metric = st.selectbox("X-axis", metric_options, index=metric_options.index(default_x))
    y_metric = st.selectbox("Y-axis", metric_options, index=metric_options.index(default_y))

    st.divider()
    st.subheader("Aspetto")
    max_points = st.slider("Max plotted players", 200, 20000, 3500, 100)
    size_mode = st.selectbox("Point size", ["Fixed", "Minutes"], index=1)
    point_scale = st.slider("Base point size", 20, 260, 90, 10)
    label_mode = st.radio("Player labels", ["Highlighted only", "Top-right 15", "No labels"], index=0)
    reference_lines = st.checkbox("Median reference lines", value=True)
    overrides_text = st.text_area(
        "Optional colour overrides",
        value="",
        help="Una riga per squadra: Team,#PRIMARY,#SECONDARY. Il bordo usa il secondo colore.",
        height=90,
    )

filtered[x_metric] = clean_numeric(filtered[x_metric])
filtered[y_metric] = clean_numeric(filtered[y_metric])
plot_df = filtered.dropna(subset=[x_metric, y_metric]).copy()

if plot_df.empty:
    st.warning("Nessun giocatore plottabile dopo filtri e metriche selezionate.")
    st.stop()

# Avoid extremely heavy figures: keep the highest-minute observations unless the user raises the cap.
if len(plot_df) > max_points:
    plot_df = plot_df.sort_values("Minutes played", ascending=False).head(max_points).copy()
    st.info(f"Per mantenere il grafico fluido ho plottato i {max_points} giocatori con più minuti tra quelli filtrati.")

player_label = plot_df["Player"].astype(str) + " — " + plot_df["Team"].astype(str)
with st.sidebar:
    highlighted_options = sorted(player_label.unique().tolist())
    highlighted_players = st.multiselect("Highlight players", highlighted_options, default=[])

color_overrides = parse_color_overrides(overrides_text)
plot_df = add_color_columns(plot_df, overrides=color_overrides)
plot_df["_player_label"] = plot_df["Player"].astype(str) + " — " + plot_df["Team"].astype(str)
plot_df["_highlight"] = plot_df["_player_label"].isin(highlighted_players)

x = clean_numeric(plot_df[x_metric])
y = clean_numeric(plot_df[y_metric])

fig = plt.figure(figsize=(10.5, 10), dpi=220, facecolor=FIG_BG)
ax = fig.add_subplot(111)
ax.set_facecolor(FIG_BG)

if size_mode == "Minutes" and "Minutes played" in plot_df.columns:
    minutes = clean_numeric(plot_df["Minutes played"]).fillna(0)
    if minutes.max() > minutes.min():
        sizes = point_scale * (0.55 + 1.35 * (minutes - minutes.min()) / (minutes.max() - minutes.min()))
    else:
        sizes = pd.Series(point_scale, index=plot_df.index)
else:
    sizes = pd.Series(point_scale, index=plot_df.index)

for is_highlight, alpha, lw, z in [(False, 0.58, 1.0, 3), (True, 0.98, 2.7, 6)]:
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

if label_mode == "Highlighted only":
    label_df = plot_df[plot_df["_highlight"]].copy()
elif label_mode == "Top-right 15":
    # Synthetic top-right score based on percentile within the plotted sample.
    x_pct = x.rank(pct=True)
    y_pct = y.rank(pct=True)
    tmp = plot_df.copy()
    tmp["_label_score"] = x_pct + y_pct
    label_df = tmp.sort_values("_label_score", ascending=False).head(15)
else:
    label_df = plot_df.iloc[0:0].copy()

for _, row in label_df.iterrows():
    label = str(row["Player"])
    if "Team" in row:
        label = f"{label}\n{row['Team']}"
    ax.annotate(
        label,
        (row[x_metric], row[y_metric]),
        xytext=(7, 5),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold" if row.get("_highlight", False) else "normal",
        color=AXIS_COLOR,
        zorder=8,
    )

if league_mode == "Big Five":
    scope_txt = "Big Five (England, France, Germany, Italy, Spain)"
elif league_mode == "Custom leagues":
    scope_txt = ", ".join(selected_leagues) or "Custom leagues"
else:
    scope_txt = league_mode
fig.text(0.08, 0.965, "Player Scatter Lab", ha="left", va="top", fontsize=22, fontweight="bold", color=AXIS_COLOR)
fig.text(
    0.08,
    0.932,
    f"{database} | {season} | {scope_txt} | {len(plot_df)} players | min {min_minutes} minutes",
    ha="left",
    va="top",
    fontsize=10.5,
    color=TEXT_MUTED,
)
fig.text(
    0.02,
    0.018,
    "Point fill = team primary colour; point border = team secondary colour. Use Highlight players for labelled focus points.",
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
    file_name=f"player_scatter_{safe_filename(x_metric)}_vs_{safe_filename(y_metric)}.png",
    mime="image/png",
)
plt.close(fig)

st.subheader("Dati plottati")
show_cols = [
    "Season", "League display", "League", "Nation", "Team", "Player", "Age", position_col, role_col, "Minutes played",
    "style_cluster_short_label", x_metric, y_metric, "_fill_color", "_edge_color",
]
show_cols = [c for c in show_cols if c in plot_df.columns]
display_df = plot_df[show_cols].sort_values(y_metric, ascending=False).copy()
for col in ["Minutes played", x_metric, y_metric]:
    if col in display_df.columns:
        display_df[col] = clean_numeric(display_df[col]).round(2)
st.dataframe(display_df, hide_index=True, use_container_width=True)

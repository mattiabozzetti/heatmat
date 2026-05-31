# Dual Role Pizza App

App Streamlit minimale per creare un unico grafico **dual pizza** con la stessa struttura del template caricato:

- Player Style a sinistra
- Performance a destra
- tante metriche ruolo-specifiche
- colori a blocchi per famiglia
- background grafico bianco

## Novità v3

La scelta del **template/percentile role** è indipendente dal giocatore selezionato.

Questo significa che puoi selezionare qualsiasi giocatore outfield e poi valutarlo, per esempio, come:

- CB
- FB/WB
- MF
- AM
- W/RML
- FW

Il template scelto controlla solo:

1. metriche visualizzate nel grafico;
2. cohort usata per i percentili;
3. etichetta `Compared as`.

Non filtra più la lista giocatori.

Per i portieri si usa il database GK e il template GK dedicato.

## Avvio locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struttura

```text
app.py
requirements.txt
runtime.txt
.python-version
.streamlit/config.toml
data/processed/players_enriched_with_clusters.csv.gz
data/processed/gk_enriched_with_clusters.csv.gz
```

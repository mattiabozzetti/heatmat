# Dual Role Pizza Radar App

App Streamlit minimale per generare un unico grafico dual pizza/radar con:

- sinistra: **Player Style**
- destra: **Performance**
- tante metriche diverse per ruolo
- colori a blocchi per famiglia metrica
- resa grafica coerente con il template `radar_attaccanti_big5_perf_stile.py`
- sfondo del grafico bianco

## Avvio

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dati inclusi

La cartella contiene già:

```text
data/processed/players_enriched_with_clusters.csv.gz
data/processed/gk_enriched_with_clusters.csv.gz
```

## Ruoli disponibili

- CB
- FB/WB
- MF
- AM
- W/RML
- FW
- GK

Ogni template mantiene la stessa struttura visuale ma cambia le metriche usate per descrivere meglio il ruolo.

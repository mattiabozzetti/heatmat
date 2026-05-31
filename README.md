# Dual Role Pizza App

App Streamlit minimale per generare il grafico dual pizza `Player Style` + `Performance` mantenendo la resa del template originale caricato.

## Struttura richiesta su Streamlit Cloud

Quando carichi su GitHub, questi file devono stare nella root del repository, cioè allo stesso livello:

```text
app.py
requirements.txt
runtime.txt
.streamlit/config.toml
data/processed/players_enriched_with_clusters.csv.gz
data/processed/gk_enriched_with_clusters.csv.gz
```

Se `requirements.txt` non è nella stessa cartella di `app.py`, Streamlit Cloud non installa `matplotlib` e l'app dà errore su:

```python
import matplotlib.pyplot as plt
```

## Avvio locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dipendenze principali

- streamlit
- pandas
- numpy
- matplotlib
- mplsoccer

## Note

Il grafico mantiene la struttura del template originale:

- singola figura dual pizza
- `Player Style` a sinistra
- `Performance` a destra
- sfondo del grafico bianco
- metriche multiple per ruolo
- colori a blocchi per famiglia tattica

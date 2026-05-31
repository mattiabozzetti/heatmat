# Dual Role Radar App

App Streamlit minimale per generare due radar affiancati per giocatore:

- **Player Style**: quanto il giocatore fa certe cose, quindi volume / coinvolgimento / identità.
- **Performance**: quanto bene le fa, quindi qualità / successo / efficienza.

Ogni template ruolo usa sempre 4 famiglie nello stesso ordine:

1. Offensive
2. Defensive
3. Possession
4. Passing

I grafici hanno sfondo bianco e UI Streamlit base.

## Avvio locale

```bash
cd dual_role_radar_app
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Dati attesi

La cartella contiene già:

```text
data/processed/players_enriched_with_clusters.csv.gz
data/processed/gk_enriched_with_clusters.csv.gz
```

L'app usa questi file per giocatori di movimento e portieri.

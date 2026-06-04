HOTFIX IMPORT STREAMLIT

Copia questi file dentro la repo locale /Users/mattiabozzetti/Documents/GitHub/heatmat mantenendo le stesse cartelle:

scatter_utils.py
pages/1_Team_Scatter.py
pages/2_Player_Scatter.py
data/processed/team_league_base.csv.gz

Poi esegui:
cd /Users/mattiabozzetti/Documents/GitHub/heatmat
git status
git add scatter_utils.py pages/1_Team_Scatter.py pages/2_Player_Scatter.py data/processed/team_league_base.csv.gz
git commit -m "Fix scatter utilities and team base"
git push origin main

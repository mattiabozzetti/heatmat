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
TEAM_BASE_FILE = DATA_DIR / "team_league_base.csv.gz"

BIG_FIVE_LEAGUES = {"Serie A", "Premier League", "La Liga", "Bundesliga", "Ligue 1"}
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
    "1.FC Heidenheim 1846": ("#3560C0", "#FFFFFF"),  # auto
    "1.FC Kaiserslautern": ("#CA693F", "#FFFFFF"),  # auto
    "1.FC Köln": ("#4520CB", "#FFFFFF"),  # auto
    "1.FC Magdeburg": ("#C03648", "#FFFFFF"),  # auto
    "1.FC Nuremberg": ("#E22482", "#FFFFFF"),  # auto
    "1.FC Slovacko": ("#23CEAE", "#000000"),  # auto
    "1.FC Union Berlin": ("#E30613", "#FFD100"),
    "1.FSV Mainz 05": ("#C8102E", "#FFFFFF"),
    "AA Argentinos Juniors": ("#3475C1", "#FFFFFF"),  # auto
    "AC Milan": ("#FB090B", "#000000"),
    "AC Monza": ("#E30613", "#FFFFFF"),
    "AC Oulu": ("#26D397", "#000000"),  # auto
    "AC Reggiana 1919": ("#800A2A", "#EFE70B"),  # auto
    "AC Sparta Prague": ("#D45538", "#FFFFFF"),  # auto
    "ACF Fiorentina": ("#4B2482", "#FFFFFF"),
    "ACSC FC Arges": ("#232AD9", "#FFFFFF"),  # auto
    "AD Ceuta FC": ("#98BA34", "#FFFFFF"),  # auto
    "ADO Den Haag": ("#C07E2A", "#FFFFFF"),  # auto
    "AEK Athens": ("#9F30CE", "#FFFFFF"),  # auto
    "AFC Bournemouth": ("#DA291C", "#000000"),
    "AFC Unirea 04 Slobozia": ("#5337D4", "#FFFFFF"),  # auto
    "AGF Aarhus": ("#C53969", "#FFFFFF"),  # auto
    "AIK": ("#D32547", "#FFFFFF"),  # auto
    "AJ Auxerre": ("#E033A2", "#FFFFFF"),  # auto
    "APO Levadiakos": ("#22C676", "#FFFFFF"),  # auto
    "AS Monaco": ("#E51B23", "#FFFFFF"),
    "AS Nancy-Lorraine": ("#D17225", "#FFFFFF"),  # auto
    "AS Roma": ("#8E1F2F", "#F0BC42"),
    "AS Saint-Étienne": ("#2A74D2", "#FFFFFF"),  # auto
    "AZ Alkmaar": ("#D71920", "#FFFFFF"),
    "Aberdeen FC": ("#A6DF38", "#000000"),  # auto
    "Abha Club": ("#99C924", "#000000"),  # auto
    "Adana Demirspor": ("#2FC0A9", "#FFFFFF"),  # auto
    "Adelaide United": ("#D13B73", "#FFFFFF"),  # auto
    "Aguilas Doradas": ("#6AE125", "#000000"),  # auto
    "Ajax Amsterdam": ("#D2122E", "#FFFFFF"),
    "Ajman": ("#288BCF", "#FFFFFF"),  # auto
    "Akhmat Grozny": ("#3E2ABC", "#FFFFFF"),  # auto
    "Akron": ("#2ECD6A", "#000000"),  # auto
    "Al Ahly FC": ("#E30613", "#FFFFFF"),
    "Al Anwar": ("#C97740", "#FFFFFF"),  # auto
    "Al Bataeh": ("#70CA3D", "#000000"),  # auto
    "Al Diraiyah FC": ("#E01F8F", "#FFFFFF"),  # auto
    "Al Jazira": ("#3CD4BC", "#000000"),  # auto
    "Al Nasr": ("#35E17B", "#000000"),  # auto
    "Al Qaisumah": ("#D04832", "#FFFFFF"),  # auto
    "Al Taraji": ("#37B6D9", "#FFFFFF"),  # auto
    "Al Ula": ("#D42886", "#FFFFFF"),  # auto
    "Al Wahda": ("#BF562E", "#FFFFFF"),  # auto
    "Al Wakrah": ("#1EAEB8", "#FFFFFF"),  # auto
    "Al-Adalah FC": ("#1DBD3D", "#FFFFFF"),  # auto
    "Al-Ahli SC": ("#29E445", "#000000"),  # auto
    "Al-Ahli SFC": ("#00843D", "#FFFFFF"),
    "Al-Ain FC": ("#A826B7", "#FFFFFF"),  # auto
    "Al-Ain SFC": ("#39E273", "#000000"),  # auto
    "Al-Arabi SC": ("#B8E43A", "#000000"),  # auto
    "Al-Batin FC": ("#6ABD2E", "#FFFFFF"),  # auto
    "Al-Bukiryah FC": ("#3F50C7", "#FFFFFF"),  # auto
    "Al-Dhafra FC": ("#B4DB26", "#000000"),  # auto
    "Al-Duhail SC": ("#C91DB9", "#FFFFFF"),  # auto
    "Al-Ettifaq FC": ("#C8102E", "#00843D"),
    "Al-Faisaly FC": ("#7DDE39", "#000000"),  # auto
    "Al-Fateh SC": ("#00AEFF", "#00A34F"),
    "Al-Fayha FC": ("#62C125", "#FFFFFF"),  # auto
    "Al-Gharafa SC": ("#C4DE41", "#000000"),  # auto
    "Al-Hazem SC": ("#C78C2B", "#FFFFFF"),  # auto
    "Al-Hilal SFC": ("#005BAC", "#FFFFFF"),
    "Al-Ittihad Club": ("#F7C600", "#000000"),
    "Al-Ittihad Kalba SC": ("#CAC029", "#000000"),  # auto
    "Al-Jabalain FC": ("#E5DE23", "#000000"),  # auto
    "Al-Jandal SC": ("#3664E0", "#FFFFFF"),  # auto
    "Al-Jubail Club": ("#2DE496", "#000000"),  # auto
    "Al-Khaleej FC": ("#8D3ADA", "#FFFFFF"),  # auto
    "Al-Kharaitiyat SC": ("#DC5033", "#FFFFFF"),  # auto
    "Al-Kholood Club": ("#742BBA", "#FFFFFF"),  # auto
    "Al-Khor SC": ("#BDCE2E", "#000000"),  # auto
    "Al-Najma SC": ("#E4332A", "#FFFFFF"),  # auto
    "Al-Nassr FC": ("#FFDD00", "#0057B8"),
    "Al-Okhdood Club": ("#D9C241", "#000000"),  # auto
    "Al-Orobah FC": ("#BE473A", "#FFFFFF"),  # auto
    "Al-Orooba FC": ("#ABBD39", "#000000"),  # auto
    "Al-Qadsiah FC": ("#C8102E", "#FFD100"),
    "Al-Raed SFC": ("#60BF3A", "#FFFFFF"),  # auto
    "Al-Rayyan SC": ("#1ED155", "#000000"),  # auto
    "Al-Riyadh SC": ("#3893BE", "#FFFFFF"),  # auto
    "Al-Sadd SC": ("#C9AF33", "#FFFFFF"),  # auto
    "Al-Safa FC": ("#C33CA7", "#FFFFFF"),  # auto
    "Al-Sailiya SC": ("#79BD1A", "#FFFFFF"),  # auto
    "Al-Shabab FC": ("#000000", "#FFFFFF"),
    "Al-Shahania SC": ("#3740BE", "#FFFFFF"),  # auto
    "Al-Shamal SC": ("#2A37C2", "#FFFFFF"),  # auto
    "Al-Taawoun FC": ("#F8E71C", "#0047AB"),
    "Al-Tai FC": ("#E5444F", "#FFFFFF"),  # auto
    "Al-Wasl FC": ("#29E2D5", "#000000"),  # auto
    "Al-Wehda FC": ("#BCB135", "#FFFFFF"),  # auto
    "Al-Zulfi SFC": ("#9FDD36", "#000000"),  # auto
    "Alanyaspor": ("#C3349F", "#FFFFFF"),  # auto
    "Albacete": ("#C27E1C", "#FFFFFF"),  # auto
    "Albirex Niigata": ("#6424CF", "#FFFFFF"),  # auto
    "Aldosivi": ("#C5431C", "#FFFFFF"),  # auto
    "Alianza Petrolera": ("#32C427", "#FFFFFF"),  # auto
    "Almere City": ("#C5D020", "#000000"),  # auto
    "Almeria": ("#5B28C5", "#FFFFFF"),  # auto
    "Aluminium Arak": ("#4127B9", "#FFFFFF"),  # auto
    "Amazonas": ("#9AD71E", "#000000"),  # auto
    "Amazulu": ("#E23979", "#FFFFFF"),  # auto
    "America Mineiro": ("#B9243F", "#FFFFFF"),  # auto
    "America de Cali": ("#2C57C3", "#FFFFFF"),  # auto
    "Amiens SC": ("#3775C0", "#FFFFFF"),  # auto
    "Angers SCO": ("#27C16E", "#FFFFFF"),  # auto
    "Ansan Greeners": ("#29D9CE", "#000000"),  # auto
    "Antalyaspor": ("#D62330", "#FFFFFF"),  # auto
    "Anyang": ("#26C491", "#FFFFFF"),  # auto
    "Arda Kardzhali": ("#A3D93E", "#000000"),  # auto
    "Aris": ("#26D47D", "#000000"),  # auto
    "Arminia Bielefeld": ("#5334DC", "#FFFFFF"),  # auto
    "Arsenal FC": ("#EF0107", "#FFFFFF"),
    "Arsenal Tivat": ("#25AEE5", "#FFFFFF"),  # auto
    "Asteras Tripolis": ("#214AD1", "#FFFFFF"),  # auto
    "Aston Villa": ("#95BFE5", "#670E36"),
    "Atalanta BC": ("#1B75BC", "#000000"),
    "Athletic Bilbao": ("#EE2523", "#FFFFFF"),
    "Athletic Club": ("#B73926", "#FFFFFF"),  # auto
    "Atlanta United": ("#DE3434", "#000000"),  # auto
    "Atlas": ("#4453DF", "#FFFFFF"),  # auto
    "Atletico Bucaramanga": ("#5EB730", "#FFFFFF"),  # auto
    "Atletico Goianiense": ("#3147DB", "#FFFFFF"),  # auto
    "Atletico Mineiro": ("#D22EC1", "#FFFFFF"),  # auto
    "Atletico Nacional": ("#B420C8", "#FFFFFF"),  # auto
    "Atletico San Luis": ("#2CCD7D", "#000000"),  # auto
    "Atlético de Madrid": ("#CB3524", "#FFFFFF"),
    "Atromitos Athens": ("#C3DB3C", "#000000"),  # auto
    "Auckland FC": ("#B9204E", "#FFFFFF"),  # auto
    "Audax Italiano": ("#38C47F", "#FFFFFF"),  # auto
    "Austin FC": ("#BD288E", "#FFFFFF"),  # auto
    "Austria Vienna": ("#C524CB", "#FFFFFF"),  # auto
    "Avai": ("#D28237", "#FFFFFF"),  # auto
    "Avellino": ("#43C0DF", "#FFFFFF"),  # auto
    "Avispa Fukuoka": ("#28CA3F", "#FFFFFF"),  # auto
    "Avs Futebol": ("#255EBD", "#FFFFFF"),  # auto
    "BSC Young Boys": ("#7432BE", "#FFFFFF"),  # auto
    "Baltika": ("#B4BF32", "#000000"),  # auto
    "Banik Ostrava": ("#67E120", "#000000"),  # auto
    "Bari": ("#DA5E37", "#FFFFFF"),  # auto
    "Bayer 04 Leverkusen": ("#E32221", "#000000"),
    "Beijing Guoan": ("#D43A9B", "#FFFFFF"),  # auto
    "Beitar Jerusalem": ("#B7A91A", "#FFFFFF"),  # auto
    "Ben Guerdane": ("#9421C0", "#FFFFFF"),  # auto
    "Beroe Stara Zagora": ("#CDC02E", "#000000"),  # auto
    "Besiktas JK": ("#000000", "#FFFFFF"),
    "Birmingham City": ("#E52D31", "#FFFFFF"),  # auto
    "Bizertin": ("#AA3DE2", "#FFFFFF"),  # auto
    "Blackburn Rovers": ("#50D028", "#000000"),  # auto
    "Blau-Weiss Linz": ("#1FD578", "#000000"),  # auto
    "Blaublitz Akita": ("#9F37CC", "#FFFFFF"),  # auto
    "Bnei Sakhnin": ("#DEA828", "#FFFFFF"),  # auto
    "Boavista FC": ("#C95036", "#FFFFFF"),  # auto
    "Bochum": ("#D3228E", "#FFFFFF"),  # auto
    "Bodrum FK": ("#C52DC3", "#FFFFFF"),  # auto
    "Bohemians Prague 1905": ("#3E22BF", "#FFFFFF"),  # auto
    "Bokelj": ("#6236C9", "#FFFFFF"),  # auto
    "Bologna FC 1909": ("#1D428A", "#C8102E"),
    "Borussia Dortmund": ("#FDE100", "#000000"),
    "Borussia M'gladbach": ("#FFFFFF", "#000000"),
    "Botafogo SP": ("#E2A429", "#FFFFFF"),  # auto
    "Botev Plovdiv": ("#7034D1", "#FFFFFF"),  # auto
    "Botev Vratsa": ("#28DB2E", "#000000"),  # auto
    "Boulogne": ("#CD4A3A", "#FFFFFF"),  # auto
    "Boyaca Chico": ("#1C8ECC", "#FFFFFF"),  # auto
    "Brentford FC": ("#E30613", "#FFFFFF"),
    "Brescia Calcio": ("#82BD30", "#FFFFFF"),  # auto
    "Brighton & Hove Albion": ("#0057B8", "#FFFFFF"),
    "Brisbane Roar": ("#3CACC2", "#FFFFFF"),  # auto
    "Bristol City": ("#6D3FE2", "#FFFFFF"),  # auto
    "Bröndby IF": ("#BD8F25", "#FFFFFF"),  # auto
    "Bucheon 1995": ("#3AD4AF", "#000000"),  # auto
    "Buducnost": ("#D122E2", "#FFFFFF"),  # auto
    "Burgos CF": ("#2D98CF", "#FFFFFF"),  # auto
    "Burnley FC": ("#6C1D45", "#99D6EA"),
    "Busan IPark": ("#3ABBB9", "#FFFFFF"),  # auto
    "CA Banfield": ("#28C8DF", "#000000"),  # auto
    "CA Barracas Central": ("#DF30DE", "#FFFFFF"),  # auto
    "CA Boca Juniors": ("#96D236", "#000000"),  # auto
    "CA Boston River": ("#C7C828", "#000000"),  # auto
    "CA Central Cordoba (SdE)": ("#1ED551", "#000000"),  # auto
    "CA Cerro": ("#34CED4", "#000000"),  # auto
    "CA Huracán": ("#3035BB", "#FFFFFF"),  # auto
    "CA Juventud": ("#B83441", "#FFFFFF"),  # auto
    "CA Newell's Old Boys": ("#252DDC", "#FFFFFF"),  # auto
    "CA Osasuna": ("#0A346F", "#D91A32"),
    "CA Progreso": ("#3B2AD2", "#FFFFFF"),  # auto
    "CA River Plate": ("#A020CE", "#FFFFFF"),  # auto
    "CA Rosario Central": ("#8820DA", "#FFFFFF"),  # auto
    "CA San Lorenzo de Almagro": ("#D420D2", "#FFFFFF"),  # auto
    "CA San Martin (San Juan)": ("#91D624", "#000000"),  # auto
    "CA Sarmiento (Junin)": ("#D9224E", "#FFFFFF"),  # auto
    "CA Velez Sarsfield": ("#6CBC39", "#FFFFFF"),  # auto
    "CD Cruz Azul": ("#DA379B", "#FFFFFF"),  # auto
    "CD Godoy Cruz Antonio Tomba": ("#DD4C44", "#FFFFFF"),  # auto
    "CD Leganés": ("#62C038", "#FFFFFF"),  # auto
    "CD Mirandés": ("#26DE50", "#000000"),  # auto
    "CD Nacional": ("#26D39F", "#000000"),  # auto
    "CD Popular Junior FC SA": ("#4BCA37", "#FFFFFF"),  # auto
    "CD Santa Clara": ("#BC2894", "#FFFFFF"),  # auto
    "CF América": ("#C47225", "#FFFFFF"),  # auto
    "CF Estrela Amadora": ("#9FB837", "#FFFFFF"),  # auto
    "CF Montreal": ("#D643B1", "#FFFFFF"),  # auto
    "CFR Cluj": ("#2DC285", "#FFFFFF"),  # auto
    "CODM Meknes": ("#366AE0", "#FFFFFF"),  # auto
    "CR Flamengo": ("#C0A93A", "#FFFFFF"),  # auto
    "CRB": ("#C1622C", "#FFFFFF"),  # auto
    "CS Sfaxien": ("#33D5B7", "#000000"),  # auto
    "CSKA 1948": ("#A930CA", "#FFFFFF"),  # auto
    "CSKA Moscow": ("#D71920", "#0057B8"),
    "CSKA-Sofia": ("#DB1F79", "#FFFFFF"),  # auto
    "Cagliari Calcio": ("#00205B", "#D50032"),
    "Calcio Padova": ("#C3773D", "#FFFFFF"),  # auto
    "Cambuur": ("#252EC0", "#FFFFFF"),  # auto
    "Carrarese Calcio 1908": ("#B0D332", "#000000"),  # auto
    "Casa Pia": ("#20C449", "#FFFFFF"),  # auto
    "Castellon": ("#8340E0", "#FFFFFF"),  # auto
    "Catanzaro": ("#43DEE2", "#000000"),  # auto
    "Caykur Rizespor": ("#2447DD", "#FFFFFF"),  # auto
    "Ceará SC": ("#80D72F", "#000000"),  # auto
    "Celta de Vigo": ("#8AD1F5", "#FFFFFF"),
    "Celtic FC": ("#018749", "#FFFFFF"),
    "Central Coast Mariners": ("#33BCB0", "#FFFFFF"),  # auto
    "Ceramica Cleo": ("#BD6B1B", "#FFFFFF"),  # auto
    "Cercle Brugge": ("#6D31D4", "#FFFFFF"),  # auto
    "Cerezo Osaka": ("#BD702F", "#FFFFFF"),  # auto
    "Cerro Largo": ("#DFDC2A", "#000000"),  # auto
    "Cesena": ("#6025D6", "#FFFFFF"),  # auto
    "Chadormalu SC": ("#31DE2A", "#000000"),  # auto
    "Chapecoense": ("#2797C0", "#FFFFFF"),  # auto
    "Charlotte FC": ("#38CA30", "#FFFFFF"),  # auto
    "Charlton Athletic": ("#3131C9", "#FFFFFF"),  # auto
    "Chelsea FC": ("#034694", "#FFFFFF"),
    "Chengdu Rongcheng": ("#D01F4B", "#FFFFFF"),  # auto
    "Cheonan City": ("#E39E2C", "#FFFFFF"),  # auto
    "Cheongju": ("#E2E047", "#000000"),  # auto
    "Cherno more": ("#3C26C5", "#FFFFFF"),  # auto
    "Chicago Fire FC": ("#27C171", "#FFFFFF"),  # auto
    "Chippa United": ("#9B26C8", "#FFFFFF"),  # auto
    "Chongqing Tongliang Long FC": ("#DA3022", "#FFFFFF"),  # auto
    "Chungnam Asan": ("#953CD7", "#FFFFFF"),  # auto
    "Cincinnati": ("#AF41CB", "#FFFFFF"),  # auto
    "Cittadella": ("#3DE274", "#000000"),  # auto
    "Clermont Foot 63": ("#39E06B", "#000000"),  # auto
    "Club Africain": ("#32BCCC", "#FFFFFF"),  # auto
    "Club Athletico Paranaense": ("#CD4066", "#FFFFFF"),  # auto
    "Club Atletico Tucuman": ("#3F2FB8", "#FFFFFF"),  # auto
    "Club Atletico Union": ("#DD6131", "#FFFFFF"),  # auto
    "Club Atlético Belgrano": ("#90DF42", "#000000"),  # auto
    "Club Atlético Tigre": ("#3BCE66", "#000000"),  # auto
    "Club Brugge KV": ("#005CAB", "#000000"),
    "Club Sportivo Miramar Misiones": ("#20C622", "#FFFFFF"),  # auto
    "Cobresal": ("#67C023", "#FFFFFF"),  # auto
    "Colo Colo": ("#3BC36C", "#FFFFFF"),  # auto
    "Colorado Rapids": ("#A1C43C", "#000000"),  # auto
    "Columbus Crew": ("#22B9B8", "#FFFFFF"),  # auto
    "Como 1907": ("#0057B8", "#FFFFFF"),
    "Consadole Sapporo": ("#E0592F", "#FFFFFF"),  # auto
    "Coquimbo Unido": ("#D14092", "#FFFFFF"),  # auto
    "Cordoba CF": ("#B51AB8", "#FFFFFF"),  # auto
    "Corinthians": ("#FFFFFF", "#000000"),
    "Coritiba": ("#2F8AD2", "#FFFFFF"),  # auto
    "Cosenza": ("#AA32D9", "#FFFFFF"),  # auto
    "Coventry City": ("#C33146", "#FFFFFF"),  # auto
    "Cremonese": ("#A6192E", "#707372"),
    "Criciúma Esporte Clube": ("#88E52A", "#000000"),  # auto
    "Cruzeiro EC": ("#AD3BBF", "#FFFFFF"),  # auto
    "Crvena Zvezda": ("#3577BC", "#FFFFFF"),  # auto
    "Crystal Palace": ("#1B458F", "#C4122E"),
    "Csikszereda": ("#27C645", "#FFFFFF"),  # auto
    "Cuiaba": ("#DD46D1", "#FFFFFF"),  # auto
    "CyD Leonesa": ("#A5BC1A", "#FFFFFF"),  # auto
    "Cádiz CF": ("#D2333C", "#FFFFFF"),  # auto
    "D.C. United": ("#40CA9F", "#000000"),  # auto
    "Daegu": ("#43C940", "#FFFFFF"),  # auto
    "Daejeon Hana Citizen": ("#E43E8F", "#FFFFFF"),  # auto
    "Dalian Yingbo": ("#7526C4", "#FFFFFF"),  # auto
    "Dallas": ("#CA5633", "#FFFFFF"),  # auto
    "Damac FC": ("#A838E1", "#FFFFFF"),  # auto
    "Danubio": ("#D62953", "#FFFFFF"),  # auto
    "De Graafschap": ("#41DE2D", "#000000"),  # auto
    "Decic": ("#26D14B", "#000000"),  # auto
    "Defensa y Justicia": ("#4FC52B", "#FFFFFF"),  # auto
    "Defensor Sporting": ("#CA1FB1", "#FFFFFF"),  # auto
    "Degerfors": ("#C07038", "#FFFFFF"),  # auto
    "Den Bosch": ("#2AC892", "#FFFFFF"),  # auto
    "Dender": ("#32BB73", "#FFFFFF"),  # auto
    "Deportes Iquique": ("#4933CB", "#FFFFFF"),  # auto
    "Deportes Limache": ("#C73227", "#FFFFFF"),  # auto
    "Deportes Tolima": ("#E1D938", "#000000"),  # auto
    "Deportivo Alavés": ("#414BDD", "#FFFFFF"),  # auto
    "Deportivo Cali": ("#C0275D", "#FFFFFF"),  # auto
    "Deportivo Pasto": ("#5DB82A", "#FFFFFF"),  # auto
    "Deportivo Pereira": ("#2C36C9", "#FFFFFF"),  # auto
    "Derby County": ("#3D47E2", "#FFFFFF"),  # auto
    "Dibba Al Fujairah": ("#3ED9AD", "#000000"),  # auto
    "Dibba Al Hisn": ("#334EDB", "#FFFFFF"),  # auto
    "Difaa El Jadida": ("#4421C1", "#FFFFFF"),  # auto
    "Dinamo Makhachkala": ("#C738CF", "#FFFFFF"),  # auto
    "Dinamo Zagreb": ("#D42EB7", "#FFFFFF"),  # auto
    "Djurgårdens IF": ("#D12F97", "#FFFFFF"),  # auto
    "Dobrudzha Dobrich": ("#2CD71F", "#000000"),  # auto
    "Dordrecht": ("#DA5236", "#FFFFFF"),  # auto
    "Dundee": ("#474DDF", "#FFFFFF"),  # auto
    "Dundee United": ("#33D98E", "#000000"),  # auto
    "Durban City": ("#36D7D7", "#000000"),  # auto
    "Dynamo Dresden": ("#4261D3", "#FFFFFF"),  # auto
    "Dynamo Moscow": ("#3D3FD8", "#FFFFFF"),  # auto
    "EA Guingamp": ("#2A5BB9", "#FFFFFF"),  # auto
    "EC Juventude": ("#7C29B7", "#FFFFFF"),  # auto
    "EC Vitória": ("#D97F26", "#FFFFFF"),  # auto
    "EGS Gafsa": ("#DAB137", "#000000"),  # auto
    "ESTAC Troyes": ("#5630C2", "#FFFFFF"),  # auto
    "Ehime FC": ("#D4346D", "#FFFFFF"),  # auto
    "Eibar": ("#3BBC7B", "#FFFFFF"),  # auto
    "Eintracht Braunschweig": ("#C2772F", "#FFFFFF"),  # auto
    "Eintracht Frankfurt": ("#E1000F", "#000000"),
    "El Gouna": ("#ABD031", "#000000"),  # auto
    "El Masry SC": ("#3CCDA8", "#000000"),  # auto
    "El Mokawloon SC": ("#C12B6D", "#FFFFFF"),  # auto
    "Elche": ("#394ED5", "#FFFFFF"),  # auto
    "Enppi SC": ("#D82CDB", "#FFFFFF"),  # auto
    "Envigado": ("#41C83A", "#FFFFFF"),  # auto
    "Esperance Tunis": ("#8745E1", "#FFFFFF"),  # auto
    "Esporte Clube Bahia": ("#27B851", "#FFFFFF"),  # auto
    "Esteghlal FC": ("#4F3AD6", "#FFFFFF"),  # auto
    "Esteghlal Khuzestan": ("#3091D2", "#FFFFFF"),  # auto
    "Estoril": ("#2D58D5", "#FFFFFF"),  # auto
    "Estudiantes": ("#BB2298", "#FFFFFF"),  # auto
    "Etoile du Sahel": ("#CE2276", "#FFFFFF"),  # auto
    "Everton": ("#3FD272", "#000000"),  # auto
    "Everton FC": ("#003399", "#FFFFFF"),
    "Excelsior Rotterdam": ("#1A99B9", "#FFFFFF"),  # auto
    "Eyüpspor": ("#E17E23", "#FFFFFF"),  # auto
    "FAR Rabat": ("#9D21D6", "#FFFFFF"),  # auto
    "FC Andorra": ("#BCC331", "#000000"),  # auto
    "FC Annecy": ("#473ED9", "#FFFFFF"),  # auto
    "FC Arouca": ("#41CE41", "#000000"),  # auto
    "FC Augsburg": ("#4CC023", "#FFFFFF"),  # auto
    "FC Baniyas": ("#9E34C9", "#FFFFFF"),  # auto
    "FC Barcelona": ("#004D98", "#A50044"),
    "FC Basel 1893": ("#D50032", "#0057B8"),
    "FC Bayern Munchen": ("#DC052D", "#0066B2"),
    "FC Botosani": ("#30C27E", "#FFFFFF"),  # auto
    "FC Copenhagen": ("#005BAB", "#FFFFFF"),
    "FC Dinamo 1948": ("#B97A37", "#FFFFFF"),  # auto
    "FC Empoli": ("#B97932", "#FFFFFF"),  # auto
    "FC Groningen": ("#3DC260", "#FFFFFF"),  # auto
    "FC Haka": ("#2F3FD2", "#FFFFFF"),  # auto
    "FC Hradec Kralove": ("#CA3895", "#FFFFFF"),  # auto
    "FC Imabari": ("#E1466E", "#FFFFFF"),  # auto
    "FC Krasnodar": ("#2DD0B7", "#000000"),  # auto
    "FC Lausanne-Sport": ("#BA3167", "#FFFFFF"),  # auto
    "FC Lorient": ("#60D22A", "#000000"),  # auto
    "FC Lugano": ("#E235DB", "#FFFFFF"),  # auto
    "FC Metz": ("#5ABA22", "#FFFFFF"),  # auto
    "FC Midtjylland": ("#7B2EC8", "#FFFFFF"),  # auto
    "FC Nantes": ("#FFF200", "#00843D"),
    "FC Nordsjaelland": ("#4FBA24", "#FFFFFF"),  # auto
    "FC Porto": ("#00428C", "#FFFFFF"),
    "FC Rapid 1923": ("#CDA32F", "#FFFFFF"),  # auto
    "FC Schalke 04": ("#BA8438", "#FFFFFF"),  # auto
    "FC St. Gallen 1879": ("#521DC3", "#FFFFFF"),  # auto
    "FC St. Pauli": ("#5B3A29", "#FFFFFF"),
    "FC Tokyo": ("#39C31D", "#FFFFFF"),  # auto
    "FC Toulouse": ("#5E2A84", "#FFFFFF"),
    "FC Trinity Zlin": ("#E3466F", "#FFFFFF"),  # auto
    "FC Utrecht": ("#ED1C24", "#FFFFFF"),
    "FC Viktoria Plzen": ("#C13A86", "#FFFFFF"),  # auto
    "FC Volendam": ("#44D9D5", "#000000"),  # auto
    "FCSB": ("#9C37CC", "#FFFFFF"),  # auto
    "FCV Farul Constanta": ("#22B91E", "#FFFFFF"),  # auto
    "FK Cukaricki": ("#309DC7", "#FFFFFF"),  # auto
    "FK Dukla Prague": ("#2777CD", "#FFFFFF"),  # auto
    "FK Jablonec": ("#D7A92E", "#FFFFFF"),  # auto
    "FK Mlada Boleslav": ("#2DAAC2", "#FFFFFF"),  # auto
    "FK Mladost Lucani": ("#4936C4", "#FFFFFF"),  # auto
    "FK Napredak Krusevac": ("#5FCD2C", "#000000"),  # auto
    "FK Novi Pazar": ("#87CA28", "#000000"),  # auto
    "FK Partizan Belgrade": ("#D8439C", "#FFFFFF"),  # auto
    "FK Radnicki 1923": ("#BC2ABC", "#FFFFFF"),  # auto
    "FK Radnicki Nis": ("#3ADE41", "#000000"),  # auto
    "FK TSC Backa Topola": ("#24D2B1", "#000000"),  # auto
    "FK Vojvodina Novi Sad": ("#C2532B", "#FFFFFF"),  # auto
    "FUS Rabat": ("#BE362B", "#FFFFFF"),  # auto
    "Fagiano Okayama": ("#B02ECA", "#FFFFFF"),  # auto
    "Fajr Sepasi": ("#D332D5", "#FFFFFF"),  # auto
    "Falkirk": ("#5EBF36", "#FFFFFF"),  # auto
    "Famalicao": ("#22D85F", "#000000"),  # auto
    "Fatih Karagümrük": ("#BBA524", "#FFFFFF"),  # auto
    "Fenerbahce": ("#002D72", "#FFED00"),
    "Ferroviaria": ("#DB7E3F", "#FFFFFF"),  # auto
    "Feyenoord Rotterdam": ("#ED1C24", "#FFFFFF"),
    "Fluminense FC": ("#DC2479", "#FFFFFF"),  # auto
    "Foolad": ("#A926CE", "#FFFFFF"),  # auto
    "Fortaleza CEIF": ("#39A0D5", "#FFFFFF"),  # auto
    "Fortaleza EC": ("#2D7EC1", "#FFFFFF"),  # auto
    "Fortuna Düsseldorf": ("#D05D2B", "#FFFFFF"),  # auto
    "Fortuna Sittard": ("#88B935", "#FFFFFF"),  # auto
    "Fredericia": ("#6721B7", "#FFFFFF"),  # auto
    "Frosinone Calcio": ("#E0AA20", "#FFFFFF"),  # auto
    "Fujieda MYFC": ("#7AC425", "#FFFFFF"),  # auto
    "Fulham": ("#FFFFFF", "#000000"),
    "Futebol Clube De Alverca": ("#DB3346", "#FFFFFF"),  # auto
    "GAIS": ("#3DD2CC", "#000000"),  # auto
    "Gabes": ("#9EC839", "#000000"),  # auto
    "Galatasaray": ("#A90432", "#FDB912"),
    "Gamba Osaka": ("#C927D6", "#FFFFFF"),  # auto
    "Gangwon": ("#72B92F", "#FFFFFF"),  # auto
    "Gaziantep FK": ("#21C670", "#FFFFFF"),  # auto
    "Genclerbirligi Ankara": ("#32BDD0", "#FFFFFF"),  # auto
    "Genoa CFC": ("#002D62", "#C8102E"),
    "Getafe": ("#005BBB", "#FFFFFF"),
    "Ghazl El Mahalla": ("#D7493C", "#FFFFFF"),  # auto
    "Gil Vicente": ("#22C29F", "#FFFFFF"),  # auto
    "Gimcheon Sangmu": ("#9340D4", "#FFFFFF"),  # auto
    "Gimhae FC": ("#7FC437", "#FFFFFF"),  # auto
    "Gimnasia La Plata": ("#C66428", "#FFFFFF"),  # auto
    "Gimpo FC": ("#3ACAA4", "#000000"),  # auto
    "Girona FC": ("#E4002B", "#FFFFFF"),
    "Gnistan": ("#2692CE", "#FFFFFF"),  # auto
    "Go Ahead Eagles": ("#D34296", "#FFFFFF"),  # auto
    "Goias": ("#50D43C", "#000000"),  # auto
    "Gol Gohar": ("#C42690", "#FFFFFF"),  # auto
    "Golden Arrows": ("#295DD5", "#FFFFFF"),  # auto
    "Granada CF": ("#24C4C6", "#FFFFFF"),  # auto
    "Grasshopper": ("#D42A8F", "#FFFFFF"),  # auto
    "Grazer Athletiksport Klub 1902": ("#943BD8", "#FFFFFF"),  # auto
    "Grenoble Foot 38": ("#BF22C5", "#FFFFFF"),  # auto
    "Grêmio FBPA": ("#C8318B", "#FFFFFF"),  # auto
    "Grêmio Novorizontino": ("#2A51CD", "#FFFFFF"),  # auto
    "Guadalajara": ("#2DCED8", "#000000"),  # auto
    "Gwangju": ("#3ED0AF", "#000000"),  # auto
    "Gyeongnam": ("#46D73D", "#000000"),  # auto
    "Göztepe": ("#6FD441", "#000000"),  # auto
    "HJK Helsinki": ("#42D666", "#000000"),  # auto
    "HNK Gorica": ("#B838B8", "#FFFFFF"),  # auto
    "HNK Hajduk Split": ("#E439A9", "#FFFFFF"),  # auto
    "Hacken": ("#B7E244", "#000000"),  # auto
    "Hajer": ("#DE473D", "#FFFFFF"),  # auto
    "Halmstad": ("#BD2FA4", "#FFFFFF"),  # auto
    "Hamburger SV": ("#0057B8", "#FFFFFF"),
    "Hammarby": ("#34DFC1", "#000000"),  # auto
    "Hannover 96": ("#A825B8", "#FFFFFF"),  # auto
    "Hapoel Be'er Sheva": ("#BD2157", "#FFFFFF"),  # auto
    "Hapoel Haifa": ("#3E8CC7", "#FFFFFF"),  # auto
    "Hapoel Ironi Kiryat-Shmona": ("#CD5B3A", "#FFFFFF"),  # auto
    "Hapoel Jerusalem": ("#2F42BE", "#FFFFFF"),  # auto
    "Hapoel Petah Tikva": ("#A422E1", "#FFFFFF"),  # auto
    "Hapoel Tel Aviv": ("#2FB1DD", "#FFFFFF"),  # auto
    "Haras El Hodood": ("#C5234E", "#FFFFFF"),  # auto
    "Hartberg": ("#DBD120", "#000000"),  # auto
    "Hassania Agadir": ("#2B9BCD", "#FFFFFF"),  # auto
    "Hatayspor": ("#3F36E2", "#FFFFFF"),  # auto
    "Heart of Midlothian FC": ("#3DAED2", "#FFFFFF"),  # auto
    "Heerenveen": ("#C3961C", "#FFFFFF"),  # auto
    "Hellas Verona": ("#003DA5", "#FFD100"),
    "Henan SSLM": ("#D5C33E", "#000000"),  # auto
    "Heracles Almelo": ("#1E62C8", "#FFFFFF"),  # auto
    "Hermannstadt": ("#C0285E", "#FFFFFF"),  # auto
    "Hertha BSC": ("#B833BE", "#FFFFFF"),  # auto
    "Hibernian FC": ("#D33779", "#FFFFFF"),  # auto
    "Holstein Kiel": ("#E5E035", "#000000"),  # auto
    "Houston Dynamo FC": ("#C036D3", "#FFFFFF"),  # auto
    "Huachipato": ("#D84628", "#FFFFFF"),  # auto
    "Huesca": ("#DB39C6", "#FFFFFF"),  # auto
    "Hull City": ("#BB972D", "#FFFFFF"),  # auto
    "Hwaseong": ("#CC2080", "#FFFFFF"),  # auto
    "IF Brommapojkarna": ("#2AB1E4", "#FFFFFF"),  # auto
    "IF Elfsborg": ("#B7C831", "#000000"),  # auto
    "IFK Göteborg": ("#DE363E", "#FFFFFF"),  # auto
    "IFK Norrköping": ("#D3753D", "#FFFFFF"),  # auto
    "IFK Värnamo": ("#2EDA1E", "#000000"),  # auto
    "IK Sirius": ("#CF6831", "#FFFFFF"),  # auto
    "IMT Novi Beograd": ("#1F21BE", "#FFFFFF"),  # auto
    "Ilves Tampere": ("#6427C2", "#FFFFFF"),  # auto
    "Incheon United": ("#65DE39", "#000000"),  # auto
    "Independiente": ("#7135BD", "#FFFFFF"),  # auto
    "Independiente Medellin": ("#B8472F", "#FFFFFF"),  # auto
    "Independiente Rivadavia": ("#BC2174", "#FFFFFF"),  # auto
    "Independiente Santa Fe": ("#BC32AB", "#FFFFFF"),  # auto
    "Instituto": ("#3F81DF", "#FFFFFF"),  # auto
    "Inter Bogota": ("#C7277B", "#FFFFFF"),  # auto
    "Inter Miami CF": ("#DE533A", "#FFFFFF"),  # auto
    "Inter Milan": ("#0057B8", "#000000"),
    "Inter Turku": ("#BCDC26", "#000000"),  # auto
    "Ipswich Town": ("#296BB7", "#FFFFFF"),  # auto
    "Ironi Tiberias": ("#D6436E", "#FFFFFF"),  # auto
    "Ismaily": ("#E42CBF", "#FFFFFF"),  # auto
    "Istra 1961": ("#37D48E", "#000000"),  # auto
    "Ittihad Alexandria": ("#A432D1", "#FFFFFF"),  # auto
    "Ittihad Tanger": ("#D9298D", "#FFFFFF"),  # auto
    "Iwaki FC": ("#D4394E", "#FFFFFF"),  # auto
    "JEF United Chiba": ("#5A2FE1", "#FFFFFF"),  # auto
    "Jaro": ("#E45345", "#FFFFFF"),  # auto
    "Javor Ivanjica": ("#9EDF39", "#000000"),  # auto
    "Jeddah SC": ("#6BB724", "#FFFFFF"),  # auto
    "Jedinstvo": ("#75DF36", "#000000"),  # auto
    "Jeju United": ("#46E53F", "#000000"),  # auto
    "Jeonbuk Hyundai Motors": ("#283BDD", "#FFFFFF"),  # auto
    "Jeonnam Dragons": ("#BC2647", "#FFFFFF"),  # auto
    "Jeunesse Sportive Omrane": ("#BDC025", "#000000"),  # auto
    "Jezero": ("#E2663F", "#FFFFFF"),  # auto
    "Juarez": ("#BC2745", "#FFFFFF"),  # auto
    "Jubilo Iwata": ("#CF6E24", "#FFFFFF"),  # auto
    "Juve Stabia": ("#DA3E4F", "#FFFFFF"),  # auto
    "Juventus FC": ("#FFFFFF", "#000000"),
    "KAA Gent": ("#3CABC6", "#FFFFFF"),  # auto
    "KRC Genk": ("#3CE3E0", "#000000"),  # auto
    "KTP": ("#2552D9", "#FFFFFF"),  # auto
    "KVC Westerlo": ("#D53EA3", "#FFFFFF"),  # auto
    "Kahraba Ismailia": ("#2B63DF", "#FFFFFF"),  # auto
    "Kairouan": ("#3492B9", "#FFFFFF"),  # auto
    "Kaizer Chiefs": ("#A6CB1F", "#000000"),  # auto
    "Karlsruher SC": ("#1E7CBF", "#FFFFFF"),  # auto
    "Karvina": ("#27B771", "#FFFFFF"),  # auto
    "Kashima Antlers": ("#C9D225", "#000000"),  # auto
    "Kashiwa Reysol": ("#BE345C", "#FFFFFF"),  # auto
    "Kasimpasa": ("#D42627", "#FFFFFF"),  # auto
    "Kataller Toyama": ("#D9E029", "#000000"),  # auto
    "Kawasaki Frontale": ("#34CB5F", "#FFFFFF"),  # auto
    "Kawkab Marrakech": ("#D22A65", "#FFFFFF"),  # auto
    "Kayserispor": ("#C4AF29", "#FFFFFF"),  # auto
    "Kheybar Khorramabad": ("#25C477", "#FFFFFF"),  # auto
    "Khorfakan": ("#2665BD", "#FFFFFF"),  # auto
    "Kifisia": ("#51B71E", "#FFFFFF"),  # auto
    "Kilmarnock FC": ("#3AD081", "#000000"),  # auto
    "Kocaelispor": ("#438CD8", "#FFFFFF"),  # auto
    "Konyaspor": ("#7F1ECF", "#FFFFFF"),  # auto
    "Krylya Sovetov": ("#BE2E33", "#FFFFFF"),  # auto
    "KuPS": ("#39D029", "#000000"),  # auto
    "Kyoto Sanga FC": ("#6343E0", "#FFFFFF"),  # auto
    "LASK": ("#AAE336", "#000000"),  # auto
    "LOSC Lille": ("#E01E3C", "#00205B"),
    "La Serena": ("#40CB54", "#FFFFFF"),  # auto
    "Lanus": ("#31E26B", "#000000"),  # auto
    "Larissa": ("#AECE1E", "#000000"),  # auto
    "Le Havre AC": ("#85D537", "#000000"),  # auto
    "Le Mans FC": ("#6424D5", "#FFFFFF"),  # auto
    "Leeds United": ("#FFCD00", "#1D428A"),
    "Leicester City": ("#003090", "#FFFFFF"),
    "Leon": ("#D339C6", "#FFFFFF"),  # auto
    "Levante UD": ("#32B8B6", "#FFFFFF"),  # auto
    "Levski Sofia": ("#A7D236", "#000000"),  # auto
    "Liaoning Shenyang Urban": ("#3AC7C5", "#000000"),  # auto
    "Liverpool": ("#2E1ABD", "#FFFFFF"),  # auto
    "Liverpool FC": ("#C8102E", "#00B2A9"),
    "Livingston": ("#298AD0", "#FFFFFF"),  # auto
    "Llaneros": ("#DCCA39", "#000000"),  # auto
    "Lokomotiv Moscow": ("#33BB47", "#FFFFFF"),  # auto
    "Lokomotiv Plovdiv": ("#BD641C", "#FFFFFF"),  # auto
    "Lokomotiv Sofia": ("#2DDB54", "#000000"),  # auto
    "Lokomotiva Zagreb": ("#4EE128", "#000000"),  # auto
    "Los Angeles": ("#C01DD4", "#FFFFFF"),  # auto
    "Los Angeles Galaxy": ("#26D674", "#000000"),  # auto
    "Ludogorets Razgrad": ("#42D548", "#000000"),  # auto
    "Luzern": ("#9C1FBB", "#FFFFFF"),  # auto
    "Macarthur": ("#2EBC33", "#FFFFFF"),  # auto
    "Maccabi Bnei Raina": ("#2235DA", "#FFFFFF"),  # auto
    "Maccabi Haifa": ("#27CDAC", "#000000"),  # auto
    "Maccabi Netanya": ("#E53EB7", "#FFFFFF"),  # auto
    "Maccabi Tel Aviv": ("#25DB23", "#000000"),  # auto
    "Machida Zelvia": ("#23D9DA", "#000000"),  # auto
    "Magesi": ("#25DFB0", "#000000"),  # auto
    "Maghreb Fes": ("#31C629", "#FFFFFF"),  # auto
    "Malaga": ("#9030C9", "#FFFFFF"),  # auto
    "Malavan": ("#CD7232", "#FFFFFF"),  # auto
    "Malmö FF": ("#82CB24", "#000000"),  # auto
    "Mamelodi Sundowns": ("#6920BB", "#FFFFFF"),  # auto
    "Manchester City": ("#6CABDD", "#1C2C5B"),
    "Manchester United": ("#DA291C", "#FBE122"),
    "Mantova": ("#31DC3E", "#000000"),  # auto
    "Mariehamn": ("#6640CB", "#FFFFFF"),  # auto
    "Marsa": ("#D1B63E", "#000000"),  # auto
    "Marumo Gallants": ("#84DD2B", "#000000"),  # auto
    "Mazatlan": ("#B1BA28", "#FFFFFF"),  # auto
    "Mechelen": ("#B6C33E", "#000000"),  # auto
    "Melbourne City": ("#2FD4AF", "#000000"),  # auto
    "Melbourne Victory": ("#2EBF48", "#FFFFFF"),  # auto
    "Mes Rafsanjan": ("#7E27BC", "#FFFFFF"),  # auto
    "Metaloglobus": ("#ADD51F", "#000000"),  # auto
    "Metlaoui": ("#6A1FDA", "#FFFFFF"),  # auto
    "Middlesbrough": ("#5027D6", "#FFFFFF"),  # auto
    "Millonarios": ("#CF2E6E", "#FFFFFF"),  # auto
    "Millwall": ("#C62176", "#FFFFFF"),  # auto
    "Minnesota United": ("#31CCD5", "#000000"),  # auto
    "Mirassol": ("#3B7CCB", "#FFFFFF"),  # auto
    "Mito Hollyhock": ("#D68235", "#FFFFFF"),  # auto
    "Mjällby AIF": ("#3042BE", "#FFFFFF"),  # auto
    "Mladost DG": ("#D4B52D", "#000000"),  # auto
    "Modena": ("#C05831", "#FFFFFF"),  # auto
    "Modern SC": ("#2BABC4", "#FFFFFF"),  # auto
    "Monastir": ("#6FD137", "#000000"),  # auto
    "Montana": ("#2C34BA", "#FFFFFF"),  # auto
    "Montedio Yamagata": ("#34BBAC", "#FFFFFF"),  # auto
    "Monterrey": ("#3DE52F", "#000000"),  # auto
    "Montevideo City Torque": ("#3A9BCF", "#FFFFFF"),  # auto
    "Montpellier HSC": ("#BE5724", "#FFFFFF"),  # auto
    "Moreirense": ("#C93C1E", "#FFFFFF"),  # auto
    "Mornar": ("#8CC33B", "#FFFFFF"),  # auto
    "Motherwell FC": ("#30DEC4", "#000000"),  # auto
    "NAC Breda": ("#D67622", "#FFFFFF"),  # auto
    "NEC Nijmegen": ("#1EC183", "#FFFFFF"),  # auto
    "NEOM SC": ("#00BFFF", "#002AFF"),
    "Nacional": ("#30C9B5", "#000000"),  # auto
    "Nagoya Grampus": ("#7C44E0", "#FFFFFF"),  # auto
    "Nashville SC": ("#4183D4", "#FFFFFF"),  # auto
    "National Bank of Egypt": ("#291EB7", "#FFFFFF"),  # auto
    "Necaxa": ("#E2DE21", "#000000"),  # auto
    "New England": ("#4387DB", "#FFFFFF"),  # auto
    "New York City": ("#DC3DCD", "#FFFFFF"),  # auto
    "New York RB": ("#DB3546", "#FFFFFF"),  # auto
    "Newcastle Jets": ("#387ADF", "#FFFFFF"),  # auto
    "Newcastle United": ("#FFFFFF", "#000000"),
    "Norwich City": ("#26DE9C", "#000000"),  # auto
    "Nottingham Forest": ("#DD0000", "#FFFFFF"),
    "Nublense": ("#9E1DB8", "#FFFFFF"),  # auto
    "O'Higgins": ("#3BC4CE", "#FFFFFF"),  # auto
    "OB": ("#C0413A", "#FFFFFF"),  # auto
    "OFI Crete FC": ("#1CC29C", "#FFFFFF"),  # auto
    "OFK Beograd": ("#24A6C0", "#FFFFFF"),  # auto
    "OGC Nice": ("#C8102E", "#000000"),
    "OH Leuven": ("#3AA9DD", "#FFFFFF"),  # auto
    "Ohod Club": ("#D03A8B", "#FFFFFF"),  # auto
    "Oita Trinita": ("#72C43B", "#FFFFFF"),  # auto
    "Olympiakos": ("#B91B64", "#FFFFFF"),  # auto
    "Olympic Safi": ("#2DD147", "#000000"),  # auto
    "Olympique Beja": ("#C89226", "#FFFFFF"),  # auto
    "Olympique Dcheira": ("#CFC31E", "#000000"),  # auto
    "Olympique Lyon": ("#004C99", "#E30613"),
    "Olympique Marseille": ("#2FAEE0", "#FFFFFF"),
    "Omiya Ardija": ("#C07D2C", "#FFFFFF"),  # auto
    "Once Caldas": ("#3A7FC4", "#FFFFFF"),  # auto
    "Operario PR": ("#D3D940", "#000000"),  # auto
    "Orbit College": ("#3665D5", "#FFFFFF"),  # auto
    "Orenburg": ("#8A26BD", "#FFFFFF"),  # auto
    "Orlando City": ("#D5215A", "#FFFFFF"),  # auto
    "Orlando Pirates": ("#D63753", "#FFFFFF"),  # auto
    "Osijek": ("#E12A22", "#FFFFFF"),  # auto
    "Osters IF": ("#35CCCC", "#000000"),  # auto
    "Oxford United": ("#1B23C2", "#FFFFFF"),  # auto
    "PAOK": ("#2DD9AE", "#000000"),  # auto
    "PEC Zwolle": ("#67CE34", "#000000"),  # auto
    "PSV": ("#C03D7F", "#FFFFFF"),  # auto
    "Pachuca": ("#2229B7", "#FFFFFF"),  # auto
    "Paju Citizen": ("#583DC6", "#FFFFFF"),  # auto
    "Palermo": ("#3DCFD5", "#000000"),  # auto
    "Palestino": ("#DA2E30", "#FFFFFF"),  # auto
    "Panathinaikos FC": ("#27D49B", "#000000"),  # auto
    "Panetolikos GFS": ("#58DC34", "#000000"),  # auto
    "Panserraikos": ("#256BBA", "#FFFFFF"),  # auto
    "Pardubice": ("#B827AA", "#FFFFFF"),  # auto
    "Pari Nizhny Novgorod": ("#1D4CCF", "#FFFFFF"),  # auto
    "Paris FC": ("#E3BB3B", "#000000"),  # auto
    "Paris Saint-Germain": ("#004170", "#DA291C"),
    "Parma Calcio 1913": ("#FFCC00", "#0033A0"),
    "Pau FC": ("#5B1DCA", "#FFFFFF"),  # auto
    "Paykan": ("#D82678", "#FFFFFF"),  # auto
    "Paysandu SC": ("#B421BE", "#FFFFFF"),  # auto
    "Penarol": ("#73B72A", "#FFFFFF"),  # auto
    "Persepolis FC": ("#C822BF", "#FFFFFF"),  # auto
    "Perth Glory": ("#D83ADC", "#FFFFFF"),  # auto
    "Pescara": ("#D23E72", "#FFFFFF"),  # auto
    "Petrojet": ("#843AC4", "#FFFFFF"),  # auto
    "Petrolul 52": ("#B91B95", "#FFFFFF"),  # auto
    "Petrovac": ("#7629CC", "#FFFFFF"),  # auto
    "Pharco FC": ("#3875CB", "#FFFFFF"),  # auto
    "Philadelphia Union": ("#1EBD3C", "#FFFFFF"),  # auto
    "Pisa Sporting Club": ("#0057B8", "#000000"),
    "Platense": ("#7ECA37", "#000000"),  # auto
    "Plaza Colonia": ("#2EE1A2", "#000000"),  # auto
    "Pohang Steelers": ("#D4339C", "#FFFFFF"),  # auto
    "Polokwane City": ("#DEA62D", "#FFFFFF"),  # auto
    "Portland Timbers": ("#D12D5C", "#FFFFFF"),  # auto
    "Portsmouth FC": ("#89DA43", "#000000"),  # auto
    "Preston North End": ("#25BCD4", "#FFFFFF"),  # auto
    "Preußen Münster": ("#DE25D3", "#FFFFFF"),  # auto
    "Puebla": ("#8EBC2A", "#FFFFFF"),  # auto
    "Pumas UNAM": ("#204DE5", "#FFFFFF"),  # auto
    "Pyramids": ("#C02E2F", "#FFFFFF"),  # auto
    "Qatar SC": ("#3940D9", "#FFFFFF"),  # auto
    "Qingdao Hainiu": ("#90C427", "#000000"),  # auto
    "Qingdao West Coast": ("#70C724", "#FFFFFF"),  # auto
    "Queens Park Rangers": ("#C7B23A", "#FFFFFF"),  # auto
    "Queretaro": ("#9D3EDE", "#FFFFFF"),  # auto
    "RAAL La Louvière": ("#2AC666", "#FFFFFF"),  # auto
    "RB Leipzig": ("#FFFFFF", "#DD0741"),
    "RC Deportivo": ("#7BB81D", "#FFFFFF"),  # auto
    "RC Lens": ("#D71920", "#FFCD00"),
    "RC Strasbourg Alsace": ("#0055A4", "#FFFFFF"),
    "RCD Espanyol Barcelona": ("#0070C0", "#FFFFFF"),
    "RCD Mallorca": ("#E30613", "#000000"),
    "RKC Waalwijk": ("#DFC233", "#000000"),  # auto
    "RSB Berkane": ("#C25635", "#FFFFFF"),  # auto
    "RSC Anderlecht": ("#552583", "#FFFFFF"),
    "Racing Club": ("#DD5D3E", "#FFFFFF"),  # auto
    "Racing Montevideo": ("#25E3D2", "#000000"),  # auto
    "Racing Santander": ("#C32C41", "#FFFFFF"),  # auto
    "Radnik Surdulica": ("#52E430", "#000000"),  # auto
    "Raja Club Athletic": ("#6F28CC", "#FFFFFF"),  # auto
    "Randers": ("#BA8633", "#FFFFFF"),  # auto
    "Rangers FC": ("#0033A0", "#FFFFFF"),
    "Rapid Vienna": ("#21C44C", "#FFFFFF"),  # auto
    "Rayo Vallecano": ("#FFFFFF", "#E53027"),
    "Real Betis": ("#00954C", "#FFFFFF"),
    "Real Madrid": ("#FFFFFF", "#FEBE10"),
    "Real Oviedo": ("#31DA55", "#000000"),  # auto
    "Real Salt Lake": ("#DF7240", "#FFFFFF"),  # auto
    "Real Sociedad": ("#0067B1", "#FFFFFF"),
    "Real Sociedad B": ("#7ED036", "#000000"),  # auto
    "Real Valladolid": ("#8A25BB", "#FFFFFF"),  # auto
    "Real Zaragoza": ("#DE2BDC", "#FFFFFF"),  # auto
    "Red Bull Bragantino": ("#27D228", "#000000"),  # auto
    "Red Bull Salzburg": ("#C8A620", "#FFFFFF"),  # auto
    "Red Star FC": ("#BA4C34", "#FFFFFF"),  # auto
    "Remo": ("#2ECDD5", "#000000"),  # auto
    "Renaissance Zemamra": ("#B93826", "#FFFFFF"),  # auto
    "Renofa Yamaguchi": ("#2F3ACF", "#FFFFFF"),  # auto
    "Rheindorf Altach": ("#DAA023", "#FFFFFF"),  # auto
    "Richards Bay": ("#DD455D", "#FFFFFF"),  # auto
    "Ried": ("#C97135", "#FFFFFF"),  # auto
    "Riestra": ("#BD3B57", "#FFFFFF"),  # auto
    "Rijeka": ("#932FE0", "#FFFFFF"),  # auto
    "Rio Ave FC": ("#6C23C3", "#FFFFFF"),  # auto
    "River Plate": ("#FFFFFF", "#E31E24"),
    "Roasso Kumamoto": ("#8AB934", "#FFFFFF"),  # auto
    "Rodez": ("#C27931", "#FFFFFF"),  # auto
    "Rostov": ("#DA7D23", "#FFFFFF"),  # auto
    "Royal Antwerpen": ("#2248D3", "#FFFFFF"),  # auto
    "Rubin Kazan": ("#E4D72D", "#000000"),  # auto
    "SAF Botafogo": ("#40E084", "#000000"),  # auto
    "SC Bastia": ("#CAB43E", "#000000"),  # auto
    "SC Farense": ("#E23B7A", "#FFFFFF"),  # auto
    "SC Freiburg": ("#E30613", "#000000"),
    "SC Internacional": ("#7B22DC", "#FFFFFF"),  # auto
    "SC Otelul Galati": ("#28D03E", "#000000"),  # auto
    "SC Paderborn 07": ("#4323D1", "#FFFFFF"),  # auto
    "SE Palmeiras": ("#C8C940", "#000000"),  # auto
    "SJ Earthquakes": ("#34C775", "#FFFFFF"),  # auto
    "SJK": ("#3DBDDD", "#FFFFFF"),  # auto
    "SK Slavia Prague": ("#D84137", "#FFFFFF"),  # auto
    "SK Sturm Graz": ("#2BC727", "#FFFFFF"),  # auto
    "SL Benfica": ("#E83030", "#FFFFFF"),
    "SS Lazio": ("#87D8F7", "#FFFFFF"),
    "SSC Napoli": ("#12A0D7", "#003B79"),
    "SV 07 Elversberg": ("#1DC956", "#FFFFFF"),  # auto
    "SV Darmstadt 98": ("#C4C339", "#000000"),  # auto
    "SV Werder Bremen": ("#1D9053", "#FFFFFF"),
    "Sagan Tosu": ("#7041E1", "#FFFFFF"),  # auto
    "Salernitana": ("#CC2125", "#FFFFFF"),  # auto
    "Sampdoria": ("#C06920", "#FFFFFF"),  # auto
    "Samsunspor": ("#42D484", "#000000"),  # auto
    "San Diego FC": ("#25E1B7", "#000000"),  # auto
    "Sanfrecce Hiroshima": ("#3BB1C2", "#FFFFFF"),  # auto
    "Santos FC": ("#FFFFFF", "#000000"),
    "Santos Laguna": ("#9BDC41", "#000000"),  # auto
    "Sassuolo": ("#00A650", "#000000"),
    "Seattle Sounders": ("#81DC3F", "#000000"),  # auto
    "Sekhukhune United": ("#2273BD", "#FFFFFF"),  # auto
    "Seongnam": ("#D1BC2B", "#000000"),  # auto
    "Seoul": ("#C82A34", "#FFFFFF"),  # auto
    "Seoul E-Land": ("#DABD35", "#000000"),  # auto
    "Sepahan FC": ("#2DD421", "#000000"),  # auto
    "Septemvri Sofia": ("#DC339F", "#FFFFFF"),  # auto
    "Servette FC": ("#C31D42", "#FFFFFF"),  # auto
    "Sevilla FC": ("#D71920", "#FFFFFF"),
    "Shabab Al Ahli": ("#3148DC", "#FFFFFF"),  # auto
    "Shams Azar Qazvin": ("#CB375A", "#FFFFFF"),  # auto
    "Shandong TaiShan": ("#41CF6B", "#000000"),  # auto
    "Shanghai Port": ("#24CF7E", "#000000"),  # auto
    "Shanghai Shenhua": ("#5542E2", "#FFFFFF"),  # auto
    "Sharjah": ("#2843CC", "#FFFFFF"),  # auto
    "Sheffield United": ("#20A9BC", "#FFFFFF"),  # auto
    "Sheffield Wednesday": ("#CA223E", "#FFFFFF"),  # auto
    "Shenzhen Peng City": ("#2EADD9", "#FFFFFF"),  # auto
    "Shimizu S-Pulse": ("#45D993", "#000000"),  # auto
    "Shonan Bellmare": ("#BC2466", "#FFFFFF"),  # auto
    "Sigma Olomouc": ("#A52AD2", "#FFFFFF"),  # auto
    "Silkeborg IF": ("#29C742", "#FFFFFF"),  # auto
    "Sint-Truidense VV": ("#CD7C23", "#FFFFFF"),  # auto
    "Sion": ("#BE4839", "#FFFFFF"),  # auto
    "Sivasspor": ("#CD1D4D", "#FFFFFF"),  # auto
    "Siwelele": ("#3EC647", "#FFFFFF"),  # auto
    "Slaven Koprivnica": ("#BC1FB5", "#FFFFFF"),  # auto
    "Slavia Sofia": ("#3698BD", "#FFFFFF"),  # auto
    "Slovan Liberec": ("#DBD344", "#000000"),  # auto
    "Smouha": ("#7D40CB", "#FFFFFF"),  # auto
    "Sochi": ("#1FD3D0", "#000000"),  # auto
    "Soliman": ("#BF3155", "#FFFFFF"),  # auto
    "Southampton FC": ("#D71920", "#000000"),
    "SpVgg Greuther Fürth": ("#AEBE1F", "#000000"),  # auto
    "Sparta Rotterdam": ("#23BCA1", "#FFFFFF"),  # auto
    "Spartak Moscow": ("#ED1C24", "#FFFFFF"),
    "Spartak Subotica": ("#D7D32C", "#000000"),  # auto
    "Spartak Varna": ("#D0D733", "#000000"),  # auto
    "Spezia Calcio": ("#D26B3D", "#FFFFFF"),  # auto
    "Sport Club Ironi Ashdod": ("#D08B36", "#FFFFFF"),  # auto
    "Sport Recife": ("#D1513E", "#FFFFFF"),  # auto
    "Sporting Braga": ("#B89637", "#FFFFFF"),  # auto
    "Sporting CP": ("#00843D", "#FFFFFF"),
    "Sporting Charleroi": ("#D6C435", "#000000"),  # auto
    "Sporting Gijón": ("#E42758", "#FFFFFF"),  # auto
    "Sporting KC": ("#2EBC29", "#FFFFFF"),  # auto
    "St. Louis City": ("#22DFAC", "#000000"),  # auto
    "St. Mirren FC": ("#1CC892", "#FFFFFF"),  # auto
    "Stade Brestois 29": ("#2DC07D", "#FFFFFF"),  # auto
    "Stade Lavallois": ("#42D62E", "#000000"),  # auto
    "Stade Reims": ("#C77A3B", "#FFFFFF"),  # auto
    "Stade Rennais FC": ("#E30613", "#000000"),
    "Stade Tunisien": ("#C7B435", "#000000"),  # auto
    "Standard Liege": ("#C9933A", "#FFFFFF"),  # auto
    "Stellenbosch": ("#C9298E", "#FFFFFF"),  # auto
    "Stoke City": ("#C226A7", "#FFFFFF"),  # auto
    "Sudtirol": ("#362CE2", "#FFFFFF"),  # auto
    "Sunderland AFC": ("#EB172B", "#FFFFFF"),
    "Sutjeska": ("#38E4DA", "#000000"),  # auto
    "Suwon": ("#D236BA", "#FFFFFF"),  # auto
    "Suwon Samsung Bluewings": ("#7F2CCE", "#FFFFFF"),  # auto
    "Swansea City": ("#2597D2", "#FFFFFF"),  # auto
    "Sydney FC": ("#BE3651", "#FFFFFF"),  # auto
    "São Paulo FC": ("#A322BC", "#FFFFFF"),  # auto
    "Sönderjyske": ("#D92D4C", "#FFFFFF"),  # auto
    "TS Galaxy": ("#782CE4", "#FFFFFF"),  # auto
    "TSG 1899 Hoffenheim": ("#005DAA", "#FFFFFF"),
    "Tala'ea El Gaish": ("#843FCE", "#FFFFFF"),  # auto
    "Talleres de Cordoba": ("#B6D140", "#000000"),  # auto
    "Tataouine": ("#E35147", "#FFFFFF"),  # auto
    "Telstar": ("#6AE12E", "#000000"),  # auto
    "Teplice": ("#43D559", "#000000"),  # auto
    "Thun": ("#4843D5", "#FFFFFF"),  # auto
    "Tianjin JMT": ("#DE26CE", "#FFFFFF"),  # auto
    "Tigres UANL": ("#7DD825", "#000000"),  # auto
    "Tijuana": ("#E3445F", "#FFFFFF"),  # auto
    "Tokushima Vortis": ("#27D1A5", "#000000"),  # auto
    "Tokyo Verdy": ("#C63A6C", "#FFFFFF"),  # auto
    "Toluca": ("#51D63E", "#000000"),  # auto
    "Tondela": ("#DC3C30", "#FFFFFF"),  # auto
    "Torino FC": ("#8A1538", "#FFFFFF"),
    "Toronto": ("#4194E1", "#FFFFFF"),  # auto
    "Tottenham Hotspur": ("#FFFFFF", "#132257"),
    "Trabzonspor": ("#6F263D", "#0072CE"),
    "Tractor FC": ("#7C38DD", "#FFFFFF"),  # auto
    "Twente Enschede FC": ("#36C778", "#FFFFFF"),  # auto
    "UD Las Palmas": ("#C12763", "#FFFFFF"),  # auto
    "US Lecce": ("#E31B23", "#FFD200"),
    "US Yacoub El Mansour": ("#C839A1", "#FFFFFF"),  # auto
    "USL Dunkerque": ("#D8349C", "#FFFFFF"),  # auto
    "UTA Arad": ("#BC3425", "#FFFFFF"),  # auto
    "UTS Rabat": ("#A93ED7", "#FFFFFF"),  # auto
    "Udinese Calcio": ("#FFFFFF", "#000000"),
    "Ulsan": ("#3656BB", "#FFFFFF"),  # auto
    "Umm Salal SC": ("#C91ED6", "#FFFFFF"),  # auto
    "Union Espanola": ("#2ED264", "#000000"),  # auto
    "Union La Calera": ("#D9A220", "#FFFFFF"),  # auto
    "Union Magdalena": ("#2DC021", "#FFFFFF"),  # auto
    "Union Saint-Gilloise": ("#C7783A", "#FFFFFF"),  # auto
    "Universidad Catolica": ("#DA338A", "#FFFFFF"),  # auto
    "Universidad de Chile": ("#86DB23", "#000000"),  # auto
    "Universitatea Cluj": ("#D02561", "#FFFFFF"),  # auto
    "Universitatea Craiova": ("#22DFD8", "#000000"),  # auto
    "Urawa Reds": ("#223DC1", "#FFFFFF"),  # auto
    "V-Varen Nagasaki": ("#34DA7C", "#000000"),  # auto
    "VPS": ("#5F30CE", "#FFFFFF"),  # auto
    "Valencia CF": ("#F18E00", "#000000"),
    "Vancouver Whitecaps": ("#4A27C7", "#FFFFFF"),  # auto
    "Varazdin": ("#32C290", "#FFFFFF"),  # auto
    "Vasco da Gama": ("#28A0BE", "#FFFFFF"),  # auto
    "Vegalta Sendai": ("#4126D0", "#FFFFFF"),  # auto
    "Vejle Boldklub": ("#DD2162", "#FFFFFF"),  # auto
    "Venezia FC": ("#000000", "#F58220"),
    "Ventforet Kofu": ("#423AC8", "#FFFFFF"),  # auto
    "VfB Stuttgart": ("#E32219", "#FFFFFF"),
    "VfL Wolfsburg": ("#65B32E", "#FFFFFF"),
    "Viborg": ("#CE39C1", "#FFFFFF"),  # auto
    "Vila Nova": ("#D1DB3A", "#000000"),  # auto
    "Villarreal CF": ("#FFE667", "#005187"),
    "Virtus Entella": ("#C94538", "#FFFFFF"),  # auto
    "Vissel Kobe": ("#C79B24", "#FFFFFF"),  # auto
    "Vitoria Guimaraes": ("#39C753", "#FFFFFF"),  # auto
    "Vizela": ("#8244DF", "#FFFFFF"),  # auto
    "Volos NFC": ("#8ED627", "#000000"),  # auto
    "Volta Redonda": ("#A62DC8", "#FFFFFF"),  # auto
    "Vukovar 1991": ("#D72297", "#FFFFFF"),  # auto
    "WSG Tirol": ("#287ED1", "#FFFFFF"),  # auto
    "Wadi Degla": ("#BC3944", "#FFFFFF"),  # auto
    "Wanderers": ("#3482BB", "#FFFFFF"),  # auto
    "Watford": ("#3ED898", "#000000"),  # auto
    "Wellington Phoenix": ("#C6DF45", "#000000"),  # auto
    "West Bromwich Albion": ("#39D5C8", "#000000"),  # auto
    "West Ham United": ("#7A263A", "#1BB1E7"),
    "Western Sydney Wanderers": ("#4FD637", "#000000"),  # auto
    "Willem II": ("#99DC35", "#000000"),  # auto
    "Winterthur": ("#51DC41", "#000000"),  # auto
    "Wolfsberger AC": ("#A6D03D", "#000000"),  # auto
    "Wolverhampton Wanderers": ("#FDB913", "#231F20"),
    "Wrexham": ("#2DC6DF", "#000000"),  # auto
    "Wuhan Three Towns": ("#99C637", "#000000"),  # auto
    "Wydad Casablanca": ("#D50032", "#FFFFFF"),
    "Yokohama F. Marinos": ("#2B65D2", "#FFFFFF"),  # auto
    "Yokohama FC": ("#1EC3C9", "#FFFFFF"),  # auto
    "Yongin City Government FC": ("#8B32D4", "#FFFFFF"),  # auto
    "Yunnan Yukun": ("#4221E2", "#FFFFFF"),  # auto
    "Zamalek SC": ("#FFFFFF", "#E30613"),
    "Zarzis": ("#2169C9", "#FFFFFF"),  # auto
    "Zed FC": ("#E42A92", "#FFFFFF"),  # auto
    "Zeleznicar Pancevo": ("#31DDA8", "#000000"),  # auto
    "Zenit": ("#007AC2", "#FFFFFF"),
    "Zhejiang FC": ("#C62948", "#FFFFFF"),  # auto
    "Zob Ahan": ("#3D74D2", "#FFFFFF"),  # auto
    "Zulte Waregem": ("#DB3A64", "#FFFFFF"),  # auto
    "Zurich": ("#D2D432", "#000000"),  # auto
    "İstanbul Başakşehir FK": ("#91C531", "#000000"),  # auto
}

NON_METRIC_EXACT = {
    "№", "Index", "Age", "Height", "Weight", "Season_key", "Season_fallback_key",
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
    if not TEAM_BASE_FILE.exists():
        raise FileNotFoundError(
            f"Non trovo il file team-level richiesto: {TEAM_BASE_FILE}. "
            "Aggiungi data/processed/team_league_base.csv.gz alla repo."
        )
    df = pd.read_csv(TEAM_BASE_FILE, compression="gzip", low_memory=False)
    return standardize_base_columns(df)


def build_league_display(df: pd.DataFrame) -> pd.Series:
    """Return 'League (Nation)' when a Nation column is available."""
    if "League" in df.columns:
        league = df["League"].astype(str).replace({"nan": np.nan, "None": np.nan})
    else:
        league = pd.Series("", index=df.index, dtype="object")

    if "Nation" not in df.columns:
        return league

    nation = df["Nation"].astype(str).replace({"nan": np.nan, "None": np.nan})
    has_nation = nation.notna() & nation.ne("")
    return league.where(~has_nation, league + " (" + nation + ")")


def standardize_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_cols = [
        "Season", "Player", "Team", "League", "Nation", "Competition",
        "Position", "Role bucket", "GK role", "style_cluster_short_label", "style_cluster_name",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": np.nan, "None": np.nan})
    if "League" in df.columns:
        df[LEAGUE_DISPLAY_COL] = build_league_display(df)
    if "Minutes played" in df.columns:
        df["Minutes played"] = clean_numeric(df["Minutes played"])
    return df


def league_display_values(df: pd.DataFrame) -> list[str]:
    work = df.copy()
    if LEAGUE_DISPLAY_COL not in work.columns and "League" in work.columns:
        work[LEAGUE_DISPLAY_COL] = build_league_display(work)
    if LEAGUE_DISPLAY_COL in work.columns:
        values = work[LEAGUE_DISPLAY_COL].dropna().astype(str).unique().tolist()
    elif "League" in work.columns:
        values = work["League"].dropna().astype(str).unique().tolist()
    else:
        values = []
    return sorted(values)


def numeric_metric_columns(df: pd.DataFrame, min_non_null: int = 8) -> list[str]:
    out: list[str] = []
    excluded = set(NON_METRIC_EXACT) | {
        "Season", "Player", "Team", "League", "Nation", "Competition", LEAGUE_DISPLAY_COL,
        "Position", "Role bucket", "GK role", "style_cluster_short_label", "style_cluster_name",
    }
    for col in df.columns:
        if col in excluded:
            continue
        lower = str(col).lower()
        if any(tok in lower for tok in NON_METRIC_TOKENS):
            continue
        values = clean_numeric(df[col])
        if values.notna().sum() >= min_non_null and values.nunique(dropna=True) > 1:
            out.append(str(col))
    preferred_order = [
        # Team-level tactical / outcome metrics
        "xG/team derived", "Goals/team derived", "xGA per match weighted", "GA per match weighted",
        "xG total derived", "Goals total derived", "Goals for total", "Goals against total",
        "xGD total derived", "Goals - xG total derived", "Goal difference total",
        "Lost balls in own half", "Lost balls", "Average distance to the goal at ball losses",
        "Ball recoveries after losses within 5 seconds",
        "Ball recoveries after losses within 5 seconds in the opponent's half of the field",
        "Ball recoveries after losses within 10 seconds",
        "Ball recoveries after losses within 10 seconds in the opponent's half of the field",
        "Goal kicks short (<15 m)", "Goal kicks medium (15-40 m)", "Goal kicks long (40+ m)",
        "Final third entries through pass", "Final third entries through carry", "Entries to the opponent's half",
        "Entries to the opponent's box", "Progressive passes", "Progressive passes accurate, %",
        # Player/GK defaults
        "Goals", "Assists", "Goals + Assists", "xG (expected goals)", "xA", "xG + xA",
        "Shots", "Shots on target, %", "Passes", "Passes accurate, %", "Key passes", "Key passes accurate, %",
        "Carry", "Dribbles", "Dribbles successful, %", "Defensive challenges", "Defensive challenges won, %",
        "Air challenges", "Air challenges won, %", "Ball recoveries", "Interceptions", "Tackles", "Tackles successful, %",
        "Ball possession, %", "Goals prevented", "Shots saved, %", "Cross claim rate",
    ]
    ranked = [c for c in preferred_order if c in out]
    ranked.extend([c for c in out if c not in ranked])
    return ranked


def big_five_mask(df: pd.DataFrame) -> pd.Series:
    """Big Five helper based on League + Nation, so Russian Premier League is excluded."""
    if "League" not in df.columns:
        return pd.Series(False, index=df.index)
    league = df["League"].astype(str)
    if "Nation" in df.columns:
        nation = df["Nation"].astype(str)
        mask = pd.Series(False, index=df.index)
        for league_name, country in BIG_FIVE_COMPETITIONS:
            mask = mask | (league.eq(league_name) & nation.eq(country))
        return mask
    # Fallback only for datasets without Nation; keep this for older local files.
    return league.isin(BIG_FIVE_LEAGUES)


def big_five_league_display_values(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    return league_display_values(df[big_five_mask(df)].copy())


def apply_base_filters(
    df: pd.DataFrame,
    season: str | None,
    league_mode: str,
    selected_leagues: list[str] | None = None,
    min_minutes: int = 0,
) -> pd.DataFrame:
    out = standardize_base_columns(df)
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

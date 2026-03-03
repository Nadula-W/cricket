import os
import json
import requests
import zipfile
import io
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# COUNTRY MAPPINGS
# -----------------------------

team_country = {
    "India": "India",
    "Australia": "Australia",
    "England": "England",
    "Pakistan": "Pakistan",
    "Sri Lanka": "Sri Lanka",
    "South Africa": "South Africa",
    "New Zealand": "New Zealand",
    "Bangladesh": "Bangladesh",
    "West Indies": "West Indies"
}

city_country = {

    # India
    "Mumbai": "India",
    "Delhi": "India",
    "Chennai": "India",
    "Kolkata": "India",
    "Bengaluru": "India",
    "Hyderabad": "India",
    "Ahmedabad": "India",
    "Pune": "India",
    "Dharamsala": "India",
    "Lucknow": "India",
    "Indore": "India",
    "Nagpur": "India",
    "Jaipur": "India",
    "Mohali": "India",

    # Australia
    "Melbourne": "Australia",
    "Sydney": "Australia",
    "Perth": "Australia",
    "Brisbane": "Australia",
    "Adelaide": "Australia",
    "Hobart": "Australia",
    "Canberra": "Australia",
    "Gold Coast": "Australia",

    # England
    "London": "England",
    "Manchester": "England",
    "Birmingham": "England",
    "Leeds": "England",
    "Nottingham": "England",
    "Southampton": "England",
    "Cardiff": "England",
    "Bristol": "England",

    # South Africa
    "Cape Town": "South Africa",
    "Johannesburg": "South Africa",
    "Durban": "South Africa",
    "Gqeberha": "South Africa",
    "Centurion": "South Africa",
    "Bloemfontein": "South Africa",
    "Paarl": "South Africa",

    # New Zealand
    "Auckland": "New Zealand",
    "Wellington": "New Zealand",
    "Christchurch": "New Zealand",
    "Dunedin": "New Zealand",
    "Hamilton": "New Zealand",
    "Napier": "New Zealand",
    "Mount Maunganui": "New Zealand",

    # Pakistan
    "Lahore": "Pakistan",
    "Karachi": "Pakistan",
    "Rawalpindi": "Pakistan",
    "Multan": "Pakistan",
    "Faisalabad": "Pakistan",

    # Sri Lanka
    "Colombo": "Sri Lanka",
    "Galle": "Sri Lanka",
    "Kandy": "Sri Lanka",
    "Dambulla": "Sri Lanka",
    "Hambantota": "Sri Lanka",

    # Bangladesh
    "Dhaka": "Bangladesh",
    "Chattogram": "Bangladesh",
    "Sylhet": "Bangladesh",
    "Khulna": "Bangladesh",

    # Zimbabwe
    "Harare": "Zimbabwe",
    "Bulawayo": "Zimbabwe",

    # Ireland
    "Dublin": "Ireland",
    "Belfast": "Ireland",

    # UAE (neutral)
    "Dubai": "UAE",
    "Abu Dhabi": "UAE",
    "Sharjah": "UAE"
}

# -----------------------------
# DOWNLOAD DATA
# -----------------------------

def download_data():
    urls = {
        "t20s": "https://cricsheet.org/downloads/t20s_json.zip",
        "odis": "https://cricsheet.org/downloads/odis_json.zip",
        "tests": "https://cricsheet.org/downloads/tests_json.zip"
    }

    os.makedirs("data", exist_ok=True)

    for folder, url in urls.items():
        print(f"Downloading {folder}...")
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        extract_path = f"data/{folder}"
        os.makedirs(extract_path, exist_ok=True)
        z.extractall(extract_path)

# -----------------------------
# EXTRACT MATCHES
# -----------------------------

def extract_matches(folder_path, match_format):
    records = []

    for file in os.listdir(folder_path):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            match = json.load(f)

        info = match["info"]

        if info.get("gender") == "female":
            continue

        if "winner" not in info.get("outcome", {}):
            continue

        records.append({
            "team1": info["teams"][0],
            "team2": info["teams"][1],
            "city": info.get("city", "Unknown"),
            "format": match_format,
            "date": info["dates"][0],
            "winner": info["outcome"]["winner"]
        })

    return records

# -----------------------------
# BUILD DATASET
# -----------------------------

def build_dataset():
    data = []

    data += extract_matches("data/t20s", "T20")
    data += extract_matches("data/odis", "ODI")
    data += extract_matches("data/tests", "TEST")

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["team1_win"] = (df["winner"] == df["team1"]).astype(int)

    return df

# -----------------------------
# HOME ADVANTAGE
# -----------------------------

def home_advantage(row):
    city = row["city"]
    team1 = row["team1"]
    team2 = row["team2"]

    city_cty = city_country.get(city)
    team1_cty = team_country.get(team1)
    team2_cty = team_country.get(team2)

    if city_cty is None:
        return 0

    if city_cty == team1_cty:
        return 1
    elif city_cty == team2_cty:
        return -1
    else:
        return 0

# -----------------------------
# RECENT FORM
# -----------------------------

def add_recent_form(df):

    def recent_win_rate(df, team, index, n=5):
        past = df.iloc[:index]
        team_matches = past[
            (past["team1"] == team) | (past["team2"] == team)
        ].tail(n)

        if len(team_matches) == 0:
            return 0.5

        wins = (team_matches["winner"] == team).sum()
        return wins / len(team_matches)

    df["team1_form"] = [
        recent_win_rate(df, row["team1"], i)
        for i, row in df.iterrows()
    ]

    df["team2_form"] = [
        recent_win_rate(df, row["team2"], i)
        for i, row in df.iterrows()
    ]

    return df

# -----------------------------
# TRAIN MODEL
# -----------------------------

def train_model(df):

    # Add home advantage
    df["home_adv"] = df.apply(home_advantage, axis=1)

    # Encode
    team_encoder = LabelEncoder()
    team_encoder.fit(pd.concat([df["team1"], df["team2"]]))

    df["team1_enc"] = team_encoder.transform(df["team1"])
    df["team2_enc"] = team_encoder.transform(df["team2"])

    city_encoder = LabelEncoder()
    df["city_enc"] = city_encoder.fit_transform(df["city"])

    format_encoder = LabelEncoder()
    df["format_enc"] = format_encoder.fit_transform(df["format"])

    # FEATURES (home_adv added back)
    X = df[[
        "team1_enc",
        "team2_enc",
        "city_enc",
        "format_enc",
        "home_adv",
        "team1_form",
        "team2_form"
    ]]

    y = df["team1_win"]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = RandomForestClassifier(
        n_estimators=800,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Model Accuracy:", accuracy)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.pkl")
    joblib.dump(team_encoder, "models/team_encoder.pkl")
    joblib.dump(city_encoder, "models/city_encoder.pkl")
    joblib.dump(format_encoder, "models/format_encoder.pkl")

    df.to_pickle("models/full_dataset.pkl")

    print("Model and encoders saved in /models")

# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    download_data()
    df = build_dataset()
    df = add_recent_form(df)
    train_model(df)
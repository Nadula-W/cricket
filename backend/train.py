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

  # --------------------------
# Encode categorical columns
# --------------------------

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    team_encoder = LabelEncoder()
    city_encoder = LabelEncoder()
    format_encoder = LabelEncoder()

    # 🔥 Clean string values (important!)
    df["team1"] = df["team1"].astype(str).str.strip()
    df["team2"] = df["team2"].astype(str).str.strip()
    df["city"] = df["city"].astype(str).str.strip()
    df["format"] = df["format"].astype(str).str.strip()

    # 🔥 FIT ON BOTH TEAM COLUMNS
    all_teams = pd.concat([df["team1"], df["team2"]]).unique()
    team_encoder.fit(all_teams)

    df["team1_enc"] = team_encoder.transform(df["team1"])
    df["team2_enc"] = team_encoder.transform(df["team2"])

    # 🔥 Fit city and format normally
    df["city_enc"] = city_encoder.fit_transform(df["city"])
    df["format_enc"] = format_encoder.fit_transform(df["format"])
    # --------------------------
    # NEW FEATURES
    # --------------------------

    # Head-to-head ratio
    df["h2h_ratio"] = df.apply(
        lambda row: (
            (
                (
                    df[
                        ((df["team1"] == row["team1"]) & (df["team2"] == row["team2"])) |
                        ((df["team1"] == row["team2"]) & (df["team2"] == row["team1"]))
                    ]["winner"] == row["team1"]
                ).sum()
            ) /
            max(
                1,
                len(
                    df[
                        ((df["team1"] == row["team1"]) & (df["team2"] == row["team2"])) |
                        ((df["team1"] == row["team2"]) & (df["team2"] == row["team1"]))
                    ]
                )
            )
        ),
        axis=1
    )

    # Recent win percentage (last 10)
    def recent_win_pct(team, date):
        past = df[
            ((df["team1"] == team) | (df["team2"] == team)) &
            (df["date"] < date)
        ].tail(10)

        if len(past) == 0:
            return 0.5

        wins = (past["winner"] == team).sum()
        return wins / len(past)

    df["recent_win_pct"] = df.apply(
        lambda row: recent_win_pct(row["team1"], row["date"]),
        axis=1
    )

    # --------------------------
    # Final Features
    # --------------------------

    features = [
        "team1_enc",
        "team2_enc",
        "city_enc",
        "format_enc",
        "home_adv",
        "team1_form",
        "team2_form",
        "h2h_ratio",
        "recent_win_pct"
    ]

    X = df[features]
    y = df["team1_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)
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
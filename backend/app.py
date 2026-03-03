from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
from train import download_data, build_dataset, add_recent_form, train_model

from features import build_latest_form, get_latest_form, calculate_home_adv

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
# Load model and encoders
os.makedirs("models", exist_ok=True)

if not os.path.exists("models/model.pkl"):
    print("Model not found. Training model...")
    download_data()
    df = build_dataset()
    df = add_recent_form(df)
    train_model(df)
else:
    print("Model found. Loading model...")

model = joblib.load("models/model.pkl")
team_encoder = joblib.load("models/team_encoder.pkl")
city_encoder = joblib.load("models/city_encoder.pkl")
format_encoder = joblib.load("models/format_encoder.pkl")
# Load dataset for form calculation
df = pd.read_pickle("models/full_dataset.pkl")
latest_form = build_latest_form(df)

# Country mappings
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

@app.route("/options", methods=["GET"])
def get_options():
    return jsonify({
        "teams": list(team_encoder.classes_),
        "cities": list(city_encoder.classes_),
        "formats": list(format_encoder.classes_)
    })

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    team1 = data["team1"]
    team2 = data["team2"]
    city = data["city"]
    match_format = data["format"]

    team1_form = get_latest_form(team1, latest_form)
    team2_form = get_latest_form(team2, latest_form)

    h2h = df[
        ((df["team1"] == team1) & (df["team2"] == team2)) |
        ((df["team1"] == team2) & (df["team2"] == team1))
    ]

    team1_h2h_wins = (h2h["winner"] == team1).sum()
    team2_h2h_wins = (h2h["winner"] == team2).sum()
    
    home_adv = calculate_home_adv(
        team1, team2, city,
        team_country, city_country
    )

    last5_team1 = list(latest_form.get(team1, []))
    last5_team2 = list(latest_form.get(team2, []))

    t1 = team_encoder.transform([team1])[0]
    t2 = team_encoder.transform([team2])[0]
    c = city_encoder.transform([city])[0]
    f = format_encoder.transform([match_format])[0]

    X_input = pd.DataFrame([{
    "team1_enc": t1,
    "team2_enc": t2,
    "city_enc": c,
    "format_enc": f,
    "home_adv": home_adv,
    "team1_form": team1_form,
    "team2_form": team2_form
}])

    prob = model.predict_proba(X_input)[0][1]

    return jsonify({
        "team1": team1,
        "team2": team2,
        "probability": round(prob * 100, 2),
        "team1_form": round(team1_form, 2),
        "team2_form": round(team2_form, 2),
        "home_adv": home_adv,
        "last5_team1": last5_team1,
        "last5_team2": last5_team2,
        "head_to_head": {
            "team1_wins": int(team1_h2h_wins),
            "team2_wins": int(team2_h2h_wins)
        }
    })


if __name__ == "__main__":
    app.run(debug=True)
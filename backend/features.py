import pandas as pd
from collections import defaultdict, deque

def build_latest_form(df):

    latest_form = defaultdict(lambda: deque(maxlen=5))

    for _, row in df.iterrows():
        t1, t2, w = row["team1"], row["team2"], row["winner"]
        latest_form[t1].append(1 if w == t1 else 0)
        latest_form[t2].append(1 if w == t2 else 0)

    return latest_form


def get_latest_form(team, latest_form):
    history = latest_form.get(team, [])
    if len(history) == 0:
        return 0.5
    return sum(history) / len(history)


def calculate_home_adv(team1, team2, city, team_country, city_country):

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
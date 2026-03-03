import React, { useState, useEffect } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement } from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement);

const API =
  process.env.REACT_APP_API_URL ||
  (window.location.hostname === "localhost"
    ? "http://127.0.0.1:5000"
    : "https://cricket-0f5q.onrender.com");

const teamFlags = {
  India: "in",
  Australia: "au",
  England: "gb",
  Pakistan: "pk",
  "Sri Lanka": "lk",
  "South Africa": "za",
  "New Zealand": "nz",
  Bangladesh: "bd",
  "West Indies": "jm"
};

function getFlag(team) {
  if (!teamFlags[team]) return "";
  return `https://flagcdn.com/w80/${teamFlags[team]}.png`;
}

function App() {

  const [teams, setTeams] = useState([]);
  const [cities, setCities] = useState([]);
  const [formats, setFormats] = useState([]);

  const [team1, setTeam1] = useState("");
  const [team2, setTeam2] = useState("");
  const [city, setCity] = useState("");
  const [format, setFormat] = useState("");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dark, setDark] = useState(false);

  useEffect(() => {
    axios.get(`${API}/options`)
      .then(res => {
        setTeams(res.data.teams || []);
        setCities(res.data.cities || []);
        setFormats(res.data.formats || []);

        if (res.data.teams?.length > 1) {
          setTeam1(res.data.teams[0]);
          setTeam2(res.data.teams[1]);
        }

        if (res.data.cities?.length > 0)
          setCity(res.data.cities[0]);

        if (res.data.formats?.length > 0)
          setFormat(res.data.formats[0]);
      })
      .catch(err => {
        console.error("Options load failed:", err);
      });
  }, []);

  const predict = async () => {
    try {
      setLoading(true);
      setResult(null);

      const res = await axios.post(`${API}/predict`, {
        team1,
        team2,
        city,
        format
      });

      setResult(res.data);
    } catch (err) {
      console.error("Prediction failed:", err);
      alert("Prediction failed. Backend might be waking up (Render sleep). Try again in 30 seconds.");
    } finally {
      setLoading(false);
    }
  };

  const chartData = result && {
    labels: [result.team1, team2],
    datasets: [{
      label: "Win Probability %",
      data: [result.probability, 100 - result.probability],
      backgroundColor: ["#22c55e", "#ef4444"]
    }]
  };

  return (
    <div className={`${dark ? "dark" : ""}`}>
      <div className="min-h-screen bg-gradient-to-br from-blue-900 to-indigo-800 dark:from-gray-900 dark:to-black text-white p-8">

        <div className="max-w-4xl mx-auto bg-white dark:bg-gray-800 text-black dark:text-white rounded-2xl shadow-2xl p-8">

          <div className="flex justify-between items-center mb-6">
            <h1 className="text-3xl font-bold">🏏 Cricket Match Predictor</h1>
            <button
              onClick={() => setDark(!dark)}
              className="bg-gray-200 dark:bg-gray-700 px-4 py-2 rounded-lg"
            >
              {dark ? "☀ Light" : "🌙 Dark"}
            </button>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Select label="Team 1" value={team1} set={setTeam1} options={teams} />
            <Select label="Team 2" value={team2} set={setTeam2} options={teams} />
            <Select label="City" value={city} set={setCity} options={cities} />
            <Select label="Format" value={format} set={setFormat} options={formats} />
          </div>

          <button
            onClick={predict}
            className="w-full mt-6 bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-xl transition"
          >
            Predict Match
          </button>

          {loading && (
            <div className="flex justify-center mt-6">
              <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-white"></div>
            </div>
          )}

          {result && (
            <div className="mt-10">

              <div className="flex justify-around items-center">
                <TeamCard team={result.team1} prob={result.probability} />
                <span className="text-2xl font-bold">VS</span>
                <TeamCard team={team2} prob={100 - result.probability} />
              </div>

              <div className="mt-8">
                <Bar data={chartData} />
              </div>

              {result.head_to_head && (
                <div className="mt-8 bg-gray-100 dark:bg-gray-700 p-4 rounded-xl">
                  <h2 className="font-bold text-lg mb-2">📈 Head to Head</h2>
                  <p>{result.team1}: {result.head_to_head.team1_wins} wins</p>
                  <p>{team2}: {result.head_to_head.team2_wins} wins</p>
                </div>
              )}

              {result.last5_team1 && (
                <div className="mt-8 bg-gray-100 dark:bg-gray-700 p-4 rounded-xl">
                  <h2 className="font-bold text-lg mb-2">📅 Last 5 Matches</h2>
                  <Last5 title={result.team1} data={result.last5_team1} />
                  <Last5 title={team2} data={result.last5_team2} />
                </div>
              )}

            </div>
          )}

        </div>
      </div>
    </div>
  );
}

function Select({ label, value, set, options }) {
  return (
    <div>
      <label className="block mb-1 font-semibold">{label}</label>
      <select
        value={value}
        onChange={e => set(e.target.value)}
        className="w-full p-2 rounded-lg border dark:bg-gray-700"
      >
        {options.map(o => (
          <option key={o}>{o}</option>
        ))}
      </select>
    </div>
  );
}

function TeamCard({ team, prob }) {
  return (
    <div className="text-center">
      {getFlag(team) && (
        <img src={getFlag(team)} alt="" className="mx-auto mb-2 w-16" />
      )}
      <h3 className="font-bold">{team}</h3>
      <h1 className="text-3xl font-bold">{prob}%</h1>
    </div>
  );
}

function Last5({ title, data }) {
  return (
    <div className="mb-4">
      <h3 className="font-semibold">{title}</h3>
      <div className="flex space-x-2 mt-2">
        {data.map((r, i) => (
          <div
            key={i}
            className={`w-8 h-8 flex items-center justify-center rounded-full text-white 
            ${r === 1 ? "bg-green-500" : "bg-red-500"}`}
          >
            {r === 1 ? "W" : "L"}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
# 🏏 IPL Player Price Prediction App

This project is a Streamlit-based web application that predicts the IPL player prices using Linear Regression.
It provides graphical comparisons between players and helps in identifying top-performing batsmen and bowlers based on their performance metrics.
The datasets used in this project were extracted from ESPN Cricinfo using BeautifulSoup, ensuring real-world and accurate IPL statistics.

---

## 📊 Features

- 🔐 Login System (Simple credential-based access)

- 🧮 Batting Price Prediction using:

    - Runs

    - Strike Rate

- 🎯 Bowling Price Prediction using:

    - Wickets

    - Economy Rate

- 📊 Comparison Dashboard:
    - Compare two top players (Batting or Bowling)

    - View insights through interactive bar charts

    - Automatic suggestion of the better pick based on predicted price and performance metrics 

---

## 🧩 Tech Stack

| Component | Technology Used |
|------------|-----------------|
| **Frontend / UI** | Streamlit |
| **Backend** | Python |
| **Data Extraction** | BeautifulSoup (Web Scraping) |
| **Machine Learning** | Scikit-learn (Linear Regression) |
| **Data Storage** | CSV |
| **Visualization** | Matplotlib / Plotly |

---

## 🧮 How It Works

1. **Data Extraction**  
   - Player stats are scraped from [ESPN Cricinfo](https://www.espncricinfo.com) using BeautifulSoup.  
   - Data is cleaned and stored in CSV format (`ipl_batting_records.csv`, `Bowling_records.csv`).

2. **Model Training**  
   - Linear Regression is used to predict performance based on past records (runs, averages, strike rate, economy, wickets, etc.).  
   - The trained model can be serialized using `joblib`.

3. **Prediction & Visualization**  
   - User selects between **Batting** or **Bowling** prediction.  
   - Top 10 players are shown with stats and images.  
   - Comparison graphs display performance insights.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Jeeva-5657/Cricket_Player_Price_Prediction.git
cd cricket-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📈 Future Enhancements

- ✅ Add All-Rounder Predictions
- ✅ Deploy to Streamlit Cloud or Hugging Face Spaces
- 🔜 Use more advanced ML algorithms (Random Forest, XGBoost)
- 🔜 Add live data updates from ESPN
- 🔜 Enhance UI with animations and charts

## 💡 Acknowledgments
- [ESPN Cricinfo](https://www.espncricinfo.com/) for player statistics
- [Scikit-learn](https://scikit-learn.org/stable/) for ML model training
- [Streamlit](https://streamlit.io/) for an easy web app framework

## 🌐 Connect
- Author : Jeeva Vadivel
- Email : jeevavadivel01@gmail.com
- Github : [Jeeva-5657](https://github.com/Jeeva-5657)
  

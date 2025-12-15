import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Simulated user credentials
USER_CREDENTIALS = {'username': 'admin', 'password': 'password123'}

# Function to check login credentials
def check_credentials(username, password):
    return username == USER_CREDENTIALS['username'] and password == USER_CREDENTIALS['password']

# Login page
def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if check_credentials(username, password):
            st.session_state['logged_in'] = True
            st.session_state['current_page'] = "prediction_options"
        else:
            st.error("Incorrect username or password")

# Prediction options page
def prediction_options_page():
    st.title("Prediction Options")
    option = st.selectbox("Select Prediction Type", ["Batting Prediction", "Bowling Prediction", "All-rounder Prediction"])

    # Navigation logic
    if st.button("Go"):
        if option == "Batting Prediction":
            st.session_state['current_page'] = "batting_prediction"
        elif option == "Bowling Prediction":
            st.session_state['current_page'] = "bowling_prediction"
        elif option == "All-rounder Prediction":
            st.write("All-rounder Prediction Page (Coming Soon)")

        # Redirect to the selected page
        # This will refresh the page and show the updated state

# Batting prediction page
def batting_prediction_page():
    st.title("Batting Prediction")

    # Load the dataset
    data = pd.read_csv("data/Batting_Records.csv") #Enter the path for batting prediction csv
    data.columns = data.columns.str.strip()  # Clean up column names

    if 'Players' not in data.columns or 'Image_URL' not in data.columns:
        st.error("The 'Players' or 'Image_URL' column is not found in the dataset.")
        return

    # Fill missing values in the dataset
    data.fillna(method='ffill', inplace=True)

    if 'SOLD PRICE' in data.columns:
        # Prepare features and target variable
        X = data[[
    'Matches',
    'Innings',
    'Runs',
    'AVG',
    'Strike_rate'
]]
        y = data['SOLD PRICE']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the model
        model = LinearRegression()
        model.fit(X_scaled, y)
        # Predict prices for all players
        data['Predicted Price'] = model.predict(X_scaled)

        # Get the top 10 batsmen based on predicted price
        top_10_batsmen = data.nlargest(10, 'Predicted Price')

        # Display full stats for the top 10 batsmen in a table
        st.write("Top 10 Batsmen Full Stats:")
        st.dataframe(top_10_batsmen)

        cols = st.columns(3)  # 3 images in a row
        for idx, (i, row) in enumerate(top_10_batsmen.iterrows()):
            with cols[idx % 3]:
                st.image(row['Image_URL'], caption=row['Players'], width=150)
                st.write(f"**Predicted Price**: ₹{row['Predicted Price']:.2f}")
                st.write(f"Runs: {row['Runs']}")
                st.write(f"Strike Rate: {row['Strike_rate']}\n")\

        # Add "Compare" button
        if st.button("Compare Top Batsmen"):
            st.session_state['current_page'] = "compare_batsmen"
            st.session_state['top_10_batsmen'] = top_10_batsmen

    else:
        st.error("The target column 'SOLD PRICE' is missing in the dataset.")

# Batsmen comparison page with graphical insights
def compare_batsmen_page():
    st.title("Compare Batsmen")

    # Retrieve the top 10 batsmen from the session state
    top_10_batsmen = st.session_state.get('top_10_batsmen')

    if top_10_batsmen is not None:
        # Allow user to select two batsmen to compare
        st.subheader("Select Two Batsmen to Compare")
        player1 = st.selectbox("Select First Batsman:", top_10_batsmen['Players'], index=0)
        player2 = st.selectbox("Select Second Batsman:", top_10_batsmen['Players'], index=1)

        # Retrieve data for the selected batsmen
        batsman1 = top_10_batsmen[top_10_batsmen['Players'] == player1].iloc[0]
        batsman2 = top_10_batsmen[top_10_batsmen['Players'] == player2].iloc[0]

        # Compare based on criteria (Predicted Price, Runs, Strike Rate, etc.)
        # Define a criteria for suggesting a pick (e.g., higher predicted price or a combination)
        if batsman1['Predicted Price'] > batsman2['Predicted Price']:
            suggested_pick = player1
        else:
            suggested_pick = player2

        # Display comparison with images and stats
        col1, col2 = st.columns(2)

        with col1:
            st.image(batsman1['Image_URL'], caption=player1, use_column_width=True)
            st.write(f"**{player1}**")
            st.write(f"Runs: {batsman1['Runs']}")
            st.write(f"Strike Rate: {batsman1['Strike_rate']}")
            st.write(f"Predicted Price: {batsman1['Predicted Price']:.2f}")
            if suggested_pick == player1:
                st.write("**Suggested Pick** 🏅")

        with col2:
            st.image(batsman2['Image_URL'], caption=player2, use_column_width=True)
            st.write(f"**{player2}**")
            st.write(f"Runs: {batsman2['Runs']}")
            st.write(f"Strike Rate: {batsman2['Strike_rate']}")
            st.write(f"Predicted Price: {batsman2['Predicted Price']:.2f}")
            if suggested_pick == player2:
                st.write("**Suggested Pick** 🏅")

        # Plot insights
        st.subheader("Performance Insights")

        # Bar Graph: Compare Runs and Strike Rate
        st.write("### Runs and Strike Rate Comparison")
        fig, ax = plt.subplots()
        metrics = ['Runs', 'Strike_rate']
        player1_values = [batsman1['Runs'], batsman1['Strike_rate']]
        player2_values = [batsman2['Runs'], batsman2['Strike_rate']]

        bar_width = 0.35
        index = np.arange(len(metrics))

        ax.bar(index, player1_values, bar_width, label=player1, color='blue')
        ax.bar(index + bar_width, player2_values, bar_width, label=player2, color='orange')

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Comparison of Runs and Strike Rate')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(metrics)
        ax.legend()

        st.pyplot(fig)

        # Bar Graph: Predicted Price Comparison
        st.write("### Predicted Price Comparison")
        fig, ax = plt.subplots()
        price_values = [batsman1['Predicted Price'], batsman2['Predicted Price']]

        ax.bar([player1, player2], price_values, color=['blue', 'orange'])
        ax.set_xlabel('Players')
        ax.set_ylabel('Predicted Price')
        ax.set_title('Predicted Price Comparison')

        st.pyplot(fig)

        # Insight: Suggest pick rationale
        st.write("### Suggestion Rationale")
        if suggested_pick == player1:
            st.write(f"**{player1}** is suggested as the better pick due to higher predicted price and performance metrics.")
        else:
            st.write(f"**{player2}** is suggested as the better pick due to higher predicted price and performance metrics.")

# Bowling prediction page
def bowling_prediction_page():
    st.title("Bowling Prediction")

    # Load the dataset
    data = pd.read_csv("/data/Bowling_records.csv") # Enter the path for the bowling data csv
    data.columns = data.columns.str.strip()  # Clean up column names

    if 'Players' not in data.columns or 'Image_URL' not in data.columns:
        st.error("The 'Players' or 'Image_URL' column is not found in the dataset.")
        return

    # Fill missing values in the data
    data.fillna(method='ffill', inplace=True)

    if 'SOLD PRICE' in data.columns:
        # Prepare features and target variable
        X = data[['Matches', 'Innings', 'Wickets', 'Economy_Rate','Bowling_Average']]
        y = data['SOLD PRICE']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Train the model
        model = LinearRegression()
        model.fit(X_scaled, y)
        # Predict prices for all players
        data['Predicted Price'] = model.predict(X_scaled)

        # Get the top 10 bowlers based on predicted price
        top_10_bowlers = data.nlargest(10, 'Predicted Price')

        # Display full stats for the top 10 bowlers in a table
        st.write("Top 10 Bowlers Full Stats:")
        st.dataframe(top_10_bowlers)

        # Display individual images and predicted prices for each bowler
        cols = st.columns(5)  # 3 images in a row
        for idx, (i, row) in enumerate(top_10_bowlers.iterrows()):
            with cols[idx % 5]:
                st.image(row['Image_URL'], caption=row['Players'], width=150)
                st.write(f"Wickets: {row['Wickets']}")
                st.write(f"Economy: {row['Economy_Rate']}")
  
        # Add "Compare" button and color red
        if st.button("Compare Top Bowlers"):
            st.session_state['current_page'] = "compare_bowlers"
            st.session_state['top_10_bowlers'] = top_10_bowlers

    else:
        st.error("The target column 'SOLD PRICE' is missing in the dataset.")

# Bowlers comparison page with graphical insights
def compare_bowlers_page():
    st.title("Compare Bowlers")

    # Retrieve the top 10 bowlers from the session state
    top_10_bowlers = st.session_state.get('top_10_bowlers')

    if top_10_bowlers is not None:
        # Allow user to select two bowlers to compare
        st.subheader("Select Two Bowlers to Compare")
        player1 = st.selectbox("Select First Bowler:", top_10_bowlers['Players'], index=0)
        player2 = st.selectbox("Select Second Bowler:", top_10_bowlers['Players'], index=1)

        # Retrieve data for the selected bowlers
        bowler1 = top_10_bowlers[top_10_bowlers['Players'] == player1].iloc[0]
        bowler2 = top_10_bowlers[top_10_bowlers['Players'] == player2].iloc[0]

        # Compare based on criteria (Predicted Price, Wickets, Economy Rate, etc.)
        # Define a criteria for suggesting a pick (e.g., higher predicted price or a combination)
        if bowler1['Predicted Price'] > bowler2['Predicted Price']:
            suggested_pick = player1
        else:
            suggested_pick = player2

        # Display comparison with images and stats
        col1, col2 = st.columns(2)

        with col1:
            st.image(bowler1['Image_URL'], caption=player1, use_column_width=True)
            st.write(f"**{player1}**")
            st.write(f"Wickets: {bowler1['Wickets']}")
            st.write(f"Economy Rate: {bowler1['Economy_Rate']}")
            st.write(f"Predicted Price: {bowler1['Predicted Price']:.2f}")
            if suggested_pick == player1:
                st.write("**Suggested Pick** 🏅")

        with col2:
            st.image(bowler2['Image_URL'], caption=player2, use_column_width=True)
            st.write(f"**{player2}**")
            st.write(f"Wickets: {bowler2['Wickets']}")
            st.write(f"Economy Rate: {bowler2['Economy_Rate']}")
            st.write(f"Predicted Price: {bowler2['Predicted Price']:.2f}")
            if suggested_pick == player2:
                st.write("**Suggested Pick** 🏅")

        # Plot insights
        st.subheader("Performance Insights")

        # Bar Graph: Compare Wickets and Economy Rate
        st.write("### Wickets and Economy Rate Comparison")
        fig, ax = plt.subplots()
        metrics = ['Wickets', 'Economy_Rate']
        player1_values = [bowler1['Wickets'], bowler1['Economy_Rate']]
        player2_values = [bowler2['Wickets'], bowler2['Economy_Rate']]

        bar_width = 0.35
        index = np.arange(len(metrics))

        ax.bar(index, player1_values, bar_width, label=player1, color='blue')
        ax.bar(index + bar_width, player2_values, bar_width, label=player2, color='orange')

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Comparison of Wickets and Economy Rate')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(metrics)
        ax.legend()

        st.pyplot(fig)

        # Bar Graph: Predicted Price Comparison
        st.write("### Predicted Price Comparison")
        fig, ax = plt.subplots()
        price_values = [bowler1['Predicted Price'], bowler2['Predicted Price']]

        ax.bar([player1, player2], price_values, color=['blue', 'orange'])
        ax.set_xlabel('Players')
        ax.set_ylabel('Predicted Price')
        ax.set_title('Predicted Price Comparison')

        st.pyplot(fig)

        # Insight: Suggest pick rationale
        st.write("### Suggestion Rationale")
        if suggested_pick == player1:
            st.write(f"**{player1}** is suggested as the better pick due to higher predicted price and performance metrics.")
        else:
            st.write(f"**{player2}** is suggested as the better pick due to higher predicted price and performance metrics.")

# Main function to control the app flow
def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "login"

    if not st.session_state['logged_in']:
        login_page()
    elif st.session_state['current_page'] == "prediction_options":
        prediction_options_page()
    elif st.session_state['current_page'] == "batting_prediction":
        batting_prediction_page()
    elif st.session_state['current_page'] == "bowling_prediction":
        bowling_prediction_page()
    elif st.session_state['current_page'] == "compare_batsmen":
        compare_batsmen_page()
    elif st.session_state['current_page'] == "compare_bowlers":
        compare_bowlers_page()

if __name__ == "__main__":
    main()

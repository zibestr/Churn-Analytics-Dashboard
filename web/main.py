import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

API_URL = os.getenv("API_URL", "http://api:8000/predict")
REQUIRED_COLS = ["customerID","gender","SeniorCitizen","Partner","Dependents",
                 "PhoneService","MultipleLines","InternetService","OnlineSecurity",
                 "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
                 "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
                 "MonthlyCharges","TotalCharges"]


def validate_data(data):
    return all(col in data.columns for col in REQUIRED_COLS)


def get_predictions(data, days):
    try:
        response = requests.post(
            API_URL,
            json={
                "data": data[REQUIRED_COLS].to_dict('records'),
                "days": days
            }
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def main():
    st.title("Customer Survival Analysis")

    uploaded_file = st.file_uploader("Upload customer data (CSV)", type=["csv"])
    if not uploaded_file:
        return

    data = pd.read_csv(uploaded_file)
    if not validate_data(data):
        st.error(f"Missing required columns. Needed: {', '.join(REQUIRED_COLS)}")
        return

    st.success(f"Loaded {len(data)} records")

    selected_day = st.slider("Select prediction tenure in month", 1, 72, 1)

    if st.button("Calculate Survival Probabilities"):
        with st.spinner("Processing..."):
            result = get_predictions(data, [selected_day])

        if result:
            df = pd.DataFrame([{
                "Customer ID": r["customerID"],
                f"Month {selected_day} Survival": r["probabilities"][str(selected_day)]
            } for r in result])

            st.dataframe(df.style.format({f"Month {selected_day} Survival": "{:.2%}"}))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            df[f"Month {selected_day} Survival"].plot.hist(ax=ax1, bins=20)
            ax1.set_title("Survival Probability Distribution")
            ax1.set_xlabel("Probability")
            ax1.set_ylabel("Customers")

            threshold = 0.5
            survived = (df[f"Month {selected_day} Survival"] >= threshold).sum()
            ax2.pie(
                [survived, len(df)-survived],
                labels=["Survived", "Churned"],
                autopct="%1.1f%%",
                colors=["#4CAF50", "#F44336"]
            )
            ax2.set_title(f"Churn Risk (Month {selected_day})")

            st.pyplot(fig)


if __name__ == "__main__":
    main()

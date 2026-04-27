import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://www.airnowapi.org/aq/data/"

def fetch_day(date, bbox):
    """
    Collect pollutant concentrations of PM 2.5, PM 10, O3, NO2, SO2, and CO for a given bounding box.
    """
    params = {
        "startDate": f"{date}T00",
        "endDate": f"{date}T23",
        "parameters": "PM25,PM10,O3,NO2,SO2,CO",
        "BBOX": bbox,
        "dataType": "C",
        "format": "application/json",
        "verbose": 0,
        "monitorType": 0,
        "includerawconcentrations": 1,
        "api_key": API_KEY
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print(f"Failed for {date}: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    return pd.DataFrame(data)


def collect_60_days(start_date="2026-02-22", days=60):
    """
    Collect hourly pollutant data for 60 days.
    """
    all_data = []

    bbox = "-97.75,30.25,-97.25,30.75"  # Austin, TX area

    start = datetime.strptime(start_date, "%Y-%m-%d")

    for i in range(days):
        date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        print(f"Fetching {date}...")

        df = fetch_day(date, bbox)

        if not df.empty:
            df["date"] = date
            all_data.append(df)

        # This line avoids rate limiting set by AirNow API
        time.sleep(1)

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)

if __name__ == "__main__":
    df = collect_60_days()
    output_path = "artifacts/hourly_pollution_60days.csv"
    df.to_csv(output_path, index=False)
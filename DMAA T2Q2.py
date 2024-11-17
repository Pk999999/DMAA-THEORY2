import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_file = "synthetic_crime_data.csv"  
crime_df = pd.read_csv(data_file)

crime_df['Date'] = pd.to_datetime(crime_df['Date'])
daily_crime = crime_df.groupby('Date').size().reset_index(name='Incident_Totals')
daily_crime.set_index('Date', inplace=True)

def compute_weighted_ma(series, window_size):
    weights = np.arange(1, window_size + 1)
    return series.rolling(window_size).apply(lambda values: np.dot(values, weights) / weights.sum(), raw=True)

def compute_hull_ma(series, window_size):
    half_window = window_size // 2
    sqrt_window = int(np.sqrt(window_size))
    wma_half = compute_weighted_ma(series, half_window)
    wma_full = compute_weighted_ma(series, window_size)
    return compute_weighted_ma(2 * wma_half - wma_full, sqrt_window)

daily_crime['SMA_30'] = daily_crime['Incident_Totals'].rolling(window=30).mean()
daily_crime['EMA_30'] = daily_crime['Incident_Totals'].ewm(span=30).mean()
daily_crime['WMA_30'] = compute_weighted_ma(daily_crime['Incident_Totals'], 30)
daily_crime['HMA_30'] = compute_hull_ma(daily_crime['Incident_Totals'], 30)

daily_crime['Short_MA'] = daily_crime['Incident_Totals'].rolling(window=10).mean()
daily_crime['Long_MA'] = daily_crime['Incident_Totals'].rolling(window=30).mean()

plt.figure(figsize=(14, 8))
plt.plot(daily_crime['Incident_Totals'], label="Incident Totals", alpha=0.6)
plt.plot(daily_crime['SMA_30'], label="Simple Moving Average (30)", linestyle='--')
plt.plot(daily_crime['EMA_30'], label="Exponential Moving Average (30)", linestyle='--')
plt.plot(daily_crime['WMA_30'], label="Weighted Moving Average (30)", linestyle='--')
plt.plot(daily_crime['HMA_30'], label="Hull Moving Average (30)", linestyle='--')
plt.title("Crime Incidents and Moving Averages\n(Analysis by Prithvi Kathuria 21BBS0158)")
plt.xlabel("Date")
plt.ylabel("Incident Totals")
plt.legend()
plt.show()

plt.figure(figsize=(14, 8))
plt.plot(daily_crime['Incident_Totals'], label="Incident Totals", alpha=0.6)
plt.plot(daily_crime['Short_MA'], label="Short-Term MA (10)", linestyle='--')
plt.plot(daily_crime['Long_MA'], label="Long-Term MA (30)", linestyle='--')
plt.title("Moving Average Crossover Analysis\n(Analysis by Prithvi Kathuria 21BBS0158)")
plt.xlabel("Date")
plt.ylabel("Incident Totals")
plt.legend()
plt.show()

output_file = "crime_analysis_with_moving_averages.csv"
daily_crime.to_csv(output_file, index=True)
print(f"Moving averages and crossover analysis saved to '{output_file}'.")
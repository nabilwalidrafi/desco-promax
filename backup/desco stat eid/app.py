import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# Load and preprocess data (cached)
@st.cache_data
def load_data():
    df = pd.read_excel('data0.xlsx')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    # Add hour_sin feature
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    # Add weekend feature (Friday and Saturday)
    df['weekend'] = df['datetime'].dt.dayofweek.isin([4, 5]).astype(int)  # 4=Friday, 5=Saturday
    # Add lag and rolling mean features
    df['demandMW_lag1'] = df['demandMW'].shift(1)
    df['demandMW_lag2'] = df['demandMW'].shift(2)
    df['demandMW_lag3'] = df['demandMW'].shift(3)
    df['demandMW_rollmean3'] = df['demandMW'].shift(1).rolling(window=3).mean()
    df = df.dropna()
    return df

df = load_data()

# Define Eid periods (Eid ul Fitr and Eid ul Adha) for 2023 and 2024
holiday_periods = {
    'eid_ul_fitr': [
        ('2023-04-20', '2023-04-26'), ('2024-04-08', '2024-04-14')
    ],
    'eid_ul_adha': [
        ('2023-06-28', '2023-07-04'), ('2024-06-16', '2024-06-22')
    ]
}

# Add Eid flags
def mark_eid_dates(df, periods):
    df['is_eid_ul_fitr'] = 0
    df['is_eid_ul_adha'] = 0
    for start, end in periods['eid_ul_fitr']:
        start = pd.to_datetime(start, utc=True).tz_localize(None)
        end = pd.to_datetime(end, utc=True).tz_localize(None)
        df.loc[(df['datetime'] >= start) & (df['datetime'] <= end), 'is_eid_ul_fitr'] = 1
    for start, end in periods['eid_ul_adha']:
        start = pd.to_datetime(start, utc=True).tz_localize(None)
        end = pd.to_datetime(end, utc=True).tz_localize(None)
        df.loc[(df['datetime'] >= start) & (df['datetime'] <= end), 'is_eid_ul_adha'] = 1
    return df

df = mark_eid_dates(df, holiday_periods)

# Calculate daily hourly means for Eid periods
def calculate_daily_hourly_eid_means(df):
    eid_fitr_df = df[df['is_eid_ul_fitr'] == 1].copy()
    eid_adha_df = df[df['is_eid_ul_adha'] == 1].copy()
    
    # Group by day and hour to get daily variability
    fitr_daily_hourly = eid_fitr_df.groupby([eid_fitr_df['datetime'].dt.date, eid_fitr_df['datetime'].dt.hour])['demandMW'].mean().unstack(fill_value=0)
    adha_daily_hourly = eid_adha_df.groupby([eid_adha_df['datetime'].dt.date, eid_adha_df['datetime'].dt.hour])['demandMW'].mean().unstack(fill_value=0)
    
    return fitr_daily_hourly, adha_daily_hourly

fitr_daily_hourly, adha_daily_hourly = calculate_daily_hourly_eid_means(df)

# Define training set
train_start = '2023-01-01'
train_end = '2025-03-10'
train_df = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()

# Update features to include weekend
features = ['demandMW_lag1', 'demandMW_lag2', 'demandMW_lag3', 'demandMW_rollmean3', 'hour_sin', 'weekend']
X_train = train_df[features]
y_train = train_df['demandMW']

# Train model (cached)
@st.cache_resource
def train_model():
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

rf_final = train_model()

st.title('DESCO Electricity Demand Predictor')

# User input for any date
user_date = st.date_input('Select a date for prediction')

# Function to check if a date falls within Eid periods (including 2025)
def is_eid_date(dt, periods):
    dt = pd.to_datetime(dt, utc=True).tz_localize(None)
    # Extend Eid periods to include 2025 for prediction
    extended_periods = {
        'eid_ul_fitr': periods['eid_ul_fitr'] + [('2025-03-29', '2025-04-04')],
        'eid_ul_adha': periods['eid_ul_adha'] + [('2025-06-05', '2025-06-14')]
    }
    for eid_type, eid_ranges in extended_periods.items():
        for start, end in eid_ranges:
            start = pd.to_datetime(start, utc=True).tz_localize(None)
            end = pd.to_datetime(end, utc=True).tz_localize(None)
            if start <= dt <= end:
                return eid_type
    return None

def get_day_offset(dt, eid_type, periods):
    dt = pd.to_datetime(dt, utc=True).tz_localize(None)
    extended_periods = {
        'eid_ul_fitr': periods['eid_ul_fitr'] + [('2025-03-29', '2025-04-04')],
        'eid_ul_adha': periods['eid_ul_adha'] + [('2025-06-05', '2025-06-14')]
    }
    for start, end in extended_periods[eid_type]:
        start = pd.to_datetime(start, utc=True).tz_localize(None)
        end = pd.to_datetime(end, utc=True).tz_localize(None)
        if start <= dt <= end:
            return (dt - start).days
    return None

if user_date:
    user_date = pd.Timestamp(user_date, tz='UTC').tz_localize(None)
    date_range = pd.date_range(start=user_date, end=user_date + pd.Timedelta(hours=23), freq='H', tz='UTC').tz_localize(None)
    
    last_known_date = df['datetime'].max()
    if last_known_date < user_date:
        pred_range = pd.date_range(start=last_known_date + pd.Timedelta(hours=1), 
                                  end=user_date - pd.Timedelta(hours=1), freq='H', tz='UTC').tz_localize(None)
        feature_data = df[df['datetime'] <= last_known_date].tail(3).copy()
        
        if len(feature_data) < 3:
            st.warning("Not enough historical data to make predictions.")
        else:
            for dt in pred_range:
                hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                weekend = 1 if dt.dayofweek in [4, 5] else 0
                demand_lag1 = feature_data['demandMW'].iloc[-1]
                demand_lag2 = feature_data['demandMW'].iloc[-2]
                demand_lag3 = feature_data['demandMW'].iloc[-3]
                roll_mean = feature_data['demandMW'].tail(3).mean()
                
                eid_type = is_eid_date(dt, holiday_periods)
                if eid_type == 'eid_ul_fitr':
                    day_offset = get_day_offset(dt, eid_type, holiday_periods)
                    if day_offset is not None and day_offset < len(fitr_daily_hourly):
                        pred = fitr_daily_hourly.iloc[day_offset % len(fitr_daily_hourly)].get(dt.hour, demand_lag1)
                        adjusted_lag1 = demand_lag1 * 0.2 + pred * 0.8 if demand_lag1 > pred else demand_lag1
                    else:
                        pred = demand_lag1
                        adjusted_lag1 = demand_lag1
                elif eid_type == 'eid_ul_adha':
                    day_offset = get_day_offset(dt, eid_type, holiday_periods)
                    if day_offset is not None and day_offset < len(adha_daily_hourly):
                        pred = adha_daily_hourly.iloc[day_offset % len(adha_daily_hourly)].get(dt.hour, demand_lag1)
                        adjusted_lag1 = demand_lag1 * 0.2 + pred * 0.8 if demand_lag1 > pred else demand_lag1
                    else:
                        pred = demand_lag1
                        adjusted_lag1 = demand_lag1
                else:
                    features_array = np.array([[demand_lag1, demand_lag2, demand_lag3, roll_mean, hour_sin, weekend]])
                    pred = rf_final.predict(features_array)[0]
                    adjusted_lag1 = demand_lag1
                
                new_row = pd.DataFrame({
                    'datetime': [dt],
                    'demandMW': [pred],
                    'hour_sin': [hour_sin],
                    'weekend': [weekend],
                    'demandMW_lag1': [adjusted_lag1],
                    'demandMW_lag2': [demand_lag2],
                    'demandMW_lag3': [demand_lag3],
                    'demandMW_rollmean3': [roll_mean]
                })
                feature_data = pd.concat([feature_data, new_row], ignore_index=True)
                feature_data = feature_data.tail(3)
    else:
        feature_data = df[df['datetime'] < user_date].tail(3).copy()
        if len(feature_data) < 3:
            st.warning("Not enough historical data to make predictions for the selected date.")
            feature_data = None
    
    if feature_data is not None:
        predictions = []
        for dt in date_range:
            hour_sin = np.sin(2 * np.pi * dt.hour / 24)
            weekend = 1 if dt.dayofweek in [4, 5] else 0
            demand_lag1 = feature_data['demandMW'].iloc[-1]
            demand_lag2 = feature_data['demandMW'].iloc[-2]
            demand_lag3 = feature_data['demandMW'].iloc[-3]
            roll_mean = feature_data['demandMW'].tail(3).mean()
            
            eid_type = is_eid_date(dt, holiday_periods)
            if eid_type == 'eid_ul_fitr':
                day_offset = get_day_offset(dt, eid_type, holiday_periods)
                if day_offset is not None and day_offset < len(fitr_daily_hourly):
                    pred = fitr_daily_hourly.iloc[day_offset % len(fitr_daily_hourly)].get(dt.hour, demand_lag1)
                    adjusted_lag1 = demand_lag1 * 0.2 + pred * 0.8 if demand_lag1 > pred else demand_lag1
                else:
                    pred = demand_lag1
                    adjusted_lag1 = demand_lag1
            elif eid_type == 'eid_ul_adha':
                day_offset = get_day_offset(dt, eid_type, holiday_periods)
                if day_offset is not None and day_offset < len(adha_daily_hourly):
                    pred = adha_daily_hourly.iloc[day_offset % len(adha_daily_hourly)].get(dt.hour, demand_lag1)
                    adjusted_lag1 = demand_lag1 * 0.2 + pred * 0.8 if demand_lag1 > pred else demand_lag1
                else:
                    pred = demand_lag1
                    adjusted_lag1 = demand_lag1
            else:
                features_array = np.array([[demand_lag1, demand_lag2, demand_lag3, roll_mean, hour_sin, weekend]])
                pred = rf_final.predict(features_array)[0]
                adjusted_lag1 = demand_lag1
            
            predictions.append(pred)
            
            new_row = pd.DataFrame({
                'datetime': [dt],
                'demandMW': [pred],
                'hour_sin': [hour_sin],
                'weekend': [weekend],
                'demandMW_lag1': [adjusted_lag1],
                'demandMW_lag2': [demand_lag2],
                'demandMW_lag3': [demand_lag3],
                'demandMW_rollmean3': [roll_mean]
            })
            feature_data = pd.concat([feature_data, new_row], ignore_index=True)
            feature_data = feature_data.tail(3)
        
        # Create results DataFrame for table
        result_df = pd.DataFrame({
            'Hour': date_range,
            'Predicted Demand (MW)': predictions
        }).set_index('Hour')
        
        # Display table
        st.subheader(f'Predicted Demand for {user_date.date()}')
        st.dataframe(result_df.style.format({
            'Predicted Demand (MW)': '{:.2f}'
        }))
        
        # Plot predictions
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(date_range, predictions, label='Predicted Demand (MW)', marker='o', color='#1f77b4')
        ax.set_title(f'Predicted Demand for {user_date.date()}')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Demand (MW)')
        ax.legend()
        ax.grid(True)

        # Set major tick locator to show ticks every 3 hours
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        # Format the major ticks to display only the hour
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.xticks(rotation=45)
        st.pyplot(fig)
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
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['demandMW_lag1'] = df['demandMW'].shift(1)
    df['demandMW_lag2'] = df['demandMW'].shift(2)
    df['demandMW_lag3'] = df['demandMW'].shift(3)
    df['demandMW_rollmean3'] = df['demandMW'].shift(1).rolling(window=3).mean()
    df = df.dropna()
    return df

df = load_data()

# Define Eid periods (Eid ul Fitr and Eid ul Adha)
holiday_periods = {
    'eid_ul_fitr': [
        ('2023-04-20', '2023-04-26'), ('2024-04-08', '2024-04-14'), ('2025-03-29', '2025-04-04'),
        ('2026-03-18', '2026-03-24'), ('2027-03-07', '2027-03-13'), ('2028-02-24', '2028-03-01'),
        ('2029-02-13', '2029-02-19'), ('2030-02-01', '2030-02-07')
    ],
    'eid_ul_adha': [
        ('2023-06-28', '2023-07-04'), ('2024-06-16', '2024-06-22'), ('2025-06-05', '2025-06-14'),
        ('2026-05-25', '2026-05-31'), ('2027-05-14', '2027-05-20'), ('2028-05-02', '2028-05-08'),
        ('2029-04-21', '2029-04-27'), ('2030-04-10', '2030-04-16')
    ]
}

# Add Eid features
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

# Define training set
train_start = '2023-01-01'
train_end = '2025-03-10'
train_df = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()

features = ['demandMW_lag1', 'demandMW_lag2', 'demandMW_lag3', 'demandMW_rollmean3', 'hour_sin']

# Train models (cached)
@st.cache_resource
def train_model(data, eid_type=None):
    if eid_type:
        data = data[data[f'is_eid_ul_{eid_type}'] == 1]
    X_train = data[features]
    y_train = data['demandMW']
    rf = RandomForestRegressor(n_estimators=350, random_state=42)
    rf.fit(X_train, y_train)
    return rf

rf_general = train_model(train_df)  # General model for non-Eid
rf_eid_fitr = train_model(train_df, 'fitr')  # Eid ul Fitr model
rf_eid_adha = train_model(train_df, 'adha')  # Eid ul Adha model

st.title('DESCO Electricity Demand Predictor (Eid-Optimized)')

# User input for any date
user_date = st.date_input('Select a date for prediction')

def is_eid_date(dt, periods):
    dt = pd.to_datetime(dt, utc=True).tz_localize(None)
    for eid_type, eid_ranges in periods.items():
        for start, end in eid_ranges:
            start = pd.to_datetime(start, utc=True).tz_localize(None)
            end = pd.to_datetime(end, utc=True).tz_localize(None)
            if start <= dt <= end:
                return eid_type
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
            eid_fitr_mean = train_df[train_df['is_eid_ul_fitr'] == 1]['demandMW'].mean()
            eid_adha_mean = train_df[train_df['is_eid_ul_adha'] == 1]['demandMW'].mean()
            
            for dt in pred_range:
                hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                demand_lag1 = feature_data['demandMW'].iloc[-1]
                demand_lag2 = feature_data['demandMW'].iloc[-2]
                demand_lag3 = feature_data['demandMW'].iloc[-3]
                roll_mean = feature_data['demandMW'].tail(3).mean()
                
                features_array = np.array([[demand_lag1, demand_lag2, demand_lag3, roll_mean, hour_sin]])
                eid_type = is_eid_date(dt, holiday_periods)
                if eid_type == 'eid_ul_fitr':
                    pred = rf_eid_fitr.predict(features_array)[0]
                    adjusted_lag1 = demand_lag1 * 0.2 + eid_fitr_mean * 0.8 if demand_lag1 > eid_fitr_mean else demand_lag1
                elif eid_type == 'eid_ul_adha':
                    pred = rf_eid_adha.predict(features_array)[0]
                    adjusted_lag1 = demand_lag1 * 0.2 + eid_adha_mean * 0.8 if demand_lag1 > eid_adha_mean else demand_lag1
                else:
                    pred = rf_general.predict(features_array)[0]
                    adjusted_lag1 = demand_lag1
                
                days_to_fitr = (pd.to_datetime(holiday_periods['eid_ul_fitr'][2][0], utc=True).tz_localize(None) - dt).days if is_eid_date(dt, {'eid_ul_fitr': holiday_periods['eid_ul_fitr']}) else None
                if days_to_fitr and 0 < days_to_fitr <= 2:
                    adjusted_lag1 = demand_lag1 * 0.3 + eid_fitr_mean * 0.7 if demand_lag1 > eid_fitr_mean else demand_lag1
                days_to_adha = (pd.to_datetime(holiday_periods['eid_ul_adha'][2][0], utc=True).tz_localize(None) - dt).days if is_eid_date(dt, {'eid_ul_adha': holiday_periods['eid_ul_adha']}) else None
                if days_to_adha and 0 < days_to_adha <= 2:
                    adjusted_lag1 = demand_lag1 * 0.3 + eid_adha_mean * 0.7 if demand_lag1 > eid_adha_mean else demand_lag1
                
                new_row = pd.DataFrame({
                    'datetime': [dt],
                    'demandMW': [pred],
                    'hour_sin': [hour_sin],
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
        eid_fitr_mean = train_df[train_df['is_eid_ul_fitr'] == 1]['demandMW'].mean()
        eid_adha_mean = train_df[train_df['is_eid_ul_adha'] == 1]['demandMW'].mean()
        
        for dt in date_range:
            hour_sin = np.sin(2 * np.pi * dt.hour / 24)
            demand_lag1 = feature_data['demandMW'].iloc[-1]
            demand_lag2 = feature_data['demandMW'].iloc[-2]
            demand_lag3 = feature_data['demandMW'].iloc[-3]
            roll_mean = feature_data['demandMW'].tail(3).mean()
            
            features_array = np.array([[demand_lag1, demand_lag2, demand_lag3, roll_mean, hour_sin]])
            eid_type = is_eid_date(dt, holiday_periods)
            if eid_type == 'eid_ul_fitr':
                pred = rf_eid_fitr.predict(features_array)[0]
                adjusted_lag1 = demand_lag1 * 0.2 + eid_fitr_mean * 0.8 if demand_lag1 > eid_fitr_mean else demand_lag1
            elif eid_type == 'eid_ul_adha':
                pred = rf_eid_adha.predict(features_array)[0]
                adjusted_lag1 = demand_lag1 * 0.2 + eid_adha_mean * 0.8 if demand_lag1 > eid_adha_mean else demand_lag1
            else:
                pred = rf_general.predict(features_array)[0]
                adjusted_lag1 = demand_lag1
            
            predictions.append(pred)
            
            new_row = pd.DataFrame({
                'datetime': [dt],
                'demandMW': [pred],
                'hour_sin': [hour_sin],
                'demandMW_lag1': [adjusted_lag1],
                'demandMW_lag2': [demand_lag2],
                'demandMW_lag3': [demand_lag3],
                'demandMW_rollmean3': [roll_mean]
            })
            feature_data = pd.concat([feature_data, new_row], ignore_index=True)
            feature_data = feature_data.tail(3)
        
        result_df = pd.DataFrame({
            'Hour': date_range,
            'Predicted Demand (MW)': predictions
        }).set_index('Hour')
        
        st.subheader(f'Predicted Demand for {user_date.date()}')
        st.dataframe(result_df.style.format({
            'Predicted Demand (MW)': '{:.2f}'
        }))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(date_range, predictions, label='Predicted Demand (MW)', marker='o', color='#1f77b4')
        ax.set_title(f'Predicted Demand for {user_date.date()}')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Demand (MW)')
        ax.legend()
        ax.grid(True)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter(''))
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader('Debug Info')
        for dt, pred in zip(date_range, predictions):
            eid_type = is_eid_date(dt, holiday_periods)
            model_used = 'Eid ul Fitr' if eid_type == 'eid_ul_fitr' else 'Eid ul Adha' if eid_type == 'eid_ul_adha' else 'General'
            st.write(f"Hour: {dt.hour}, Predicted: {pred:.2f}, Model: {model_used}")
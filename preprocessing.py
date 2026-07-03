"""
=============================================================================
MODULE 1: DATA PREPROCESSING
Project: AI Road Accident Prevention Intelligence System
=============================================================================
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_dataset(n=100000, seed=42):
    np.random.seed(seed)

    weather_opts = ['Clear', 'Rainy', 'Foggy', 'Stormy', 'Snowy']
    road_opts = ['Good', 'Wet', 'Damaged', 'Icy', 'Under Construction']
    time_opts = ['Morning', 'Afternoon', 'Evening', 'Night']

    # YOUR FULL COUNTRY LIST
    country_opts = [
        'USA','UK','Canada','India','China','Japan','Russia',
        'Brazil','Germany','Australia'
    ]

    vehicle_opts = ['Good','Fair','Poor']

    weather = np.random.choice(weather_opts,n,p=[0.40,0.25,0.15,0.12,0.08])
    road_condition = np.random.choice(road_opts,n,p=[0.35,0.25,0.20,0.12,0.08])
    time_of_day = np.random.choice(time_opts,n,p=[0.25,0.30,0.25,0.20])
    country = np.random.choice(country_opts,n)
    vehicle_cond = np.random.choice(vehicle_opts,n,p=[0.5,0.3,0.2])

    speed = np.clip(np.random.normal(65,25,n),0,180).astype(int)
    fatigue = np.random.randint(0,11,n)
    traffic_density = np.random.randint(1,11,n)
    distraction = np.random.randint(0,11,n)
    visibility = np.clip(np.random.normal(7,2.5,n),0,10).round(1)
    mobile_usage = np.random.choice([0,1],n,p=[0.65,0.35])

    # FIXED DATE SHUFFLE
    dates = pd.date_range('2020-01-01', periods=n, freq='5min')
    dates = np.array(dates, copy=True)
    np.random.shuffle(dates)

    # Risk maps
    weather_risk = {'Clear':1,'Rainy':3,'Foggy':4,'Stormy':5,'Snowy':5}
    road_risk = {'Good':1,'Wet':3,'Damaged':4,'Icy':5,'Under Construction':4}
    vehicle_risk = {'Good':1,'Fair':2,'Poor':4}
    time_risk = {'Morning':2,'Afternoon':2,'Evening':3,'Night':5}

    w = np.array([weather_risk[x] for x in weather])
    r = np.array([road_risk[x] for x in road_condition])
    v = np.array([vehicle_risk[x] for x in vehicle_cond])
    t = np.array([time_risk[x] for x in time_of_day])

    base_risk = (
        0.25*(speed/180*10)+
        0.20*fatigue+
        0.15*w*2+
        0.15*traffic_density+
        0.10*r*2+
        0.10*distraction+
        0.05*(10-visibility)
    )

    noise = np.random.normal(0,0.5,n)

    accident_risk = np.clip(base_risk+noise,0,10).round(2)

    severity = pd.cut(
        accident_risk,
        bins=[-np.inf,3,5,7,np.inf],
        labels=['Low','Medium','High','Critical']
    )

    df = pd.DataFrame({
        'Date':dates,
        'Country':country,
        'Speed':speed,
        'Fatigue':fatigue,
        'Weather':weather,
        'TrafficDensity':traffic_density,
        'RoadCondition':road_condition,
        'Distraction':distraction,
        'Visibility':visibility,
        'TimeOfDay':time_of_day,
        'VehicleCondition':vehicle_cond,
        'MobileUsage':mobile_usage,
        'AccidentRisk':accident_risk,
        'Severity':severity
    })

    # Add nulls
    for col in ['Speed','Fatigue','Visibility','TrafficDensity']:
        mask = np.random.rand(n) < 0.02
        df.loc[mask,col] = np.nan

    # Add duplicates
    dup = df.sample(frac=0.005,random_state=1)
    df = pd.concat([df,dup],ignore_index=True)

    return df


# ============================================================================
# LOAD
# ============================================================================

def load_dataset(path='road_accident_data.csv'):
    try:
        df = pd.read_csv(path, parse_dates=['Date'])
        print("Dataset loaded")
    except:
        print("Generating dataset...")
        df = generate_dataset()
        df.to_csv(path,index=False)

    return df


# ============================================================================
# CLEANING
# ============================================================================

def remove_duplicates(df):
    return df.drop_duplicates()

def handle_nulls(df):
    for col in df.select_dtypes(include=np.number):
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object'):
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def feature_engineering(df):

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek']>=5).astype(int)

    df['SpeedFatigueInteraction'] = (
        df['Speed']/180 * df['Fatigue']/10
    )

    df['VisibilityRisk'] = 1-(df['Visibility']/10)

    df['BehaviorRiskScore'] = (
        df['Distraction']+df['Fatigue']+df['MobileUsage']*5
    )/3

    df['EnvironmentRiskScore'] = (
        df['TrafficDensity']+df['VisibilityRisk']*10
    )/2

    return df


# ============================================================================
# ENCODING
# ============================================================================

CATEGORICAL_COLS = [
    'Weather',
    'RoadCondition',
    'TimeOfDay',
    'Country',
    'VehicleCondition'
]

def encode_categoricals(df):

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col+'_enc'] = le.fit_transform(df[col])

    return df


# ============================================================================
# MASTER
# ============================================================================

def run_preprocessing():

    print("="*60)
    print("AI ROAD ACCIDENT PREPROCESSING")
    print("="*60)

    df = load_dataset()

    df = remove_duplicates(df)
    df = handle_nulls(df)
    df = feature_engineering(df)
    df = encode_categoricals(df)

    df.to_csv("processed_data.csv",index=False)

    print("Completed")
    print(df.shape)

    return df


if __name__ == "__main__":
    run_preprocessing()

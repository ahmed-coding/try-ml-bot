import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, roc_auc_score
from binance.client import Client
import time
import joblib

def fetch_data(client, symbol, interval='1m', years=5, start_time=None):
    """
    Fetch historical candlestick data from Binance.
    """
    candles_per_day = 1440 // int(interval[:-1])
    days_in_year = 365
    total_candles = candles_per_day * days_in_year * years if not start_time else 1000

    limit = 1000
    all_data = []
    current_time = int(time.time() * 1000)
    if not start_time:
        start_time = current_time - (years * 365 * 24 * 60 * 60 * 1000)

    while total_candles > 0:
        candles = client.futures_klines(symbol=symbol, interval=interval, limit=limit, startTime=start_time)
        if not candles:
            break

        all_data.extend(candles)
        start_time = candles[-1][6]
        total_candles -= limit
        time.sleep(0.2)

    df = pd.DataFrame(all_data, columns=[
        'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_Time', 'Quote_Asset_Volume', 'Number_Of_Trades',
        'Taker_Buy_Base_Asset_Volume', 'Taker_Buy_Quote_Asset_Volume', 'Ignore'
    ])
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['Quote_Asset_Volume'] = df['Quote_Asset_Volume'].astype(float)
    df['Taker_Buy_Base_Asset_Volume'] = df['Taker_Buy_Base_Asset_Volume'].astype(float)
    df['Taker_Buy_Quote_Asset_Volume'] = df['Taker_Buy_Quote_Asset_Volume'].astype(float)
    df['Number_Of_Trades'] = df['Number_Of_Trades'].astype(int)
    df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms')
    df['Change_Percentage'] = ((df['High'] - df['Open']) / df['Open']) * 100
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume','Quote_Asset_Volume','Taker_Buy_Quote_Asset_Volume','Number_Of_Trades', 'Change_Percentage']]


def identify_zones(df, threshold=0.002):
    """Identify supply and demand zones based on price movements."""
    zones = []
    for i in range(1, len(df) - 1):
        if abs(df['High'][i] - df['Low'][i]) / df['Low'][i] > threshold:
            zones.append({
                'type': 'supply' if df['Close'][i] < df['Open'][i] else 'demand',
                'price': df['Close'][i]
            })
    return zones

def find_fvg(df):
    """Calculate Fair Value Gaps (FVG)."""
    fvg = []
    for i in range(1, len(df) - 1):
        if df['High'][i - 1] < df['Low'][i + 1]:
            fvg.append({
                'start': df['High'][i - 1],
                'end': df['Low'][i + 1],
                'midpoint': (df['High'][i - 1] + df['Low'][i + 1]) / 2
            })
    return fvg

def detect_market_structure(df):
    """Identify Market Structure Break (MSB)."""
    structure = {'highs': [], 'lows': []}
    for i in range(1, len(df) - 1):
        if df['High'][i] > df['High'][i - 1] and df['High'][i] > df['High'][i + 1]:
            structure['highs'].append(df['High'][i])
        if df['Low'][i] < df['Low'][i - 1] and df['Low'][i] < df['Low'][i + 1]:
            structure['lows'].append(df['Low'][i])
    return structure

api_key = ''
api_secret = ''
client = Client(api_key=api_key, api_secret=api_secret)

# symbol = 'BTCUSDT'
symbol = 'ETHUSDT'
interval = '5m'
years = 1
df = fetch_data(client, symbol, interval=interval, years=years)

zones = identify_zones(df)
fvg_zones = find_fvg(df)
market_structure = detect_market_structure(df)

# df['Supply_Zone'] = [zone['price'] if zone['type'] == 'supply' else None for zone in zones]
# df['Demand_Zone'] = [zone['price'] if zone['type'] == 'demand' else None for zone in zones]
# df['FVG_Midpoint'] = [fvg['midpoint'] for fvg in fvg_zones] if fvg_zones else None


# Add Supply and Demand Zones as NaN for rows with no zones
df['Supply_Zone'] = [zone['price'] if zone['type'] == 'supply' else None for zone in zones] + [None] * (len(df) - len(zones))
df['Demand_Zone'] = [zone['price'] if zone['type'] == 'demand' else None for zone in zones] + [None] * (len(df) - len(zones))

# Add FVG Midpoint as NaN for rows with no zones
df['FVG_Midpoint'] = [fvg['midpoint'] for fvg in fvg_zones] + [None] * (len(df) - len(fvg_zones))


# Now fill missing values (NaN) where appropriate
df['Supply_Zone'].fillna(method='ffill', inplace=True)
df['Demand_Zone'].fillna(method='ffill', inplace=True)
df['FVG_Midpoint'].fillna(method='ffill', inplace=True)


df['Next_Close'] = df['Close'].shift(-1)

X = df[['Open', 'High', 'Low', 'Close', 'Volume','Quote_Asset_Volume','Taker_Buy_Quote_Asset_Volume','Number_Of_Trades', 'Change_Percentage', 'Supply_Zone', 'Demand_Zone', 'FVG_Midpoint']]
y = df['Next_Close']

X.fillna(method='ffill', inplace=True)
X.fillna(0, inplace=True)
y = y.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [100, 300, 500, 1000],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'lambda': [1, 1.5, 2],
    'alpha': [0, 0.5, 1],
}


grid_search = GridSearchCV(
    xgb.XGBRegressor(objective='reg:squarederror'),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred_prob = best_model.predict_proba(X_test)[:, 1]

y_pred = (y_pred_prob >= 0.5).astype(int)

# تقييم الأداء
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("دقة النموذج:", accuracy)
print("AUC:", roc_auc)
print("\nتقرير التصنيف:\n", classification_report(
    y_test, y_pred, zero_division=1))




predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

model_path = f'{symbol}_{interval}_ict_smc_xgboost_model.pkl'
joblib.dump(best_model, model_path)
print(f"Model saved at {model_path}")

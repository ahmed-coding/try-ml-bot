import joblib
from binance.client import Client
import pandas as pd
import joblib





symbol = 'NEIROUSDT'
interval = '1h'
api_key = ''
api_secret = ''

# model_path = f'{symbol}_{interval}_xgboost_model.pkl'

model_path = f'neirousdt_xgboost_price_model_with_gridsearch.pkl'

client = Client(api_key=api_key, api_secret=api_secret)


candles = client.futures_klines(symbol=symbol, interval=interval, limit=5)

# إنشاء DataFrame
df = pd.DataFrame(candles, columns=[
    'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close_Time', 'Quote_Asset_Volume', 'Number_Of_Trades',
    'Taker_Buy_Base_Asset_Volume', 'Taker_Buy_Quote_Asset_Volume', 'Ignore'
])
df['Open'] = df['Open'].astype(float)
df['High'] = df['High'].astype(float)
df['Low'] = df['Low'].astype(float)
df['Close'] = df['Close'].astype(float)
df['Volume'] = df['Volume'].astype(float)
df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms')

# حساب نسبة التغيير لكل شمعة

df['Change_Percentage'] = ((df['High'] - df['Open']) / df['Open']) * 100

# print(df.iloc[-1:])

new_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1:]
# new_data = df[['Open', 'High', 'Low', 'Close', 'Volume']]


loaded_model = joblib.load(model_path)

new_predictions = loaded_model.predict(new_data)

print(new_predictions)
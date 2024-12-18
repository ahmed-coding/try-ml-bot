import xgboost as xgb
import pandas as pd

from binance.client import Client
import time
import joblib


# إعداد مفتاح API الخاص بـ Binance
api_key = ''
api_secret = ''

# دالة لجلب البيانات التاريخية


# إعداد API Binance
client = Client(api_key=api_key, api_secret=api_secret)

# جلب البيانات
symbol = 'NEIROUSDT'
interval = '5m'
years = 2
limit = 432
look_ahead=1
# candles = client.futures_klines(symbol=symbol, interval='5m', limit=432)


# إضافة الهدف (Target) لتصنيف الشموع
# تقسيم البيانات إلى تدريب واختبار

# حفظ النموذج المدرب
# model_path = f'{symbol}_{interval}_xgboost_model.pkl'
model_path = f'neirousdt_xgboost_price_model_with_gridsearch.pkl'

# joblib.dump(best_model, model_path)
# print(f"تم حفظ النموذج في {model_path}")

# دالة لتحميل النموذج وإعادة استخدامه

def fetch_data(client, symbol, interval='1m', limit=432):
    
    candles = client.futures_klines(symbol=symbol, interval=interval, limit=limit)

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
    df['Change_Percentage'] = ((df['High'] - df['Open']) / df['Open']) * 100

    return df[['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change_Percentage']]



def load_model(path):
    """تحميل النموذج المحفوظ."""
    return joblib.load(path)



def incremental_learning(new_data, model_path):
    """تحديث النموذج باستخدام بيانات جديدة."""
    # تحميل النموذج الحالي
    model = load_model(model_path)

    # تقسيم الميزات والمخرجات
    # X_new = new_data[['Open', 'High', 'Low', 'Close', 'Volume','Change_Percentage']]
    X_new = new_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y_new = new_data['Future_Close']

    # تحويل البيانات إلى DMatrix
    dtrain_new = xgb.DMatrix(X_new, label=y_new)

    # تحديث النموذج
    model.fit(X_new, y_new, xgb_model=model_path)

    # حفظ النموذج المحدّث
    joblib.dump(model, model_path)
    print("تم تحديث النموذج وحفظه.")


new_data = fetch_data(client=client,symbol=symbol,limit=limit)

print(len(new_data))
new_data['Future_Close'] = new_data['Close'].shift(-look_ahead)  # السعر بعد 5 شمعات

new_data = new_data.dropna(subset=['Future_Close'])




incremental_learning(new_data,model_path)
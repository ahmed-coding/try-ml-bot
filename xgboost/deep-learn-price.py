import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
from binance.client import Client
import time
from sklearn.model_selection import train_test_split


# دالة لجلب البيانات التاريخية 
def fetch_data(client, symbol, interval='1m', years=5, min_percentage=1, start_time=None):
    candles_per_day = 288  # عدد الشموع اليومية
    days_in_year = 365
    total_candles = candles_per_day * days_in_year * years if not start_time else 1000

    limit = 1000
    all_data = []
    current_time = int(time.time() * 1000)  # الوقت الحالي بالمللي ثانية
    if not start_time:
        start_time = current_time - (years * 365 * 24 * 60 * 60 * 1000)

    while total_candles > 0:
        candles = client.futures_klines(symbol=symbol, interval=interval, limit=limit, startTime=start_time)

        if not candles:
            break  # إذا لم تكن هناك بيانات إضافية

        all_data.extend(candles)
        start_time = candles[-1][6]  # وقت إغلاق آخر شمعة
        total_candles -= limit
        time.sleep(0.2)  # تأخير بين الطلبات

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
    df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms')

    df['Change_Percentage'] = ((df['High'] - df['Open']) / df['Open']) * 100
    # filtered_df = df[df['Change_Percentage'] >= min_percentage]

    return df[['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change_Percentage']]

# إعداد API Binance
client = Client(api_key='your_api_key', api_secret='your_api_secret')

# جلب البيانات
symbol = 'NEIROUSDT'
interval = '5m'
years = 2
min_percentage = 0.5
# إعداد الهدف للتنبؤ بسعر الإغلاق في الشمعة القادمة (مثل التنبؤ بالإغلاق بعد 5 شمعات)
look_ahead = 1


# param_grid = {
#     'max_depth': [7],
#     'learning_rate': [0.3],
#     'n_estimators': [1000],
#     'subsample': [0.8],
#     'colsample_bytree': [0.8],
#     'lambda': [1],
#     'alpha': [0.5],
#     'gamma': [0, 0.1, 0.2],
#     'min_child_weight': [1, 2, 3]
#     # 'eval_metric': 'logloss',
# }

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 300, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'lambda': [1, 1.5, 2],
    'alpha': [0, 0.5, 1],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 2, 3]
}




df = fetch_data(client, symbol, interval=interval, years=years, min_percentage=min_percentage)




# ---------------------------- التنبؤ بالسعر القادم ------------------------------
# # تحديد الهدف وهو السعر الذي نريد التنبؤ به (سعر الإغلاق للشمعة التالية)
# df['Next_Close'] = df['Close'].shift(-1)  # تحديد سعر الإغلاق للشمعة التالية كهدف

# # التخلص من الصف الأخير لأنه لا يحتوي على قيمة للتنبؤ
# df = df.dropna()

# # تقسيم الميزات والمخرجات
# X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
# y = df['Next_Close']  # الهدف هو سعر الإغلاق للشمعة التالية

# # إعداد معلمات GridSearchCV


# # إنشاء نموذج XGBoost
# xg_reg = xgb.XGBRegressor(objective='reg:squarederror')

# # إعداد GridSearchCV
# grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# # تدريب النموذج باستخدام GridSearchCV
# grid_search.fit(X, y)

# # طباعة أفضل المعلمات
# print(f"أفضل معلمات: {grid_search.best_params_}")

# # حفظ النموذج الأفضل
# best_model = grid_search.best_estimator_
# joblib.dump(best_model, 'neirousdt_xgboost_price_model_with_gridsearch.pkl')

# # اختبار النموذج الأفضل
# new_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1:]  # آخر سطر من البيانات
# new_prediction = best_model.predict(new_data)

# print(f"التنبؤ بسعر الإغلاق للشمعة التالية: {new_prediction[0]}")


# ---------------------------- التنبؤ بالسعر القادم لعدد محدد من الشمعات ------------------------------





# إضافة عمود للتنبؤ بالسعر بعد 5 شمعات (التنبؤ بسعر الإغلاق)
df['Future_Close'] = df['Close'].shift(-look_ahead)  # السعر بعد 5 شمعات

# إزالة الصفوف التي تحتوي على قيم NaN في العمود الجديد
df = df.dropna(subset=['Future_Close'])

# تقسيم البيانات إلى ميزات (X) وأهداف (y)
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # يمكنك إضافة ميزات أخرى حسب الحاجة
y = df['Future_Close']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# تدريب نموذج XGBoost

# إنشاء نموذج XGBoost
xg_reg = xgb.XGBRegressor(objective='reg:squarederror')

# إعداد GridSearchCV
grid_search = GridSearchCV(
    estimator=xg_reg,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring='neg_mean_squared_error'
    )

# تدريب النموذج باستخدام GridSearchCV
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'neirousdt_xgboost_price_model_with_gridsearch.pkl')


# التنبؤ بالقيم المستقبلية (سعر الإغلاق بعد 5 شمعات)
# # اختبار النموذج الأفضل
new_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1:]  # آخر سطر من البيانات
new_prediction = best_model.predict(new_data)
# عرض النتائج
# for i in range(10):
print(f"الشمعة: التنبؤ بسعر الإغلاق بعد 5 شمعات = {new_prediction}, السعر الفعلي = {y_test.iloc[-1]:.2f}")
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from binance.client import Client
import time
import joblib
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


# إعداد مفتاح API الخاص بـ Binance
api_key = ''
api_secret = ''

# دالة لجلب البيانات التاريخية
def fetch_data(client, symbol, interval='1m', years=5, min_percentage=1, start_time=None):
    """
    جلب بيانات الشموع التاريخية (1m) لمدة معينة من Binance مع فلترة بناءً على نسبة الارتفاع.
    """

    # حساب عدد الشموع المطلوبة
    candles_per_day = 96  # عدد الشموع اليومية
    days_in_year = 365
    total_candles = candles_per_day * days_in_year * years if not start_time else 1000

    # حدود Binance: 1000 شمعة لكل طلب
    limit = 1000

    # قائمة لتخزين البيانات
    all_data = []

    # وقت البداية
    current_time = int(time.time() * 1000)  # الوقت الحالي بالمللي ثانية
    if not start_time:
        start_time = current_time - (years * 365 * 24 * 60 * 60 * 1000)

    while total_candles > 0:
        # جلب البيانات من Binance
        candles = client.futures_klines(
            symbol=symbol, interval=interval, limit=limit, startTime=start_time
        )

        if not candles:
            break  # إذا لم تكن هناك بيانات إضافية

        # إضافة البيانات إلى القائمة
        all_data.extend(candles)

        # تحديث وقت البداية للنطاق التالي
        start_time = candles[-1][6]  # وقت إغلاق آخر شمعة

        # تقليل عدد الشموع المتبقية
        total_candles -= limit

        # التأخير لتجنب حظر Binance
        time.sleep(0.2)  # تأخير بين الطلبات

    # إنشاء DataFrame
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

    # حساب نسبة التغيير لكل شمعة
    df['Change_Percentage'] = ((df['High'] - df['Open']) / df['Open']) * 100

    # تصفية الشموع بناءً على النسبة المدخلة
    filtered_df = df[df['Change_Percentage'] >= min_percentage]

    return filtered_df[['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change_Percentage']]

# إعداد API Binance
client = Client(api_key=api_key, api_secret=api_secret)

# جلب البيانات
symbol = 'NEIROUSDT'
interval = '15m'
years = 2
min_percentage = 0.5
df = fetch_data(client, symbol, interval=interval, years=years, min_percentage=min_percentage)

# إضافة الهدف (Target) لتصنيف الشموع
threshold = 1  # الهدف: ارتفاع أكبر من 1%
df['Target'] = (df['Change_Percentage'] > threshold).astype(int)

# تقسيم الميزات والمخرجات
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Target']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ضبط المعلمات باستخدام GridSearchCV

dtrain = xgb.DMatrix(X_train, label=y_train)




param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 300, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'lambda': [1, 1.5, 2],
    'alpha': [0, 0.5, 1],
}

best_params = None
best_score = float('inf')

# for max_depth in param_grid['max_depth']:
#     for eta in param_grid['eta']:
#         for subsample in param_grid['subsample']:
#             for colsample_bytree in param_grid['colsample_bytree']:
#                 params = {
#                     'objective': 'binary:logistic',
#                     'max_depth': max_depth,
#                     'eta': eta,
#                     'subsample': subsample,
#                     'colsample_bytree': colsample_bytree,
#                     'eval_metric': 'logloss'
#                 }
#                 cv_results = xgb.cv(params, dtrain, num_boost_round=100, nfold=3, early_stopping_rounds=10)
#                 mean_score = cv_results['test-logloss-mean'].min()
#                 if mean_score < best_score:
#                     best_score = mean_score
#                     best_params = params


grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss', 
    ),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1
)


# model.set_params(**best_params) 
grid_search.fit(X_train, y_train)

# طباعة أفضل معلمات
# print("أفضل معلمات:", model.best_params_)




# البحث عن أفضل معلمات


print("أفضل معلمات:", best_params)


# تقييم النموذج النهائي
best_model = grid_search.best_estimator_
best_model = grid_search
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

# تقييم الأداء
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("دقة النموذج:", accuracy)
print("AUC:", roc_auc)
print("\nتقرير التصنيف:\n", classification_report(y_test, y_pred))

# حفظ النموذج المدرب
model_path = 'best_xgboost_model.pkl'
joblib.dump(best_model, model_path)
print(f"تم حفظ النموذج في {model_path}")

# دالة لتحميل النموذج وإعادة استخدامه
def load_model(path):
    """تحميل النموذج المحفوظ."""
    return joblib.load(path)

# مثال على استخدام النموذج المحفوظ
# loaded_model = load_model(model_path)
# new_predictions = loaded_model.predict(X_test)
# print("نتائج باستخدام النموذج المحمل:", new_predictions)

# تحسين النموذج باستخدام بيانات جديدة
def incremental_learning(new_data, model_path):
    """تحديث النموذج باستخدام بيانات جديدة."""
    # تحميل النموذج الحالي
    model = load_model(model_path)

    # تقسيم الميزات والمخرجات
    X_new = new_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y_new = new_data['Target']

    # تحويل البيانات إلى DMatrix
    dtrain_new = xgb.DMatrix(X_new, label=y_new)

    # تحديث النموذج
    model.fit(X_new, y_new, xgb_model=model_path)

    # حفظ النموذج المحدّث
    joblib.dump(model, model_path)
    print("تم تحديث النموذج وحفظه.")

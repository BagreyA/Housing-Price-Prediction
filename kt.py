import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings

# Отключение предупреждений
warnings.filterwarnings("ignore")

# Установка стиля графиков
sns.set_theme(style="whitegrid")

# Загрузка данных
print("\nЗагружаем обучающую и тестовую выборки...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Просмотр первых строк таблицы
print("\nПервые строки обучающей выборки:")
print(train.head())

# Информация о признаках и типах данных
print("\nИнформация о признаках и типах данных:")
print(train.info())

# Статистики числовых признаков
print("\nСтатистики числовых признаков:")
print(train.describe())

# Определение числовых и категориальных признаков
numeric_feats = train.select_dtypes(include=[np.number]).columns
cat_feats = train.select_dtypes(include=['object']).columns

# Ограничение количества отображаемых признаков
max_plots = 12
num_feats_to_plot = numeric_feats[:max_plots]
cat_feats_to_plot = cat_feats[:max_plots]

# Объединение train и test для унифицированного анализа
all_df = pd.concat([train, test], ignore_index=True)

# Визуализация гистограмм числовых признаков
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
fig.suptitle('Гистограммы числовых признаков', fontsize=16)
numeric_feats = train.select_dtypes(include=[np.number]).columns[:12]
for ax, feature in zip(axes.flatten(), numeric_feats):
    sns.histplot(all_df[feature], kde=True, bins=30, ax=ax)
    ax.set_title(feature)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Boxplot для оценки выбросов и распределения
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
fig.suptitle('Boxplot числовых признаков', fontsize=16)
for ax, feature in zip(axes.flatten(), num_feats_to_plot):
    sns.boxplot(x=all_df[feature], ax=ax)
    ax.set_title(feature)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Countplot для категориальных признаков
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
fig.suptitle('Распределение категориальных признаков', fontsize=16)
for ax, feature in zip(axes.flatten(), cat_feats_to_plot):
    sns.countplot(data=all_df, x=feature, ax=ax)
    ax.set_title(feature)
    ax.tick_params(axis='x', rotation=90)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Логарифмирование целевой переменной для уменьшения скошенности
y = np.log1p(train['SalePrice'])
train.drop(['SalePrice'], axis=1, inplace=True)
n_train = train.shape[0]

# Распределение логарифма целевой переменной
plt.figure(figsize=(8, 5))
sns.histplot(y, bins=30, kde=True)
plt.title('Распределение логарифма цены продажи')
plt.xlabel('log(Цена продажи)')
plt.show()

# Анализ пропущенных значений
print("\nТоп-10 признаков с наибольшей долей пропущенных значений:")
missing_frac = train.isnull().mean().sort_values(ascending=False)
top_missing = missing_frac[missing_frac > 0].head(10)
print(top_missing)

# График пропущенных значений
plt.figure(figsize=(8, 5))
top_missing.plot.bar()
plt.title('Наиболее неполные признаки')
plt.ylabel('Доля пропусков')
plt.show()

# Корреляционный анализ с целевой переменной
print("\nТоп-10 признаков с наибольшей корреляцией с лог(ценой):")
numeric_feats = train.select_dtypes(include=[np.number]).columns
corr_matrix = pd.concat([train, y.rename('SalePrice')], axis=1)[numeric_feats.tolist() + ['SalePrice']].corr()
top_corr = corr_matrix['SalePrice'].abs().sort_values(ascending=False).head(10)
print(top_corr)

# Тепловая карта сильных корреляций
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix.loc[top_corr.index, top_corr.index], annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица сильных признаков')
plt.show()

# Повторное объединение данных после логарифмирования
print("\nОбъединяем обучающую и тестовую выборки для единых преобразований...")
all_df = pd.concat([train, test], ignore_index=True)

# Заполнение пропущенных значений
cat_cols = all_df.select_dtypes(include=['object']).columns
num_cols = all_df.select_dtypes(exclude=['object']).columns

print("Заполняем пропущенные категориальные значения значением 'None'")
cat_imputer = SimpleImputer(strategy='constant', fill_value='None')
all_df[cat_cols] = cat_imputer.fit_transform(all_df[cat_cols])

print("Заполняем пропущенные числовые значения медианой")
num_imputer = SimpleImputer(strategy='median')
all_df[num_cols] = num_imputer.fit_transform(all_df[num_cols])

# Создание новых признаков на основе существующих
print("Создаём новые признаки: TotalBath и TotalSF")
all_df['TotalBath'] = (all_df['FullBath'] + 0.5 * all_df['HalfBath']
                       + all_df['BsmtFullBath'] + 0.5 * all_df['BsmtHalfBath'])
all_df['TotalSF'] = (all_df['TotalBsmtSF'] + all_df['1stFlrSF'] + all_df['2ndFlrSF'])

# Логарифмирование скошенных признаков
print("Корректируем асимметрию признаков логарифмированием...")
skew_vals = all_df[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_feats = skew_vals[abs(skew_vals) > 0.75].index
for feat in skewed_feats:
    all_df[feat] = np.log1p(all_df[feat])

# One-hot кодирование категориальных признаков
print("Преобразуем категориальные признаки в числовые с помощью One-Hot кодирования...")
all_df = pd.get_dummies(all_df, drop_first=True)

# Делим обратно на train и test
X = all_df.iloc[:n_train, :]
X_test = all_df.iloc[n_train:, :]

# Делим train на тренировочную и валидационную выборки
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Функция для вычисления RMSE
def rmse(model, X, y):
    preds = model.predict(X)
    return np.sqrt(mean_squared_error(y, preds))

# Обучение моделей и оценка на валидации
print("\nОбучаем модели и оцениваем RMSE на валидации...")

models = {
    'Ridge': RidgeCV(alphas=np.logspace(-3, 2, 50), cv=5),
    'Lasso': LassoCV(alphas=np.logspace(-4, 1, 60), cv=5),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.7, colsample_bytree=0.7, random_state=42
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=32,
        feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5,
        min_data_in_leaf=20, verbose=-1, random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    rmse_train = rmse(model, X_train, y_train)
    rmse_valid = rmse(model, X_valid, y_valid)
    print(f"{name} - RMSE на трейне: {rmse_train:.4f} | RMSE на валидации: {rmse_valid:.4f}")

# Обучение лучшей модели на всех данных и генерация прогноза
print("\nОбучаем лучшую модель (Ridge) на всех данных и создаём прогноз...")
best_model = RidgeCV(alphas=np.logspace(-3, 2, 50), cv=5)
best_model.fit(X, y)
final_preds = best_model.predict(X_test)

# Обратное преобразование log1p и сохранение результатов
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': np.expm1(final_preds)
})
submission.to_csv('submission.csv', index=False)
print("Файл отправки сохранён как 'submission.csv'.")

# Визуализация важности признаков (LightGBM)
print("\nВизуализируем важность признаков по модели LightGBM...")
# Извлекаем обученную модель
lgb_model = models['LightGBM']

# Получаем значения важности признаков
importances = lgb_model.feature_importances_
feature_names = X_train.columns

# Создаем DataFrame для визуализации
importance_df = pd.DataFrame({
    'Признак': feature_names,
    'Важность': importances
}).sort_values(by='Важность', ascending=False)

# Выбираем топ-N признаков
top_n = 20
top_features_df = importance_df.head(top_n)

plt.figure(figsize=(10, 8))
sns.barplot(x='Важность', y='Признак', data=top_features_df, palette='viridis')
plt.title(f'Топ {top_n} признаков по важности (LightGBM)')
plt.tight_layout()
plt.show()

# Визуализация важности признаков (XGBoost)
print("\nВизуализируем важность признаков по модели XGBoost...")
xgb_model = models['XGBoost']
importances = xgb_model.feature_importances_
feature_names = X_train.columns

# Создаем DataFrame для визуализации
importance_df = pd.DataFrame({
    'Признак': feature_names,
    'Важность': importances
}).sort_values(by='Важность', ascending=False)

top_n = 20
top_features_df = importance_df.head(top_n)

plt.figure(figsize=(10, 8))
sns.barplot(x='Важность', y='Признак', data=top_features_df, palette='viridis')
plt.title(f'Топ {top_n} признаков по важности (XGBoost)')
plt.xlabel('Оценка важности')
plt.ylabel('Признак')
plt.tight_layout()
plt.show()

# Сравнение распределения признаков в train и test
train_copy = X.copy()
test_copy = X_test.copy()
train_copy['Тип'] = 'Обучающая выборка'
test_copy['Тип'] = 'Тестовая выборка'

# Объединим для визуализации
merged = pd.concat([train_copy, test_copy], axis=0)

# Визуализация распределения признаков
features_to_plot = ['GrLivArea', 'TotalSF']

for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=merged, x=feature, hue='Тип', kde=True, bins=40, element="step", stat="density", common_norm=False)
    plt.title(f'Сравнение распределения признака "{feature}" в train и test')
    plt.xlabel(feature)
    plt.ylabel('Плотность')
    plt.tight_layout()
    plt.show()

for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged, x='Тип', y=feature)
    plt.title(f'Boxplot сравнение признака "{feature}" между Train и Test')
    plt.xlabel('Набор данных')
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

# Визуализация предсказаний vs фактических значений
preds_valid = best_model.predict(X_valid)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=np.expm1(y_valid), y=np.expm1(preds_valid), alpha=0.6)
plt.xlabel("Фактическая цена")
plt.ylabel("Предсказанная цена")
plt.plot([0, max(np.expm1(y_valid))], [0, max(np.expm1(y_valid))], color='red', linestyle='--')
plt.title("Фактическая vs Предсказанная цена")
plt.show()

# Анализ остатков
residuals = y_valid - preds_valid
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Распределение остатков (Residuals)")
plt.xlabel("Остаток")
plt.show()

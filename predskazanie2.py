import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasetitog.csv', sep=';', encoding='utf-8')

# Приводим названия колонок к нижнему регистру, заменяем пробелы на _
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Чистка названий городов
df['city'] = df['city'].astype(str).str.strip().str.replace('г.', '', regex=False).str.strip()
df['city'] = df['city'].str.replace('г ', '', regex=False).str.strip()

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

# Статистика до обработки
total_cities_before = df['city'].nunique()
total_rows_before = len(df)
cities_2024_before = df[df['year'] == 2024]['city'].nunique()

print("СТАТИСТИКА ДО ОБРАБОТКИ")
print(f"Всего городов в исходном датасете: {total_cities_before}")
print(f"Всего строк в исходном датасете: {total_rows_before}")
print(f"Городов с данными за 2024: {cities_2024_before}")

# Список признаков
feature_columns = [
    'people', 'air_general_level', 'net_salary', 'avg_age', 'ilm', 'birth',
    'crimes', 'criminals', 'death', 'pens', 'preschool_child', 'length_of_roads',
    'energy_consumption', 'migration_net', 'housing_stock', 'wage', 'workers',
    'poverty_level', 'housing_price'
]

# Признаки, которые не могут быть нулевыми в реальности
non_zero_features = [
    'people', 'net_salary', 'wage', 'workers', 'housing_price',
    'length_of_roads', 'energy_consumption', 'ilm', 'birth',
    'crimes', 'criminals', 'death', 'pens', 'preschool_child',
    'air_general_level', 'avg_age', 'housing_stock', 'poverty_level'
]

def clean_feature(df, col):
    df[col] = df[col].astype(str).str.replace(' ', '').str.replace(',', '.').str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].replace(-9.0, np.nan)
    if col in non_zero_features:
        df[col] = df[col].replace(0, np.nan)
    return df

for col in feature_columns:
    if col in df.columns:
        df = clean_feature(df, col)

df = df.drop_duplicates(subset=['city', 'year'])

print(f"\nГородов после удаления дубликатов: {df['city'].nunique()}")


# Интерполяция и заполнение пропусков

def interpolate_and_fill(df, feature_name):
    df_temp = df[['city', 'year', feature_name]].copy().sort_values(['city', 'year'])
    df_temp[feature_name] = df_temp.groupby('city')[feature_name].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    def fill_median(group):
        if group.notna().any():
            if feature_name in non_zero_features:
                pos_vals = group[group > 0]
                if len(pos_vals) > 0:
                    return group.fillna(pos_vals.median())
                else:
                    return group
            else:
                return group.fillna(group.median())
        else:
            return group
    df_temp[feature_name] = df_temp.groupby('city')[feature_name].transform(fill_median)
    df_temp[feature_name] = df_temp.groupby('city')[feature_name].ffill()
    if feature_name in non_zero_features:
        global_vals = df_temp[feature_name][df_temp[feature_name] > 0]
        global_median = global_vals.median() if len(global_vals) > 0 else np.nan
        if np.isnan(global_median):
            min_pos = df_temp[feature_name][df_temp[feature_name] > 0].min()
            global_median = min_pos if not np.isnan(min_pos) else 1.0
    else:
        global_median = df_temp[feature_name].median()
        if np.isnan(global_median):
            global_median = 0
    df_temp[feature_name] = df_temp[feature_name].fillna(global_median)
    return df_temp[['city', 'year', feature_name]]

df_full = None
for col in feature_columns:
    if col in df.columns:
        feat_df = interpolate_and_fill(df, col)
        if df_full is None:
            df_full = feat_df
        else:
            df_full = df_full.merge(feat_df, on=['city', 'year'], how='outer')

# Замена оставшихся нулей на минимальное положительное значение
for col in non_zero_features:
    if col in df_full.columns:
        positive_vals = df_full[col][df_full[col] > 0]
        if len(positive_vals) > 0:
            min_pos = positive_vals.min()
            df_full[col] = df_full[col].replace(0, min_pos)
        else:
            df_full[col] = df_full[col].replace(0, 1.0)

# Сохраняем датасет после интерполяции
df_full.to_csv('after_interpolation.csv', index=False, encoding='utf-8-sig')
print("\nСохранён after_interpolation.csv (после интерполяции, исходные единицы)")

# Нормализация

scalers = {}
norm_cols = []
for col in feature_columns:
    if df_full[col].notna().sum() == 0:
        print(f"Предупреждение: {col} – все NaN, пропускаем")
        continue
    scaler = MinMaxScaler()
    norm_name = f'{col}_norm'
    df_full[norm_name] = scaler.fit_transform(df_full[[col]])
    scalers[col] = scaler
    norm_cols.append(norm_name)

# Веса признаков (позитивные / негативные) - сумма весов = 1

pos_weights = {
    'net_salary_norm': 0.10,
    'ilm_norm': 0.04,
    'birth_norm': 0.02,
    'preschool_child_norm': 0.02,
    'length_of_roads_norm': 0.04,
    'migration_net_norm': 0.02,
    'housing_stock_norm': 0.04,
    'wage_norm': 0.04,
    'workers_norm': 0.02,
    'housing_price_norm': 0.02,
}

neg_weights = {
    'people_norm': 0.10,
    'air_general_level_norm': 0.16,
    'avg_age_norm': 0.04,
    'crimes_norm': 0.08,
    'criminals_norm': 0.04,
    'death_norm': 0.04,
    'pens_norm': 0.02,
    'energy_consumption_norm': 0.08,
    'poverty_level_norm': 0.08,
}

# Расчёт raw_score (диапазон 0...1)
raw_score = pd.Series(0.0, index=df_full.index)
for col, w in pos_weights.items():
    if col in df_full.columns:
        raw_score += w * df_full[col]
for col, w in neg_weights.items():
    if col in df_full.columns:
        raw_score += w * (1 - df_full[col])

# env_score в диапазоне 0...100
df_full['env_score'] = raw_score * 100

# Создание лагов
df_full = df_full.sort_values(['city', 'year'])
for col in norm_cols + ['env_score']:
    df_full[f'lag_{col}'] = df_full.groupby('city')[col].shift(1)

lag_cols = [f'lag_{col}' for col in norm_cols + ['env_score']]
df_lagged = df_full.dropna(subset=lag_cols)   # удаляем строки без лага (первые годы)

# Сохраняем датасет после лагов (нормализованные признаки и лаги, до разбиения)
df_lagged.to_csv('after_lags.csv', index=False, encoding='utf-8-sig')
print("Сохранён after_lags.csv (нормализованные признаки, env_score и их лаги)")

# Разбиение 80% / 20% (случайное)
X = df_lagged[lag_cols]
y = df_lagged['env_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=842, shuffle=True
)

print(f"\nРазмер обучающей выборки: {len(X_train)} строк")
print(f"Размер тестовой выборки: {len(X_test)} строк")

# Обучение моделей и оценка качества
print("\nЛИНЕЙНАЯ РЕГРЕССИЯ")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"R² train: {r2_train:.4f}")
print(f"R² test:  {r2_test:.4f}")
print(f"RMSE train: {rmse_train:.4f}")
print(f"RMSE test:  {rmse_test:.4f}")

print("\nRANDOM FOREST")
rf = RandomForestRegressor(n_estimators=100, random_state=842, n_jobs=-1)
rf.fit(X_train, y_train)
y_train_pred_rf = rf.predict(X_train)
y_test_pred_rf = rf.predict(X_test)

r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
rmse_train_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))

print(f"R² train: {r2_train_rf:.4f}")
print(f"R² test:  {r2_test_rf:.4f}")
print(f"RMSE train: {rmse_train_rf:.4f}")
print(f"RMSE test:  {rmse_test_rf:.4f}")

# Итоговая статистика по обработанному датасету

final_cities = df_full['city'].nunique()
final_rows = len(df_full)
final_cities_2024 = df_full[df_full['year'] == 2024]['city'].nunique()

print("\n=== ИТОГОВАЯ СТАТИСТИКА ДАТАСЕТА (после интерполяции) ===")
print(f"Всего городов в обработанном датасете: {final_cities}")
print(f"Всего строк в обработанном датасете: {final_rows}")
print(f"Городов с данными за 2024: {final_cities_2024}")

# ТАБЛИЦА ФАКТ 2024 И ПРОГНОЗ 2025

last_year = df_full[df_full['year'] == 2024].copy()
if not last_year.empty:
    X_2025 = last_year[norm_cols + ['env_score']].rename(
        columns={c: f'lag_{c}' for c in norm_cols + ['env_score']}
    )
    last_year['pred_LR'] = lr.predict(X_2025)
    last_year['pred_RF'] = rf.predict(X_2025)

    # Формируем таблицу: город, год (2024), все признаки, прогнозы LR и RF
    cols_to_show = ['city', 'year'] + feature_columns + ['pred_LR', 'pred_RF']
    # Добавляем также текущий env_score за 2024 (рассчитанный по данным 2024)
    cols_to_show.insert(2, 'env_score')
    print("\n=== ФАКТИЧЕСКИЕ ДАННЫЕ ЗА 2024 И ПРОГНОЗ ENV_SCORE НА 2025 (первые 10 городов) ===")
    print(last_year[cols_to_show].head(10).to_string(index=False))

# Прогноз на 2025 год
last_year = df_full[df_full['year'] == 2024].copy()
if not last_year.empty:
    X_2025 = last_year[norm_cols + ['env_score']].rename(
        columns={c: f'lag_{c}' for c in norm_cols + ['env_score']}
    )
    last_year['pred_LR'] = lr.predict(X_2025)
    last_year['pred_RF'] = rf.predict(X_2025)
    print("\n=== ПРОГНОЗ НА 2025 (первые 5 городов) ===")
    print(last_year[['city', 'pred_LR', 'pred_RF']].head(5).to_string(index=False))
    # Сохраняем прогноз в CSV
    forecast_2025 = last_year[['city', 'year', 'pred_LR', 'pred_RF'] + feature_columns]
    forecast_2025.to_csv('forecast_2025.csv', index=False, encoding='utf-8-sig')
    print("\nСохранён прогноз на 2025: forecast_2025.csv")
else:
    print("\nНет данных за 2024 год для прогноза на 2025.")

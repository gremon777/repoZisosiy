import pandas as pd
import os

# --- НАСТРОЙКИ ---
CONFIG = {
    "1": {"name": "Экология", "file": "air.csv", "neg": ['emissions', 'выброс', 'index', 'co_', 'so_']},
    "2": {"name": "Транспорт", "file": "transport.csv", "neg": ['убыток', 'износ', 'дтп', 'время']},
    "3": {"name": "Экономика", "file": "economy.csv", "neg": ['безработица', 'инфляция', 'бедные']}
}


def get_data(file_name):
    """Загружает файл и находит ключевые колонки."""
    if not os.path.exists(file_name):
        return None, None, f"Файл {file_name} не найден."

    # Загрузка с авто-определением разделителя
    try:
        df = pd.read_csv(file_name, sep=None, engine='python', encoding='utf-8')
    except:
        df = pd.read_csv(file_name, sep=None, engine='python', encoding='windows-1251')

    df.columns = df.columns.str.strip()

    # Авто-поиск названий колонок (Город и Год)
    obj_col = next((c for c in df.columns if any(w in c.lower() for w in ['city', 'object', 'регион'])), "city")
    year_col = next((c for c in df.columns if any(w in c.lower() for w in ['year', 'год'])), "year")


    # Финальная чистка чисел и мусора
    for col in df.columns:
        if col not in [obj_col, year_col]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

    df = df[~df[obj_col].astype(str).str.contains('Федерация|округ', case=False, na=False)].dropna(subset=[year_col])
    return df, (obj_col, year_col), None


def calculate_score(df, meta, neg_list):
    """Считает индекс 0-100 на основе профиля."""
    obj_col, year_col = meta
    # Берем только числовые колонки (метрики)
    metrics = [c for c in df.select_dtypes(include=['number']).columns if c not in [year_col] and 'ok' not in c.lower()]

    for col in metrics:
        v_min, v_max = df[col].min(), df[col].max()
        if v_max == v_min: continue

        # Инверсия: если колонка "плохая", то чем меньше значение, тем выше балл
        is_neg = any(w in col.lower() for w in neg_list)
        if is_neg:
            df[f'n_{col}'] = (v_max - df[col]) / (v_max - v_min)
        else:
            df[f'n_{col}'] = (df[col] - v_min) / (v_max - v_min)

    # Итоговый балл — среднее по нормализованным колонкам
    df['Score'] = (df[[c for c in df.columns if c.startswith('n_')]].mean(axis=1) * 100).round(2)
    return df


# --- ИНТЕРФЕЙС ---
while True:
    print(f"\n{'=' * 40}\n  АНАЛИТИЧЕСКИЙ ЦЕНТР\n{'=' * 40}")
    for k, v in CONFIG.items(): print(f"{k}. {v['name']}")
    print("0. Выход")

    mode = input("\nВыберите раздел: ")
    if mode == "0": break
    if mode not in CONFIG: continue

    conf = CONFIG[mode]
    df, meta, err = get_data(conf['file'])

    if err:
        print(f"Ошибка: {err}")
        continue

    df = calculate_score(df, meta, conf['neg'])
    obj_col, year_col = meta

    print("\n1. Топ-10 за год | 2. Динамика города")
    sub_mode = input("Действие: ")

    if sub_mode == "1":
        y = int(input(f"Введите год (доступно до {int(df[year_col].max())}): ") or df[year_col].max())
        res = df[df[year_col] == y].groupby(obj_col)['Score'].mean().reset_index()
        print(f"\n--- РЕЙТИНГ ЗА {y} ГОД ---")
        print(res.sort_values('Score', ascending=False).head(10).to_string(index=False))

    else:
        city = input("Введите город: ").strip().upper()
        res = df[df[obj_col].astype(str).str.upper().str.contains(city, na=False)].groupby(year_col)[
            'Score'].mean().reset_index()

        if res.empty:
            print("Город не найден.")
        else:
            print(f"\n--- ДИНАМИКА: {city} ---")
            print(res.to_string(index=False))

            # Математический итог
            diff = round(res.iloc[-1]['Score'] - res.iloc[0]['Score'], 2)
            print(f"\n{'#' * 40}\nВЕРДИКТ: {'ПРОГРЕСС' if diff > 0 else 'РЕГРЕСС'} на {abs(diff)} б.\n{'#' * 40}")
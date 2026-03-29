import requests

city = input("Введите название города: ")

# 1. Получаем координаты
url_geo = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&language=ru&format=json"
res_geo = requests.get(url_geo).json()

if "results" not in res_geo:
    print("Ошибка: город не найден или допущена опечатка.")
    exit()

lat = res_geo["results"][0]["latitude"]
lon = res_geo["results"][0]["longitude"]
name = res_geo["results"][0]["name"]

print(f"\nДанные для города: {name} (Широта: {lat}, Долгота: {lon})")
print("-" * 40)

# 2. Получаем прогноз погоды
url_weather = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&timezone=auto"
res_weather = requests.get(url_weather).json()

print("Прогноз температуры на ближайшие 5 часов:")
for i in range(5):
    time = res_weather['hourly']['time'][i].replace("T", " ")
    temp = res_weather['hourly']['temperature_2m'][i]
    print(f"Время: {time} | Температура: {temp}°C")

print("-" * 40)

# 3. Получаем данные по экологии
url_eco = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=european_aqi,pm10,pm2_5&timezone=auto"
res_eco = requests.get(url_eco).json()

aqi = res_eco["current"]["european_aqi"]
pm10 = res_eco["current"]["pm10"]
pm25 = res_eco["current"]["pm2_5"]

print("Экологическая обстановка:")

# Оценка общего качества воздуха (AQI)
if aqi <= 20:
    print(f"Индекс AQI: {aqi} (Отлично - воздух чистый)")
elif aqi <= 50:
    print(f"Индекс AQI: {aqi} (Хорошо - норма)")
elif aqi <= 80:
    print(f"Индекс AQI: {aqi} (Среднее - есть смог)")
else:
    print(f"Индекс AQI: {aqi} (Плохо - лучше закрыть окна)")

# Оценка крупной пыли (PM10)
if pm10 <= 20:
    desc_pm10 = "чисто"
elif pm10 <= 40:
    desc_pm10 = "терпимо"
else:
    desc_pm10 = "пыльно, скрипит на зубах"

print(f"Крупная пыль (PM10): {pm10} мкг/м³ ({desc_pm10})")

# Оценка мелкой опасной пыли (PM2.5)
if pm25 <= 10:
    desc_pm25 = "чисто"
elif pm25 <= 25:
    desc_pm25 = "норма превышена"
else:
    desc_pm25 = "опасно, мелкий смог"

print(f"Мелкая пыль (PM2.5): {pm25} мкг/м³ ({desc_pm25})")
print("-" * 40)
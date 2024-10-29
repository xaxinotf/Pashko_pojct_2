import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Функція для симуляції Вінерівського процесу
def simulate_wiener_process(T, dt, num_paths):
    num_steps = int(T / dt)
    dW = np.sqrt(dt) * np.random.normal(0, 1, (num_paths, num_steps))
    W = np.cumsum(dW, axis=1)
    t = np.linspace(0, T, num_steps)
    return t, W

# Функція для пошуку часу першого перетину порогового значення
def find_first_passage_time(W, threshold, t):
    first_passage_times = []
    for w in W:
        above_threshold = np.where(np.abs(w) >= threshold)[0]
        if len(above_threshold) > 0:
            first_passage_times.append(t[above_threshold[0]])
        else:
            first_passage_times.append(np.nan)  # Якщо не було перетину
    return np.array(first_passage_times)

# Основні параметри
T = 1.0         # Кінцевий час
dt = 0.001      # Крок часу
num_paths = 100 # Кількість реалізацій
threshold = 1.0 # Порогове значення для першого перетину

# Моделювання Вінерівського процесу
t, W = simulate_wiener_process(T, dt, num_paths)

# Знаходження часу першого перетину
first_passage_times = find_first_passage_time(W, threshold, t)

# Фільтруємо значення без перетину (NaN)
first_passage_times = first_passage_times[~np.isnan(first_passage_times)]

# Побудова гістограми часу першого перетину
plt.figure(figsize=(12, 6))
plt.hist(first_passage_times, bins=20, alpha=0.7, color='b', edgecolor='black')
plt.xlabel('Час першого перетину')
plt.ylabel('Частота')
plt.title('Гістограма часу першого перетину порогового значення (±1) для 100 реалізацій')
plt.grid(True)
plt.show()

# Побудова першої реалізації з відображенням першого перетину
plt.figure(figsize=(12, 6))
plt.plot(t, W[0], lw=2, label='Реалізація 1')
plt.axhline(y=threshold, color='r', linestyle='--', label='Поріг +1')
plt.axhline(y=-threshold, color='r', linestyle='--', label='Поріг -1')

# Додавання точки першого перетину для першої реалізації
above_threshold = np.where(np.abs(W[0]) >= threshold)[0]
if len(above_threshold) > 0:
    first_crossing_time = t[above_threshold[0]]
    first_crossing_value = W[0, above_threshold[0]]
    plt.scatter(first_crossing_time, first_crossing_value, color='red', zorder=5)
    plt.text(first_crossing_time, first_crossing_value, f'  t={first_crossing_time:.3f}', color='red')

plt.xlabel('Час')
plt.ylabel('Значення')
plt.title('Аналіз часу першого виходу - Реалізація 1')
plt.legend()
plt.grid(True)
plt.show()
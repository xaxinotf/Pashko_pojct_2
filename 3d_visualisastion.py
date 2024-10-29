import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def simulate_poisson_process(lambda_rate, T):
    """
    Симулює Пуассонівський процес.

    :param lambda_rate: Інтенсивність процесу
    :param T: Кінцевий час симуляції
    :return: Масив часів подій
    """
    t = 0
    events = []
    while t < T:
        t += np.random.exponential(1 / lambda_rate)
        if t < T:
            events.append(t)
    return np.array(events)

def plot_3d_process_with_histogram(num_realizations, T):
    """
    Будує 3D візуалізацію декількох реалізацій процесу та гістограму інтервалів між подіями.

    :param num_realizations: Кількість реалізацій для візуалізації
    :param T: Кінцевий час симуляції
    """
    fig = plt.figure(figsize=(16, 12))

    # 3D візуалізація Пуассонівського процесу
    ax_3d = fig.add_subplot(211, projection='3d')
    all_intervals = []  # Збираємо всі інтервали для гістограми

    for i in range(num_realizations):
        lambda_rate = random.uniform(0.5, 5.0)  # Випадкове значення інтенсивності в діапазоні від 0.5 до 5.0
        events = simulate_poisson_process(lambda_rate, T)
        x = events
        y = np.ones_like(events) * i
        z = np.arange(1, len(events) + 1)
        ax_3d.plot(x, y, z, marker='o', linestyle='-', label=f'Реалізація {i+1}, λ={lambda_rate:.2f}')
        if len(events) > 1:
            all_intervals.extend(np.diff(events))

    ax_3d.set_xlabel('Час')
    ax_3d.set_ylabel('Номер реалізації')
    ax_3d.set_zlabel('Кількість подій')
    ax_3d.set_title('3D візуалізація реалізацій Пуассонівського процесу')

    # Гістограма інтервалів між подіями
    ax_hist = fig.add_subplot(212)
    ax_hist.hist(all_intervals, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax_hist.set_xlabel('Інтервал між подіями')
    ax_hist.set_ylabel('Частота')
    ax_hist.set_title('Гістограма інтервалів між подіями для всіх реалізацій')
    ax_hist.grid(True)

    plt.tight_layout()
    plt.show()

# Параметри симуляції
T = 10  # Кінцевий час симуляції
num_realizations = 10  # Кількість реалізацій

# Виконання симуляції та візуалізації з гістограмою
plot_3d_process_with_histogram(num_realizations, T)

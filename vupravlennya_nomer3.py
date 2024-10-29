import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
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


def plot_process_realization(events, T, lambda_rate):
    """
    Будує графік реалізації процесу та гістограму інтервалів між подіями.

    :param events: Масив часів подій
    :param T: Кінцевий час симуляції
    :param lambda_rate: Інтенсивність процесу
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Графік реалізації процесу
    axs[0].step(events, range(1, len(events) + 1), where='post')
    axs[0].set_xlabel('Час')
    axs[0].set_ylabel('Кількість подій')
    axs[0].set_title(f'Реалізація Пуассонівського процесу (інтенсивність = {lambda_rate:.4f})')
    axs[0].grid(True)

    # Гістограма інтервалів між подіями
    if len(events) > 1:
        intervals = np.diff(events)
        axs[1].hist(intervals, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        axs[1].set_xlabel('Інтервал між подіями')
        axs[1].set_ylabel('Частота')
        axs[1].set_title('Гістограма інтервалів між подіями')
        axs[1].grid(True)
    else:
        axs[1].text(0.5, 0.5, 'Недостатньо подій для побудови гістограми',
                    horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].set_axis_off()

    plt.tight_layout()
    plt.show()


# Параметри симуляції
T = 10  # Кінцевий час симуляції

# Виконання декількох симуляцій з випадковою інтенсивністю
num_simulations = 5
for i in range(num_simulations):
    lambda_rate = random.uniform(0.5, 5.0)  # Випадкове значення інтенсивності в діапазоні від 0.5 до 5.0

    # Симуляція та візуалізація
    events = simulate_poisson_process(lambda_rate, T)
    plot_process_realization(events, T, lambda_rate)

    # Вивід значення інтенсивності та статистики
    print(f"Симуляція {i + 1}")
    print(f"Інтенсивність, яка використовується для моделювання: {lambda_rate}")
    print(f"Кількість змодельованих подій: {len(events)}")
    if len(events) > 1:
        print(f"Середній інтервал між подіями: {np.mean(np.diff(events)):.4f}")
    else:
        print("Недостатньо подій для обчислення середнього інтервалу")
    print(f"Теоретичний середній інтервал: {1 / lambda_rate:.4f}")
    print("-")

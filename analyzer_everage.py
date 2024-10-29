import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson


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


def plot_process_realization(events, T):
    """
    Будує графік реалізації процесу.

    :param events: Масив часів подій
    :param T: Кінцевий час симуляції
    """
    plt.figure(figsize=(12, 6))
    plt.step(events, range(1, len(events) + 1), where='post')
    plt.xlabel('Час')
    plt.ylabel('Кількість подій')
    plt.title('Реалізація Пуассонівського процесу')
    plt.grid(True)
    plt.show()


def analyze_k_events(lambda_rate, T, k, num_simulations=1000):
    """
    Аналізує середній час та інтервали між першими k подіями для заданої кількості симуляцій.

    :param lambda_rate: Інтенсивність процесу
    :param T: Кінцевий час симуляції
    :param k: Кількість перших подій для аналізу
    :param num_simulations: Кількість симуляцій
    """
    event_times = []
    inter_event_times = []

    for _ in range(num_simulations):
        events = simulate_poisson_process(lambda_rate, T)
        if len(events) >= k:
            event_times.append(events[:k])
            inter_event_times.append(np.diff(events[:k]))

    # Обчислення середніх значень
    mean_event_times = np.mean(event_times, axis=0)
    mean_inter_event_times = np.mean(inter_event_times, axis=0)

    # Побудова графіків
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, k + 1), mean_event_times, marker='o', linestyle='-', color='b', label='Середній час події')
    plt.xlabel('Номер події')
    plt.ylabel('Середній час')
    plt.title(f'Середній час перших {k} подій')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, k), mean_inter_event_times, marker='o', linestyle='-', color='r', label='Середній інтервал між подіями')
    plt.xlabel('Номер інтервалу')
    plt.ylabel('Середній інтервал часу')
    plt.title(f'Середній інтервал між першими {k} подіями')
    plt.grid(True)
    plt.legend()
    plt.show()


# Параметри симуляції
lambda_rate = 2  # Інтенсивність процесу
T = 10  # Кінцевий час симуляції
k = 5  # Кількість перших подій для аналізу

# Базова симуляція та візуалізація
events = simulate_poisson_process(lambda_rate, T)
plot_process_realization(events, T)

# Аналіз середніх значень для перших k подій
analyze_k_events(lambda_rate, T, k)

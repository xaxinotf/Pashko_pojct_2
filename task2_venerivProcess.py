import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import seaborn as sns
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# Переконайтеся, що встановлені необхідні бібліотеки:
# pip install numpy matplotlib scipy seaborn pillow

def simulate_wiener_process(T, dt, num_paths):
    """
    Симулює вінерівський випадковий процес.

    :param T: Кінцевий час
    :param dt: Крок часу
    :param num_paths: Кількість реалізацій
    :return: Масив часових точок та масив реалізацій процесу
    """
    num_steps = int(T / dt)
    dW = np.sqrt(dt) * np.random.normal(0, 1, (num_paths, num_steps))
    W = np.cumsum(dW, axis=1)
    t = np.linspace(0, T, num_steps)
    return t, W

def plot_wiener_realizations_3d(t, W, num_to_plot=5):
    """
    Будує 3D графік реалізацій вінерівського процесу з гістограмою.

    :param t: Масив часових точок
    :param W: Масив реалізацій процесу
    :param num_to_plot: Кількість реалізацій для відображення
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121, projection='3d')

    for i in range(min(num_to_plot, W.shape[0])):
        ax.plot(t, [i] * len(t), W[i], label=f'Реалізація {i + 1}')

    ax.set_xlabel('Час')
    ax.set_ylabel('Номер реалізації')
    ax.set_zlabel('Значення')
    ax.set_title('3D візуалізація реалізацій вінерівського процесу')
    ax.legend()

    # Гістограма для першої реалізації
    ax2 = fig.add_subplot(122)
    ax2.hist(W[0], bins=20, alpha=0.7, color='b', edgecolor='black')
    ax2.set_xlabel('Значення')
    ax2.set_ylabel('Частота')
    ax2.set_title('Гістограма значень для першої реалізації')

    plt.tight_layout()
    plt.show()

def plot_fractal_dimension(t, W, num_to_plot=5):
    """
    Оцінює та візуалізує фрактальну розмірність траєкторій для декількох реалізацій.

    :param t: Масив часових точок
    :param W: Масив реалізацій процесу
    :param num_to_plot: Кількість реалізацій для оцінки
    """
    def box_count(x, y, box_size):
        x_boxes = np.floor(x / box_size).astype(int)
        y_boxes = np.floor(y / box_size).astype(int)
        return len(np.unique(list(zip(x_boxes, y_boxes))))

    fig, ax = plt.subplots(figsize=(10, 6))
    box_sizes = np.logspace(-3, 0, 20)

    for i in range(min(num_to_plot, W.shape[0])):
        counts = [box_count(t, W[i], size) for size in box_sizes]
        ax.loglog(box_sizes, counts, 'o-', label=f'Реалізація {i + 1}')

        # Лінійна апроксимація для оцінки фрактальної розмірності
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(box_sizes), np.log(counts))
        ax.loglog(box_sizes, np.exp(intercept + slope * np.log(box_sizes)), 'r--',
                  label=f'Нахил для реалізації {i + 1}: {-slope:.3f}')

    ax.set_xlabel('Розмір коробки')
    ax.set_ylabel('Кількість коробок')
    ax.set_title('Оцінка фрактальної розмірності для кількох реалізацій')
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    # Параметри симуляції
    T = 1.0         # Кінцевий час
    dt = 0.001      # Крок часу
    num_paths = 10  # Кількість реалізацій

    # Моделювання вінерівського процесу
    t, W = simulate_wiener_process(T, dt, num_paths)

    # 3D візуалізація з гістограмою
    plot_wiener_realizations_3d(t, W)

    # Оцінка фрактальної розмірності для кількох реалізацій
    plot_fractal_dimension(t, W)
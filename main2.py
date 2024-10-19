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
    Будує 3D графік реалізацій вінерівського процесу.

    :param t: Масив часових точок
    :param W: Масив реалізацій процесу
    :param num_to_plot: Кількість реалізацій для відображення
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(min(num_to_plot, W.shape[0])):
        ax.plot(t, [i] * len(t), W[i], label=f'Реалізація {i + 1}')

    ax.set_xlabel('Час')
    ax.set_ylabel('Номер реалізації')
    ax.set_zlabel('Значення')
    ax.set_title('3D візуалізація реалізацій вінерівського процесу')
    plt.legend()
    plt.show()

def animate_wiener_process(t, W, save_path='wiener_process.gif', frame_step=5):
    """
    Створює анімацію вінерівського процесу та зберігає її як GIF.

    :param t: Масив часових точок
    :param W: Масив реалізацій процесу
    :param save_path: Шлях для збереження GIF-файлу
    :param frame_step: Крок між кадрами (зменшує кількість кадрів)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    lines = [ax.plot([], [], lw=2, label=f'Реалізація {i+1}')[0] for i in range(min(5, W.shape[0]))]

    ax.set_xlim(0, t[-1])
    ax.set_ylim(np.min(W), np.max(W))
    ax.set_xlabel('Час')
    ax.set_ylabel('Значення')
    ax.set_title('Анімація вінерівського процесу')
    ax.grid(True)
    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate_frame(i):
        for j, line in enumerate(lines):
            line.set_data(t[:i], W[j, :i])
        return lines

    # Зменшення кількості кадрів шляхом пропуску деяких
    frames = range(0, len(t), frame_step)
    if len(frames) == 0:
        frames = [0]  # Забезпечення принаймні одного кадру

    anim = animation.FuncAnimation(fig, animate_frame, init_func=init,
                                   frames=frames, interval=20, blit=True)

    # Збереження анімації як GIF
    anim.save(save_path, writer='pillow')
    plt.close(fig)
    print(f"Анімація збережена як {save_path}")

def plot_quantiles(t, W):
    """
    Будує графік квантилів процесу.

    :param t: Масив часових точок
    :param W: Масив реалізацій процесу
    """
    quantiles = np.percentile(W, [2.5, 25, 50, 75, 97.5], axis=0)

    plt.figure(figsize=(12, 6))
    plt.fill_between(t, quantiles[0], quantiles[4], alpha=0.3, label='95% довірчий інтервал')
    plt.fill_between(t, quantiles[1], quantiles[3], alpha=0.3, label='50% довірчий інтервал')
    plt.plot(t, quantiles[2], label='Медіана')
    plt.xlabel('Час')
    plt.ylabel('Значення')
    plt.title('Квантилі вінерівського процесу')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_heatmap(t, W):
    """
    Будує теплову карту реалізацій процесу.

    :param t: Масив часових точок
    :param W: Масив реалізацій процесу
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(W[:100], cmap='viridis', cbar_kws={'label': 'Значення'})
    plt.xlabel('Крок часу')
    plt.ylabel('Номер реалізації')
    plt.title('Теплова карта реалізацій вінерівського процесу')
    plt.show()

def plot_distribution_evolution(t, W, save_path='distribution_evolution.gif', frame_step=20):
    """
    Будує еволюцію розподілу процесу в часі та зберігає як GIF.

    :param t: Масив часових точок
    :param W: Масив реалізацій процесу
    :param save_path: Шлях для збереження GIF-файлу
    :param frame_step: Крок між кадрами (зменшує кількість кадрів)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    def update(frame):
        ax.clear()
        sns.kdeplot(W[:, frame], ax=ax)
        ax.set_xlim(np.min(W), np.max(W))
        ax.set_ylim(0, 0.5)
        ax.set_title(f'Розподіл вінерівського процесу в момент t={t[frame]:.2f}')
        ax.set_xlabel('Значення')
        ax.set_ylabel('Густина')

    # Зменшення кількості кадрів шляхом пропуску деяких
    frames = range(0, len(t), frame_step)
    if len(frames) == 0:
        frames = [0]  # Забезпечення принаймні одного кадру

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=100)
    anim.save(save_path, writer='pillow')
    plt.close(fig)
    print(f"Еволюція розподілу збережена як {save_path}")

def plot_fractal_dimension(t, W):
    """
    Оцінює та візуалізує фрактальну розмірність траєкторій.

    :param t: Масив часових точок
    :param W: Масив реалізацій процесу
    """

    def box_count(x, y, box_size):
        x_boxes = np.floor(x / box_size).astype(int)
        y_boxes = np.floor(y / box_size).astype(int)
        return len(np.unique(list(zip(x_boxes, y_boxes))))

    box_sizes = np.logspace(-3, 0, 20)
    counts = [box_count(t, W[0], size) for size in box_sizes]

    plt.figure(figsize=(10, 6))
    plt.loglog(box_sizes, counts, 'bo-')
    plt.xlabel('Розмір коробки')
    plt.ylabel('Кількість коробок')
    plt.title('Оцінка фрактальної розмірності')

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(box_sizes), np.log(counts))
    plt.loglog(box_sizes, np.exp(intercept + slope * np.log(box_sizes)), 'r-',
               label=f'Нахил: {-slope:.3f} (розмірність: {-slope:.3f})')
    plt.legend()
    plt.grid(True)
    plt.show()

def interactive_first_passage_time(t, W):
    """
    Інтерактивний графік для аналізу часу першого виходу.

    :param t: Масив часових точок
    :param W: Масив реалізацій процесу
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    line, = ax.plot(t, W[0], lw=2)
    ax.axhline(y=1, color='r', linestyle='--')
    ax.axhline(y=-1, color='r', linestyle='--')

    ax.set_xlabel('Час')
    ax.set_ylabel('Значення')
    ax.set_title('Аналіз часу першого виходу - Реалізація 1')

    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'Реалізація', 0, W.shape[0] - 1, valinit=0, valstep=1)

    def update(val):
        i = int(slider.val)
        line.set_ydata(W[i])
        ax.set_title(f'Аналіз часу першого виходу - Реалізація {i + 1}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    # Параметри симуляції
    T = 1.0         # Кінцевий час
    dt = 0.001      # Крок часу
    num_paths = 1000  # Кількість реалізацій

    # Моделювання вінерівського процесу
    t, W = simulate_wiener_process(T, dt, num_paths)

    # 3D візуалізація
    plot_wiener_realizations_3d(t, W)

    # Анімація в реальному часі (збереження як GIF)
    animate_wiener_process(t, W, save_path='wiener_process.gif', frame_step=5)

    # Графік квантилів
    plot_quantiles(t, W)

    # Теплова карта
    plot_heatmap(t, W)

    # Еволюція розподілу (збереження як GIF)
    plot_distribution_evolution(t, W, save_path='distribution_evolution.gif', frame_step=20)

    # Оцінка фрактальної розмірності
    plot_fractal_dimension(t, W)

    # Інтерактивний аналіз часу першого виходу
    interactive_first_passage_time(t, W)

    print("Аналіз вінерівського процесу завершено.")

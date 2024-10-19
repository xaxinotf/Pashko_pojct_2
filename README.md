ЗАВДАННЯ 1

1. Моделювання Пуассонівського потоку з заданою інтенсивністю:

Функція `simulate_poisson_process(lambda_rate, T)` моделює Пуассонівський потік:



```python
def simulate_poisson_process(lambda_rate, T):
    t = 0
    events = []
    while t < T:
        t += np.random.exponential(1/lambda_rate)
        if t < T:
            events.append(t)
    return np.array(events)
```

Ця функція використовує той факт, що інтервали між подіями у Пуассонівському процесі розподілені експоненційно з параметром λ (lambda_rate). Вона генерує послідовність подій до досягнення кінцевого часу T.

2. Побудова графіків реалізацій процесу:

Функція `plot_process_realization(events, T)` будує графік реалізації процесу:

```python
def plot_process_realization(events, T):
    plt.figure(figsize=(12, 6))
    plt.step(events, range(1, len(events) + 1), where='post')
    plt.xlabel('Час')
    plt.ylabel('Кількість подій')
    plt.title('Реалізація Пуассонівського процесу')
    plt.grid(True)
    plt.show()
```

Ця функція створює ступінчатий графік, де кожен крок відповідає появі нової події.

3. Побудова гістограм розподілів:

а) Час появи заданої події (перша, друга, n-та):

Функція `plot_event_time_histogram(events, event_number)` будує гістограму часу появи заданої події:

```python
def plot_event_time_histogram(events, event_number):
    if event_number <= len(events):
        plt.figure(figsize=(10, 6))
        plt.hist(events[event_number - 1], bins=30, edgecolor='black')
        plt.xlabel('Час')
        plt.ylabel('Частота')
        plt.title(f'Гістограма часу появи {event_number}-ї події')
        plt.grid(True)
        plt.show()
    else:
        print(f"Недостатньо подій для побудови гістограми {event_number}-ї події")
```

б) Інтервал між подіями:

Функція `plot_interarrival_time_histogram(events)` будує гістограму інтервалів між подіями:

```python
def plot_interarrival_time_histogram(events):
    interarrival_times = np.diff(events)
    plt.figure(figsize=(10, 6))
    plt.hist(interarrival_times, bins=30, edgecolor='black')
    plt.xlabel('Інтервал часу')
    plt.ylabel('Частота')
    plt.title('Гістограма інтервалів між подіями')
    plt.grid(True)
    plt.show()
```

в) Поява рівно n подій:

Функція `plot_n_events_histogram(lambda_rate, T, n, num_simulations=1000)` будує гістограму появи рівно n подій:

```python
def plot_n_events_histogram(lambda_rate, T, n, num_simulations=1000):
    event_counts = [len(simulate_poisson_process(lambda_rate, T)) for _ in range(num_simulations)]
    plt.figure(figsize=(10, 6))
    plt.hist(event_counts, bins=range(min(event_counts), max(event_counts) + 2), align='left', rwidth=0.8, edgecolor='black')
    plt.xlabel('Кількість подій')
    plt.ylabel('Частота')
    plt.title(f'Гістограма появи подій (очікувана кількість: {n})')
    plt.grid(True)
    plt.show()
```

Додаткові розширення:

1. 3D візуалізація:

Функція `plot_3d_process_realization(events, T, num_realizations=10)` створює 3D візуалізацію декількох реалізацій процесу:

```python
def plot_3d_process_realization(events, T, num_realizations=10):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(num_realizations):
        events = simulate_poisson_process(lambda_rate, T)
        x = events
        y = np.ones_like(events) * i
        z = np.arange(1, len(events) + 1)
        ax.plot(x, y, z, marker='o')
    
    ax.set_xlabel('Час')
    ax.set_ylabel('Номер реалізації')
    ax.set_zlabel('Кількість подій')
    ax.set_title('3D візуалізація реалізацій Пуассонівського процесу')
    plt.show()
```

2. Графік в реальному часі:

Функція `plot_real_time_process(lambda_rate, T, update_interval=100)` будує графік процесу в реальному часі:

```python
def plot_real_time_process(lambda_rate, T, update_interval=100):
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], 'b-')
    ax.set_xlim(0, T)
    ax.set_ylim(0, lambda_rate * T * 1.5)
    ax.set_xlabel('Час')
    ax.set_ylabel('Кількість подій')
    ax.set_title('Пуассонівський процес в реальному часі')
    ax.grid(True)

    events = []

    def init():
        line.set_data([], [])
        return line,

    def animate(frame):
        t = frame * T / 1000
        if not events or events[-1] <= t:
            new_event = simulate_poisson_process(lambda_rate, t)
            events.extend(new_event[new_event > (events[-1] if events else 0)])
        x = events
        y = range(1, len(events) + 1)
        line.set_data(x, y)
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=1000, init_func=init, blit=True, interval=update_interval)
    plt.show()
```

3. Інтерактивний аналіз впливу інтенсивності:

Функція `interactive_rate_analysis(T, max_rate=10)` створює інтерактивний графік з можливістю змінювати інтенсивність:

```python
def interactive_rate_analysis(T, max_rate=10):
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    events = simulate_poisson_process(1, T)
    line, = ax.step(events, range(1, len(events) + 1), where='post')
    
    ax.set_xlim(0, T)
    ax.set_ylim(0, max_rate * T)
    ax.set_xlabel('Час')
    ax.set_ylabel('Кількість подій')
    ax.set_title('Вплив інтенсивності на Пуассонівський процес')
    ax.grid(True)

    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'Інтенсивність', 0.1, max_rate, valinit=1)

    def update(val):
        rate = slider.val
        events = simulate_poisson_process(rate, T)
        line.set_xdata(events)
        line.set_ydata(range(1, len(events) + 1))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
```

4. Аналіз збіжності:

Функція `analyze_convergence(lambda_rate, T, num_simulations=1000)` аналізує збіжність середнього числа подій до теоретичного значення:

```python
def analyze_convergence(lambda_rate, T, num_simulations=1000):
    event_counts = [len(simulate_poisson_process(lambda_rate, T)) for _ in range(num_simulations)]
    cumulative_means = np.cumsum(event_counts) / np.arange(1, num_simulations + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_simulations + 1), cumulative_means, label='Емпіричне середнє')
    plt.axhline(y=lambda_rate * T, color='r', linestyle='--', label='Теоретичне середнє')
    plt.xlabel('Кількість симуляцій')
    plt.ylabel('Середня кількість подій')
    plt.title('Збіжність середнього числа подій')
    plt.legend()
    plt.grid(True)
    plt.show()
```

Всі ці функції разом дозволяють всебічно дослідити та візуалізувати Пуассонівський процес, включаючи його базові характеристики та більш складні аспекти, такі як збіжність до теоретичних значень та вплив зміни параметрів на поведінку процесу.


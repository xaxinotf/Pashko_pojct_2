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



================================ЗАВДАННЯ 2================================



## Вимоги Завдання та Їх Виконання

### 1. **Змоделювати неперервний вінерівський випадковий процес.**

**Виконання:**
- **Функція `simulate_wiener_process`** відповідає цій вимозі. Вона симулює вінерівський процес (також відомий як Броунівський рух) за допомогою нормального розподілу для генерації випадкових змінних.
- **Параметри Симуляції:**
  - `T = 1.0`: Кінцевий час симуляції.
  - `dt = 0.001`: Часовий крок.
  - `num_paths = 1000`: Кількість реалізацій процесу (значно більше 100, що відповідає вимозі).
- **Механізм Симуляції:**
  - Генерується матриця `dW` розміру `(num_paths, num_steps)`, де `num_steps = T / dt`.
  - Використовується накопичення кроків (`np.cumsum`) для створення траєкторій процесу `W(t)`.
  - Додається початкова точка `W(0) = 0` для кожної реалізації.

```python
def simulate_wiener_process(T, dt, num_paths):
    num_steps = int(T / dt)
    dW = np.sqrt(dt) * np.random.normal(0, 1, (num_paths, num_steps))
    W = np.cumsum(dW, axis=1)
    W = np.hstack((np.zeros((num_paths, 1)), W))
    t = np.linspace(0, T, num_steps + 1)
    return t, W
```

### 2. **За реалізаціями (кількість реалізацій > 100) оцінити середнє значення та дисперсію.**

**Виконання:**
- **Кількість Реалізацій:** Встановлено `num_paths = 1000`, що перевищує мінімальну вимогу в 100 реалізацій.
- **Функція `compute_statistics`:** Обчислює середнє значення (`mean_W`) та дисперсію (`var_W`) для кожного часовго кроку за всіма реалізаціями.
  
```python
def compute_statistics(W):
    mean_W = np.mean(W, axis=0)
    var_W = np.var(W, axis=0)
    return mean_W, var_W
```

- **Візуалізація:** Функція `plot_mean_variance` відображає графіки середнього значення та дисперсії на одному малюнку з двома осями Y, що дозволяє легко порівнювати ці статистики.

```python
def plot_mean_variance(t, mean_W, var_W):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Час')
    ax1.set_ylabel('Середнє W(t)', color=color)
    ax1.plot(t, mean_W, color=color, label='Середнє')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # Створення другого осі для дисперсії
    color = 'tab:red'
    ax2.set_ylabel('Дисперсія', color=color)
    ax2.plot(t, var_W, color=color, label='Дисперсія')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Середнє та дисперсія вінерівського процесу')
    plt.grid(True)
    plt.show()
```

### 3. **Знайти емпіричний закон розподілу ймовірностей часу першого виходу вінерівського процесу за заданий рівень.**

**Виконання:**
- **Функція `find_first_exit_times`:** Визначає час першого виходу за заданий рівень (наприклад, `level = 1.0`) для кожної реалізації процесу. Якщо процес не виходить за рівень, записується `NaN`.

```python
def find_first_exit_times(W, level=1.0):
    num_paths, num_steps = W.shape
    first_exit_times = np.full(num_paths, np.nan)  # Ініціалізація з NaN

    for i in range(num_paths):
        exit_indices = np.where(np.abs(W[i, :]) >= level)[0]
        if exit_indices.size > 0:
            first_exit_times[i] = exit_indices[0]
    return first_exit_times
```

- **Функція `plot_first_exit_distribution`:** Створює гістограму з ядровою оцінкою густини (KDE) для часу першого виходу, що дозволяє візуально оцінити емпіричний розподіл.

```python
def plot_first_exit_distribution(first_exit_times, dt, level=1.0):
    exit_times = first_exit_times[~np.isnan(first_exit_times)] * dt

    plt.figure(figsize=(12, 6))
    sns.histplot(exit_times, bins=30, kde=True, stat="density")
    plt.xlabel('Час першого виходу')
    plt.ylabel('Густина ймовірності')
    plt.title(f'Емпіричний розподіл часу першого виходу за рівень {level}')
    plt.grid(True)
    plt.show()
```

### 4. **Забезпечити візуалізацію результатів.**

**Виконання:**
- **Зразкові Траєкторії:** Функція `plot_sample_paths` відображає кілька випадкових реалізацій процесу для візуального аналізу поведінки процесу.

```python
def plot_sample_paths(t, W, num_samples=5):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.plot(t, W[i], lw=1, label=f'Реалізація {i+1}')
    plt.xlabel('Час')
    plt.ylabel('W(t)')
    plt.title('Зразкові траєкторії вінерівського процесу')
    plt.legend()
    plt.grid(True)
    plt.show()
```

- **Середнє Значення та Дисперсія:** Як описано вище, функція `plot_mean_variance` візуалізує ці статистики.

- **Розподіл Часу Першого Виходу:** Функція `plot_first_exit_distribution` забезпечує графічне відображення емпіричного розподілу часу першого виходу.

- **Анімація Траєкторій:** Функція `animate_sample_paths` створює анімацію зразкових траєкторій процесу та зберігає її у форматі GIF. Для оптимізації ресурсу використовується параметр `frame_step`, який зменшує кількість кадрів.

```python
def animate_sample_paths(t, W, num_samples=5, save_path='wiener_animation.gif', frame_step=10):
    fig, ax = plt.subplots(figsize=(12, 6))
    lines = [ax.plot([], [], lw=2, label=f'Реалізація {i+1}')[0] for i in range(num_samples)]

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(np.min(W[:num_samples]) - 1, np.max(W[:num_samples]) + 1)
    ax.set_xlabel('Час')
    ax.set_ylabel('W(t)')
    ax.set_title('Анімація зразкових траєкторій вінерівського процесу')
    ax.grid(True)
    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        for j, line in enumerate(lines):
            line.set_data(t[:i], W[j, :i])
        return lines

    frames = range(0, len(t), frame_step)
    if len(frames) == 0:
        frames = [0]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=50, blit=True)

    # Збереження анімації як GIF
    anim.save(save_path, writer='pillow')
    plt.close(fig)
    print(f"Анімація збережена як {save_path}")
```

### 5. **Оцінка Фрактальної Розмірності (Додатково)**

**Виконання:**
- **Функція `plot_fractal_dimension`:** Оцінює та візуалізує фрактальну розмірність траєкторій процесу за допомогою методу коробок (Box-Counting).

```python
def plot_fractal_dimension(t, W):
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
```

- **Опис:** Ця функція аналізує першу траєкторію процесу `W[0]`, оцінюючи, як кількість коробок змінюється зі зміною розміру коробки, що дозволяє визначити фрактальну розмірність.

### 6. **Інтерактивний Аналіз Часу Першого Виходу (Додатково)**

**Виконання:**
- **Функція `interactive_first_passage_time`:** Створює інтерактивний графік з використанням слайдера, дозволяючи користувачу обирати різні реалізації для аналізу часу першого виходу.

```python
def interactive_first_passage_time(t, W):
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
```

- **Опис:** Дозволяє користувачу вибирати різні реалізації процесу та аналізувати їхню поведінку щодо часу першого виходу за заданий рівень.

## Підсумок Відповідності Вимогам

1. **Змоделювати неперервний вінерівський випадковий процес.**
   - **Виконано:** Функція `simulate_wiener_process` генерує вінерівський процес з великою кількістю реалізацій (>100).

2. **За реалізаціями (кількість реалізацій > 100) оцінити середнє значення та дисперсію.**
   - **Виконано:** Функція `compute_statistics` обчислює середнє та дисперсію для кожного часовго кроку за 1000 реалізаціями, що відповідає вимозі.

3. **Знайти емпіричний закон розподілу ймовірностей часу першого виходу вінерівського процесу за заданий рівень.**
   - **Виконано:** Функція `find_first_exit_times` визначає час першого виходу за рівень для кожної реалізації, а функція `plot_first_exit_distribution` візуалізує емпіричний розподіл цих часів.

4. **Забезпечити візуалізацію результатів.**
   - **Виконано:** Код включає кілька функцій для візуалізації:
     - `plot_sample_paths`: Зразкові траєкторії процесу.
     - `plot_mean_variance`: Середнє та дисперсія.
     - `plot_first_exit_distribution`: Розподіл часу першого виходу.
     - `animate_sample_paths`: Анімація траєкторій.
     - `plot_fractal_dimension`: Оцінка фрактальної розмірності.
     - `interactive_first_passage_time`: Інтерактивний аналіз часу першого виходу.


## Висновок

Представлений код повністю відповідає всім поставленим вимогам завдання зі статистичного моделювання вінерівського випадкового процесу. Він забезпечує симуляцію процесу, обчислення необхідних статистик, аналіз часу першого виходу та їх візуалізацію через різноманітні графіки та анімації. Додатково включено оцінку фрактальної розмірності та інтерактивний аналіз, що розширює можливості дослідження процесу.


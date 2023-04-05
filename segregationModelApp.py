from tkinter import *
from tkinter import ttk
from scipy.signal import correlate2d
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from time import sleep
from IPython.display import clear_output


class Cell2D:
    """Базовый класс для построения 2-D клеточнего автомата."""

    def __init__(self, n, m=None):
        """Инициализация атрибутов

        n: количество строк
        m: количество столбцов
        """
        m = n if m is None else m
        self.array = np.zeros((n, m), np.uint8)

    def add_cells(self, row, col, *strings):
        """Добавление клеток в заданную локацию

        row: вернхий индекс строки
        col: левый индекс столбца
        strings: список строк из 0 и 1
        """
        for i, s in enumerate(strings):
            self.array[row + i, col:col + len(s)] = np.array([int(b) for b in s])

    def loop(self, iters=1):
        """Запуск заданного количества шагов"""
        for i in range(iters):
            self.step()

    def draw(self, **options):
        """Отрисовка массива"""
        draw_array(self.array, **options)

    def animate(self, frames, interval=None, step=None):
        """Анимация автомата

        frames: количество отрисовываемых фреймов
        interval: время между фреймами в секундах
        step: функция, реализующая 1 шаг перемещений
        """
        if step is None:
            step = self.step

        try:
            for i in range(frames - 1):
                self.draw(animation=True)
                plt.show()
                if interval:
                    sleep(interval)
                step()
                clear_output(wait=True)
            self.draw(animation=True)
            plt.show()
        except KeyboardInterrupt:
            pass


def draw_array(array, animation=None, **options):
    """Отрисовка клеток"""
    n, m = array.shape
    options = underride(options,
                        cmap='Greens',
                        alpha=0.7,
                        vmin=0, vmax=1,
                        interpolation='none',
                        origin='upper',
                        extent=[0, m, 0, n])
    if animation == True:
        plt.figure(figsize=(10, 10))
    plt.axis([0, m, 0, n])
    plt.xticks([])
    plt.yticks([])

    return plt.imshow(array, **options)



#### Вспомогательные функции-утилиты

def three_frame(world, n_seq, seed=17):
    """Отрисовка снимков 3х состояний сегрегации системы

    world: объект класса Schelling с функциями step, loop, and draw
    n_seq: картеж из 3 значений, количество шагов перед каждым снимком
    seed: случайное see для NumPy
    """
    np.random.seed(seed)
    plt.subplots(figsize=(20, 20))
    for i, n in enumerate(n_seq):
        # plt.figure(figsize=(10,10))
        plt.subplot(1, 3, i + 1)
        plt.title(f"{n} шагов")
        world.loop(n)
        world.draw()

    plt.tight_layout()


def underride(d, **options):
    """Добавление пар ключ-значение в словарь-опций (d), если ключа нет в d
    d: словарь с опциями для графика
    options: ключевые слова args для добавление в словарь-опций d
    """
    for key, val in options.items():
        d.setdefault(key, val)
    return d


def decorate(**options):
    """Декорирование осей графика

    Вызов декорирования с агрументами-опциями вида:
    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    Агрументы-опции могут быть любыми из опций декорирования осей
    https://matplotlib.org/api/axes_api.html

    Дополнительно, вы можете использовать `legend=False` для отключения легенды

    Используйте `loc` для указания положения легенды на графике
    (значение по-умолачанию - 'best')
    """
    loc = options.pop("loc", "best")
    if options.pop("legend", True):
        legend(loc=loc)

    plt.gca().set(**options)
    # plt.figure(figsize=(10, 6))
    plt.tight_layout()


def legend(**options):
    """Отрисовка легенды, если есть хотя бы 1 элемент с лейблом.
    опции передаются в plt.legend()
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
    """
    underride(options, loc="best", frameon=False)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **options)


def set_palette(*args, **kwds):
    """Задание цвеовой паллитры для matplotlib

    args, kwds: то же что и для sns.color_palette

    Также принимает kwd (boolean), `reverse`, для обозначения того
    будет ли порядок цветов паллитры обратным

    returns: список цветов
    """
    reverse = kwds.pop('reverse', False)
    palette = sns.color_palette(*args, **kwds)

    palette = list(palette)
    if reverse:
        palette.reverse()

    cycler = plt.cycler(color=palette)
    plt.gca().set_prop_cycle(cycler)
    return palette


def locs_where(condition):
    """ Находит клетки, где логический массив равен True (is True)
    condition: логический массив
    returns: список из кортежей местоположения (list of location tuples)
    """
    return list(zip(*np.nonzero(condition)))


# Справка: "похожие" соседи - это соседи того же цвета, что и исследуемый агент

# создаем пользовательскую цветную карту
palette = sns.color_palette('muted')
colors = 'white', palette[1], palette[0]
cmap = LinearSegmentedColormap.from_list('cmap', colors)


class Schelling(Cell2D):
    """Представление сетки для агентов Шеллинга"""

    # для функции correlate2d:
    # параметры same - результат будет того же размера, что и входные данные
    # wrap - верхний край сворачивается так, чтобы соответствовать нижнему,
    # а левый край – правому.
    options = dict(mode='same', boundary='wrap')
    # указываем, что соседство = 8 агентам
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.int8)

    def __init__(self, n, p):
        """Инициализация агентов на сетке.

        n: количество строк и столбцов (= размер сетки)
        p: порог толерантности (минимальная доля агентов того же цвета в соседстве)
        напр. если p = 0.3, то агенты цвета Х
        будут довольны, когда 30% или более соседей цвета Х.
        """
        self.p = p
        # 0 -пустая клетка, 1 - красный агент, 2 - синий агент
        choices = np.array([0, 1, 2], dtype=np.int8)
        # распределение типов клеток: 10% - пустые, по 45% синих и красных
        probs = [0.1, 0.45, 0.45]
        # сгенерированная карта
        self.array = np.random.choice(choices, (n, n), p=probs)

    def count_neighbors(self):
        """Исследование соседних клеток

        возвращает: кортежи (пустые клетки; доли красных; доли синих; доли одинаковых)
            empty: логический кортеж, в котором True соответствует пустой клетке
            frac_red: доли красных соседей вокруг каждой клетки
            frac_blue:  доли синих соседей вокруг каждой клетки
            frac_same: доли "похожих соседей"
        """
        a = self.array

        # формируем логические массивы
        # пустые ячейки на карте
        empty = a == 0
        # красные ячейки на карте
        red = a == 1
        # синие ячейки на карте
        blue = a == 2

        # подсчитываем красных и синих соседей для каждого местоположения
        num_red = correlate2d(red, self.kernel, **self.options)
        num_blue = correlate2d(blue, self.kernel, **self.options)
        num_neighbors = num_red + num_blue

        # вычисляем доли красных/синих соседей для каждого агента
        frac_red = num_red / num_neighbors
        frac_blue = num_blue / num_neighbors

        # учитываем редкий случай:
        # принимаем отсуствие соседей в окрестности равным отсуствию "похожих" соседей
        frac_red[num_neighbors == 0] = 0
        frac_blue[num_neighbors == 0] = 0

        # каждому агенту присваиваем долю соседей того же цвета, что и агент
        frac_same = np.where(red, frac_red, frac_blue)

        # для пустых клеток, frac_same получает NaN
        frac_same[empty] = np.nan

        return empty, frac_red, frac_blue, frac_same

    def segregation(self):
        """Расчет итоговой степени сегрегации, являющейся средней по агентам

        returns: доля похожих соседей, усредненная для каждой клетки
        """
        _, _, _, frac_same = self.count_neighbors()
        return np.nanmean(frac_same)

    def step(self):
        """функция, реализующая перестановку всех недовольных агентов

        возвращает: долю одинаковых соседей, усредненную для каждой клетки
        """
        a = self.array
        empty, _, _, frac_same = self.count_neighbors()

        # поиск недовольных агентов (игнорируем NaN в frac_same)
        with np.errstate(invalid='ignore'):
            unhappy = frac_same < self.p
        unhappy_locs = locs_where(unhappy)

        # поиск координат пустых клеток
        empty_locs = locs_where(empty)

        # перемешиваем клетки с несчастными агентами
        if len(unhappy_locs):
            np.random.shuffle(unhappy_locs)

        # для каждого несчастного агента выбираем случайную пустую клетку для перемещения
        num_empty = np.sum(empty)
        for source in unhappy_locs:
            i = np.random.randint(num_empty)
            # координаты пустой случайной клетки, куда перемещаемся
            dest = empty_locs[i]

            # перемещение
            a[dest] = a[source]
            a[source] = 0
            empty_locs[i] = source

        # проверка: количество пустых клеток должно остаться прежним
        num_empty2 = np.sum(a == 0)
        assert num_empty == num_empty2

        # возвращаем усредненное значение долей одинаковых соседей
        print('frac_same: ', np.nanmean(frac_same))
        return np.nanmean(frac_same)

    def draw(self, animation=None):
        """Отрисовка ячеек"""
        return draw_array(self.array, animation, cmap=cmap, vmax=2)

def generate_map(*args):
    try:
        plt.close()
        global grid
        size_value = size.get()
        tolerance_value = tolerance.get()
        grid = Schelling(n=size_value, p=tolerance_value)
        grid.draw()
        segregation.set(round(grid.segregation(), 3))
        plt.title(f"Модель сегрегации Шеллинга размером n = {size.get()}х{size.get()}")
        plt.show()
        segregation.set(grid.segregation())
    except ValueError:
        pass

def snapshot_model(*args):
    try:
        plt.close()
        steps = [step1.get(), step2.get(), step3.get()]
        three_frame(grid, steps)
        plt.suptitle(f"Модель сегрегации Шеллинга размером n = {size.get()}х{size.get()}")
        plt.show()
    except ValueError:
        pass

def tolerance_params_plot(*args):
    try:
        plt.close()
        set_palette('Blues', 5, reverse=True)
        np.random.seed(17)
        labels = []
        for row in range(12, 16):
            for column in range(3):
                label = ttk.Label(mainframe, width=5)
                label.grid(row= row + 1, column = column + 1, padx=10, sticky=W)
                labels.append(label)
        p_list = []
        seg_list = []
        for p in [tolerance1.get(), tolerance2.get(), tolerance3.get(), tolerance4.get()]:
            grid = Schelling(n=100, p=p)
            segs = [grid.step() for i in range(12)]
            plt.plot(segs, label='p = %.1f' % p)
            p_list.append(p)
            seg_list.append(segs[-1])
        cell = 0
        for i in range(4):
            labels[cell].configure(text=p_list[i])
            labels[cell+1].configure(text=round(seg_list[i], 3))
            labels[cell+2].configure(text=round(seg_list[i] - p_list[i], 3))
            cell += 3
        decorate(xlabel='шаг', ylabel='уровень сегрегации (усредненный)',
                 loc='lower right', ylim=[0, 1], title='Степень сегрегации в модели Шеллинга\n'
                                                       'с течением времени для диапазона p')
        plt.grid()
        plt.show()
    except ValueError:
        pass


root = Tk()
root.title("Модель сегрегации Шеллинга")
root.resizable(width=False, height=False)
global figure
figure = 1

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

size = IntVar()
size_entry = ttk.Entry(mainframe, width=2, textvariable=size)
size.set(30)
size_entry.grid(column=1, row=1, sticky=(W, E))

tolerance = DoubleVar()
tolerance_entry = ttk.Entry(mainframe, width=2, textvariable=tolerance)
tolerance.set(0.3)
tolerance_entry.grid(column=1, row=2, sticky=(W, E))

ttk.Label(mainframe, text="размер карты").grid(column=2, row=1, sticky=W)
ttk.Label(mainframe, text="толерантность (p)").grid(column=2, row=2, sticky=W)
segregation = StringVar()
ttk.Label(mainframe, textvariable=segregation).grid(column=1, row=3, sticky=(W, E))
ttk.Label(mainframe, text="итоговая сегрегация (усредненная)").\
    grid(column=2, row=3, columnspan = 2, sticky=W)
ttk.Button(mainframe, text="Сгенерировать карту", command=generate_map)\
    .grid(column=3, columnspan = 2, row=4, sticky=W)

ttk.Label(mainframe, text="Сегрегация на\nх, y, z шагах:").grid(column=1, row=5, sticky=E)
step1 = IntVar()
step2 = IntVar()
step3 = IntVar()
step1_entry = ttk.Entry(mainframe, width=4, textvariable=step1).grid(column=1, row=6, sticky=W)
step1.set(0)
step2_entry = ttk.Entry(mainframe, width=4, textvariable=step2).grid(column=2, row=6, sticky=W)
step2.set(2)
step3_entry = ttk.Entry(mainframe, width=4, textvariable=step3).grid(column=3, row=6, sticky=W)
step3.set(8)
ttk.Button(mainframe, text="Запуск моделирования", command=snapshot_model)\
    .grid(column=3, columnspan=2, row=7, sticky=W)

tolerance1 = DoubleVar()
tolerance2 = DoubleVar()
tolerance3 = DoubleVar()
tolerance4 = DoubleVar()
ttk.Label(mainframe, text="Увеличение степени сегрегации при разных p")\
    .grid(column=1, row=8,  columnspan = 3, sticky=E)
ttk.Entry(mainframe, width=4, textvariable=tolerance1).grid(column=1, row=9, sticky=W)
tolerance1.set(0.2)
ttk.Entry(mainframe, width=4, textvariable=tolerance2).grid(column=2, row=9, sticky=W)
tolerance2.set(0.3)
ttk.Entry(mainframe, width=4, textvariable=tolerance3).grid(column=3, row=9, sticky=W)
tolerance3.set(0.4)
ttk.Entry(mainframe, width=4, textvariable=tolerance4).grid(column=4, row=9, sticky=W)
tolerance4.set(0.5)
ttk.Button(mainframe, text="Построить график", command=tolerance_params_plot)\
    .grid(column=3, columnspan=2, row=10, sticky=W)

ttk.Label(mainframe, text="Увеличение степени сегрегации при разных p (график):")\
    .grid(column=1, row=11,  columnspan = 3, sticky=E)
ttk.Label(mainframe, text="толерантность").grid(column=1, row=12, sticky=W)
ttk.Label(mainframe, text="итоговая сегрегация").grid(column=2, row=12, sticky=W)
ttk.Label(mainframe, text="разница").grid(column=3, row=12, sticky=W)

for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

size_entry.focus()
root.bind("<Return>", generate_map)

root.mainloop()



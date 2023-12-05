import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Queue
from matplotlib.patches import Circle
from matplotlib.axes import Axes
from matplotlib.widgets import Slider
from icecream import ic

import modules.data as d

def parseToResults(t_values, state_vector_values):
    """
    Сведение результатов интегрирования, к итоговому виду
    """
    #Объединение t_values и state_vector_values в список results
    results = list(zip(t_values, state_vector_values))
    #Распаковка state_vector_values
    results = [(t_values, *state_vector_values) for t_values, state_vector_values in results]
    #Приведение к СИ
    results = [(t_values, x, y, v_x * 1000, v_y * 1000, m) for t_values, x, y, v_x, v_y, m in results]
    #Добавление НУ в начало списка
    results.insert(0, (d.T_0, d.X_0, d.Y_0, d.V_X_0 * 1000, d.V_Y_0 * 1000, d.M_0))
    return results

def writeToExcel(results: list, filename: str):
    if not d.EXCEL:
        return
    
    #Для фиксации времени исполнения
    ic()
    #columns = ['t, с', 'm, кг' ,'v_x, м/с', 'v_y, м/с', 'x, км', 'y, км', 'h, км', 'V, м/с', 'r, км', 'ϑ, град','θ_c, град', 'α, град', 'ϕ, град']
    columns = ["t, с", "x, км", "y, км", "v_x, м/с", "v_y, м/с", "m, кг"]
    df = pd.DataFrame(results, columns=columns)

    if not os.path.exists("results"):
        os.mkdir("results")

    df.to_excel(f"results/{filename}", index=False)

    #Для фиксации времени исполнения
    ic()

def createPlot(results: list, num_points: int = 1000, H_MS: float = d.H_MS_11, interactive: bool = d.INTERACTIVE):
    """
    :param results: список всех вектор состояний
    :param num_points: количество точек для построения графика
    :param H_MS: высота целевой орбиты
    """
    #Для фиксации времени исполнения
    ic()

    #Массив равномерно прореженных индексов
    indices = np.linspace(0, len(results)-1, num_points, dtype=int)

    #Прорежение исходного списка результатов
    thined_results = [results[i] for i in indices]

    #Распаковка результатов
    t_results, x_results, y_results, v_x_results, v_y_results, m_results = zip(*thined_results)
    
    # Перенос стартовой СК на поверхность Луны, начало абсолютных координат в ц.м. Луны
    h_results = [y + d.R_MOON for y in y_results]

    # Создаем график
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    plt.gca().set_xlim([-2200, 2200])
    plt.gca().set_ylim([-2200, 2200])

    ax.set_aspect('equal')

    # Добавляем луну и целевую орбиту
    moon = Circle((0, 0), d.R_MOON, fill=True, color="grey")
    target_orbit = Circle((0, 0), d.R_MOON + H_MS, fill=False, color="red", linestyle="--")
    ax.add_patch(moon)
    ax.add_patch(target_orbit)

    if not interactive:
        l, = plt.plot(x_results, h_results, visible=True)
        plt.show()
        return
    
    l, = plt.plot(x_results, h_results, visible=False)

    # # Устанавливаем шаг для оси X равным 20
    # plt.xticks(np.arange(min(x_results), max(x_results)+1, 10))

    # # Устанавливаем шаг для оси Y равным 20
    # plt.yticks(np.arange(min(y_results), max(y_results)+1, 10))

    # Создаем слайдер
    axslider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(axslider, 'TIME', valmin=t_results[0], valmax=t_results[-1], valinit=0)

    #Создаем текстовый блок со скоростью
    text_box = plt.text(0.5, -0.3, "Velocity = 0.00 m/s", transform=ax.transAxes, va="top", ha="center", fontweight="bold")

    def findClosest(list: list, target: float):
        """
        В этой функции min ищет элемент в list, который минимизирует значение функции key. 
        В данном случае, функция key вычисляет абсолютное значение разницы между x и target, 
        поэтому min вернет элемент, который находится ближе всего к target.
        """
        return min(list, key=lambda x:abs(x-target))

    # Обновляем график при изменении значения слайдера
    def update(val):
        max_t = findClosest(t_results, slider.val)
        index_of_max_t = t_results.index(max_t)
        x = x_results[:index_of_max_t]
        h = h_results[:index_of_max_t]
        l.set_xdata(x)
        l.set_ydata(h)
        l.set_visible(True)
        velocity = (v_x_results[index_of_max_t]**2 + v_y_results[index_of_max_t]**2)**(1/2)
        text_box.set_text(f"Velocity = {velocity:.2f} m/s")
        # ax.relim()
        # ax.autoscale_view()
        fig.canvas.draw_idle()

    slider.on_changed(update)

    #Для фиксации времени исполнения
    ic()

    plt.show()

def plotProccess(q: Queue, num_points: int = 1000, H_MS: float = d.H_MS_11):
    fig, ax = plt.subplots()
    ax: Axes

    # plt.gca().set_xlim([-2200, 2200])
    # plt.gca().set_ylim([-2200, 2200])
    ax.set_aspect("equal")
    ax.set(xlim=[-500, 1000], ylim=[1000, 2200])

    moon = Circle((0, 0), d.R_MOON, fill=True, color="grey")
    target_orbit = Circle((0, 0), d.R_MOON + H_MS, fill=False, color="red", linestyle="--")
    ax.add_patch(moon)
    ax.add_patch(target_orbit)
    # plt.show()
    flag = False
    while True:
        if not q.empty():
            results: list = q.get(block=False)
            #Массив равномерно прореженных индексов
            indices = np.linspace(0, len(results)-1, num_points, dtype=int)

            #Прорежение исходного списка результатов
            thined_results = [results[i] for i in indices]

            #Распаковка результатов
            t_results, x_results, y_results, v_x_results, v_y_results, m_results = zip(*thined_results)

            # Перенос стартовой СК на поверхность Луны, начало абсолютных координат в ц.м. Луны
            h_results = [y + d.R_MOON for y in y_results]
            ax.plot(x_results, h_results)
            plt.draw()
            flag = True
        if flag:
            plt.pause(3)
            # plt.ion()

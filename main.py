import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import sys

from icecream import ic
from functools import partial

import modules.data as d

from modules.integration import RKSolver, integrateFunction
from modules.math_model import MathModel
from modules.boundary_problem import BPSolver
from modules.optimization import LMSolver
from modules.results import createPlot, writeToExcel, parseToResults, plotProccess


def main(h_ms, thrust, file=None):
    if file:
        sys.stderr = open(file, "w")

    #Начальный вектор состояния
    initial_state_vector = np.array([d.X_0, d.Y_0, d.V_X_0, d.V_Y_0, d.M_0])
	
    #Инициализация математической модели
    math_model = MathModel(d.M_0, d.M_FUEL_MAX, d.W_EFF, thrust, d.DPITCH_DT, d.PITCH_2, d.T_0, d.T_1, d.T_2, d.T_3, d.T_END, d.T_VERTICAL, d.MU_MOON, d.R_MOON)
    
    #Решатель для интегрирования
    rk_solver = RKSolver(integrateFunction, math_model, h_ms, initial_state_vector, [d.T_1, d.T_2])

    #Решатель для краевой задачи
    # bp_solver = BPSolver(rk_solver, partial(plotProccess, H_MS = h_ms))
    bp_solver = BPSolver(rk_solver)
    # U_i, m = bp_solver.solve()
    # print("DPITCH_DT =", repr(U_i[0][0]), "\nPITCH_2 = ", repr(U_i[1][0]), "\nMASS = ", m)

    #Решатель для алгоритма Левенберга-Марквардта
    lm_solver = LMSolver(bp_solver, partial(plotProccess, H_MS = h_ms))
    # lm_solver = LMSolver(bp_solver)
    lm_solver.solve()

    results = parseToResults(*rk_solver.solve(stop_on_boundary=True))

    # #Запись результатов в Excel
    writeToExcel(results, d.EXCEL_FILE)

    # #Создание окна с графиком и слайдером
    createPlot(results, h_ms)

if __name__ == "__main__":
    main(d.H_MS_21, d.THRUST_2)
    
    # p1 = mp.Process(target=main, args=("tsk1.log", d.H_MS_11, d.THRUST_1))
    # p2 = mp.Process(target=main, args=("tsk2.log", d.H_MS_12, d.THRUST_1))
    # p3 = mp.Process(target=main, args=("tsk3.log", d.H_MS_21, d.THRUST_2))
    # p4 = mp.Process(target=main, args=("tsk4.log", d.H_MS_22, d.THRUST_2))

    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()

    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
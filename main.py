import numpy as np
import matplotlib.pyplot as plt

from icecream import ic

import modules.data as d

from modules.integration import myRK, integrateFunction
from modules.math_model import MathModel
from modules.boundary_problem import BoundaryProblem
from modules.optimization import LevenbergMarquardt
from modules.results import createPlot, writeToExcel, parseToResults, plotProccess

def main():
    #Начальный вектор состояния
    initial_state_vector = np.array([d.X_0, d.Y_0, d.V_X_0, d.V_Y_0, d.M_0])

    #Инициализация математической модели
    math_model = MathModel(d.M_0, d.M_FUEL_MAX, d.W_EFF, d.THRUST_1, d.DPITCH_DT, d.PITCH_2, d.T_0, d.T_1, d.T_2, d.T_3, d.T_END, d.T_VERTICAL, d.MU_MOON, d.R_MOON)
    
    #Решатель для интегрирования
    rk_solver = myRK(integrateFunction, math_model, d.H_MS_11, initial_state_vector, [d.T_1, d.T_2])

    #Решатель для краевой задачи
    bp_solver = BoundaryProblem(rk_solver, plotProccess)
    U_i, m = bp_solver.solve()
    print("DPITCH_DT =", repr(U_i[0][0]), "\nPITCH_2 = ", repr(U_i[1][0]), "\nMASS = ", m)

    # #Решатель для алгоритма Левенберга-Марквардта
    # lm_solver = LevenbergMarquardt(bp_solver, 0.1, d.C_1, d.C_2, d.ALPHA_0)
    # lm_solver.solve()

    results = parseToResults(*rk_solver.solve(stop_on_boundary=True))

    # #Запись результатов в Excel
    writeToExcel(results, d.EXCEL_FILE)

    # #Создание окна с графиком и слайдером
    createPlot(results, num_points=1000)

if __name__ == "__main__":
    main()
import numpy as np
import time
import multiprocessing as mp

from icecream import ic
from typing import Callable
from modules.integration import RKSolver
from modules.results import parseToResults
from modules.data import DEBUG

class PDMixin():
    @staticmethod
    def _partial_derivate(f2, f1, delta):
        return (f2 - f1) / (2*delta)

class BPSolver(PDMixin):
    def __init__(self, rk_solver: RKSolver, plot_generator: Callable = None) -> None:
        """
        :param initial_dpitch_dt: в радианах
        :param initial_pitch: в радианах
        """
        self._plot_generator = plot_generator
        self._rk_solver = rk_solver
        self.math_model = rk_solver.math_model
        if not DEBUG:
            ic.disable()

    def solve(self):
        """
        Решение краевой задачи
        :param dpitch_dt: приближенное значение для решения
        :param pitch_2: приближенное значение для решения
        """
        ic("BP ENTRY")
        if self._plot_generator:
            self._q = mp.Queue()
            self._p = mp.Process(target=self._plot_generator, args=(self._q,))
            self._p.start()

        delta_dpitch_dt = 10**(-8)
        delta_pitch2 = 10**(-5)

        r_vect_diff = 1
        flight_path_diff = 1

        U_i = np.array([[self.math_model.DPITCH_DT], [self.math_model.PITCH_2]])

        while True:
            dpitch_dt = U_i[0][0]
            pitch2 = U_i[1][0]
            
            t_values, state_vector_values = self._rk_solver.solve()
            results = parseToResults(t_values, state_vector_values)

            if self._plot_generator:
                if self._q:
                    self._q.put(results)

            r_vect_start, flight_path_start = self._get_boundary_params(state_vector_values)
    
            f_u = self._residual(r_vect_start, flight_path_start)
            r_vect_diff: float = f_u[0][0]
            flight_path_diff: float = f_u[1][0]

            #BP END CONDITION
            if not (abs(round(flight_path_diff, 6)) > 0 or abs(round(r_vect_diff, 6)) > 0):
                self._rk_solver.reset(dpitch_dt=U_i[0][0], pitch2=U_i[1][0])
                break

            self._rk_solver.reset(dpitch_dt=dpitch_dt - delta_dpitch_dt, pitch2=pitch2)
            t_values, state_vector_values = self._rk_solver.solve()
            r_vect_f1_dpitch_dt, flight_path_f1_dpitch_dt = self._get_boundary_params(state_vector_values)

            self._rk_solver.reset(dpitch_dt=dpitch_dt + delta_dpitch_dt, pitch2=pitch2)
            t_values, state_vector_values = self._rk_solver.solve()
            r_vect_f2_dpitch_dt, flight_path_f2_dpitch_dt = self._get_boundary_params(state_vector_values)

            self._rk_solver.reset(dpitch_dt=dpitch_dt, pitch2=pitch2 - delta_pitch2)
            t_values, state_vector_values = self._rk_solver.solve()
            r_vect_f1_pitch2, flight_path_f1_pitch2 = self._get_boundary_params(state_vector_values)

            self._rk_solver.reset(dpitch_dt=dpitch_dt, pitch2=pitch2 + delta_pitch2)
            t_values, state_vector_values = self._rk_solver.solve()
            r_vect_f2_pitch2, flight_path_f2_pitch2 = self._get_boundary_params(state_vector_values)

            dr_dpitchdt = self._partial_derivate(r_vect_f2_dpitch_dt, r_vect_f1_dpitch_dt, delta_dpitch_dt)
            dr_dpitch2 = self._partial_derivate(r_vect_f2_pitch2, r_vect_f1_pitch2, delta_pitch2)
            dfp_dpitchdt = self._partial_derivate(flight_path_f2_dpitch_dt, flight_path_f1_dpitch_dt, delta_dpitch_dt)
            dfp_dpitch2 = self._partial_derivate(flight_path_f2_pitch2, flight_path_f1_pitch2, delta_pitch2)            

            U_i = self._iteration(U_i, f_u, dr_dpitchdt, dr_dpitch2, dfp_dpitchdt, dfp_dpitch2)

            self._rk_solver.reset(dpitch_dt=U_i[0][0], pitch2=U_i[1][0])

        if self._plot_generator: 
            if self._p:
                self._p.terminate()
                self._p = None
                self._q = None
        mass = state_vector_values[-1][4]
        ic(U_i)
        ic(f_u[0][0])
        ic(f_u[1][0])
        return U_i, mass
    
    def _residual(self, r_vect_start: float, flight_path_start: float):
        math_model = self._rk_solver.math_model
        f_u = -np.array([
            [r_vect_start - (math_model._R_BODY + self._rk_solver._H_MS)],
            [flight_path_start]
        ])
        return f_u

    def _iteration(self, U_prev: np.ndarray, f_u: np.ndarray, df1_dx1: float, df1_dx2: float, df2_dx1: float, df2_dx2: float):
        jacobian = np.array([
            [df1_dx1, df1_dx2],
            [df2_dx1, df2_dx2]
        ])
        # ic(df1_dx1, df1_dx2, df2_dx1, df2_dx2)
        # ic(jacobian)
        inv_jacobian = np.linalg.inv(jacobian)

        U_i = U_prev + inv_jacobian @ f_u
        return U_i

    def _get_boundary_params(self, state_vector_values: list):
        last_state_values = state_vector_values[-1]
        x = last_state_values[0]
        y = last_state_values[1]
        v_x = last_state_values[2]
        v_y = last_state_values[3]

        radius_vector = self._rk_solver.math_model.getRadiusVectorValue(x, y)
        flight_path_angle = self._rk_solver.math_model.getFlightPathAngle(x, y, v_x, v_y)
        return radius_vector, flight_path_angle
            
    def reset(self, t_1, t_2):
        self._rk_solver.reset(t_1=t_1, t_2=t_2)
        self.__init__(self._rk_solver, self._plot_generator)

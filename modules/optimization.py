import numpy as np
import multiprocessing as mp



from icecream import ic
from typing import Callable
from modules.integration import myRK
from modules.boundary_problem import BoundaryProblem, PDMixin
from modules.results import parseToResults
from modules.data import DEBUG


class LevenbergMarquardt(PDMixin):
    _number_of_calc = 0
    def __init__(self, bp_solver: BoundaryProblem, accuracy, C_1, C_2, ALPHA_0):
        self._bp_solver = bp_solver
        self._rk_solver = bp_solver._rk_solver
        self.math_model = bp_solver.math_model
        self._T_1 = self.math_model.T_1
        self._T_2 = self.math_model.T_2
        self._epsilon = accuracy
        self._C_1 = C_1
        self._C_2 = C_2
        self._ALPHA_0 = ALPHA_0

    def solve(self):
        #TODO FIX IT
        dt1 = 0.01
        dt2 = 0.01

        self._bp_solver.solve()
        t_values, state_vector_values = self._rk_solver.solve()
        g_new = self.math_model.M_0 - state_vector_values[4][-1]
        
        # t1
        self._rk_solver.reset(t_1=self._T_1 - dt1, t_2=self._T_2)
        self._bp_solver.solve()
        t_values, state_vector_values = self._rk_solver.solve()
        g_grad_11 = self.math_model.M_0 - state_vector_values[4][-1]

        self._rk_solver.reset(t_1=self._T_1 + dt1, t_2=self._T_2)
        self._bp_solver.solve()
        t_values, state_vector_values = self._rk_solver.solve()
        g_grad_12 = self.math_model.M_0 - state_vector_values[4][-1]

        g_grad_t1 = self._partial_derivate(g_grad_12, g_grad_11, dt1)

        # t2
        self._rk_solver.reset(t_1=self._T_1, t_2=self._T_2 - dt2)
        self._bp_solver.solve()
        t_values, state_vector_values = self._rk_solver.solve()
        g_grad_21 = self.math_model.M_0 - state_vector_values[4][-1]

        self._rk_solver.reset(t_1=self._T_1, t_2=self._T_2 + dt2)
        self._bp_solver.solve()
        t_values, state_vector_values = self._rk_solver.solve()
        g_grad_22 = self.math_model.M_0 - state_vector_values[4][-1]

        g_grad_t2 = self._partial_derivate(g_grad_22, g_grad_21, dt2)

        grad_g = np.array([[g_grad_t1], [g_grad_t2]])

        # matrixes for iteration formula
        I = np.eye(2)

        X_new = np.array([[T_1], [T_2]])

        ic("FIRST ENTRY")
        while abs(np.sqrt(grad_g[0][0]**2 + grad_g[0][1]**2)) >= self._epsilon:
            T_1 = X_new[0][0]
            T_2 = X_new[1][0]

            g = g_new

            # ///// mixed derivatives /////
            # f1 = f5
            U_i_1 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, T_1 - dt1, T_2 - dt2)
            Vx, Vy, x, y, m, t = integration(U_i_1[0][0], U_i_1[1][0], ind.dt, T_1 - dt1, T_2 - dt2)
            g_1 = ind.m_0 - m

            # f2 = f7
            U_i_2 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, T_1 - dt1, T_2 + dt2)
            Vx, Vy, x, y, m, t = integration(U_i_2[0][0], U_i_2[1][0], ind.dt, T_1 - dt1, T_2 + dt2)
            g_2 = ind.m_0 - m

            # f3 = f6
            U_i_3 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, T_1 + dt1, T_2 - dt2)
            Vx, Vy, x, y, m, t = integration(U_i_3[0][0], U_i_3[1][0], ind.dt, T_1 + dt1, T_2 - dt2)
            g_3 = ind.m_0 - m

            # f4 = f8
            U_i_4 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, T_1 + dt1, T_2 + dt2)
            Vx, Vy, x, y, m, t = integration(U_i_4[0][0], U_i_4[1][0], ind.dt, T_1 + dt1, T_2 + dt2)
            g_4 = ind.m_0 - m

            # put it together for t1t2
            f1_dt1_der = of.partial_derivative(g_2, g_1, dt2)
            f2_dt1_der = of.partial_derivative(g_4, g_3, dt2)
            f_t1_t2_der = of.partial_derivative(f2_dt1_der, f1_dt1_der, dt1)

            # put it together for t2t1
            f1_dt2_der = of.partial_derivative(g_3, g_1, dt1)
            f2_dt2_der = of.partial_derivative(g_4, g_2, dt1)
            f_t2_t1_der = of.partial_derivative(f2_dt2_der, f1_dt2_der, dt2)

            print('25%')

            # ///// total derivatives /////

            # for t_1
            # f11
            U_i_11 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, T_1 - dt1 - dt2, T_2)
            Vx, Vy, x, y, m, t = integration(U_i_11[0][0], U_i_11[1][0], ind.dt, T_1 - dt1 - dt2, T_2)
            g_11 = ind.m_0 - m

            # f12
            U_i_12 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, T_1 - dt1 + dt2, T_2)
            Vx, Vy, x, y, m, t = integration(U_i_12[0][0], U_i_12[1][0], ind.dt, T_1 - dt1 + dt2, T_2)
            g_12 = ind.m_0 - m

            # f13
            U_i_13 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, T_1 + dt1 - dt2, T_2)
            Vx, Vy, x, y, m, t = integration(U_i_13[0][0], U_i_13[1][0], ind.dt, T_1 + dt1 - dt2, T_2)
            g_13 = ind.m_0 - m

            # f14
            U_i_14 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, T_1 + dt1 + dt2, T_2)
            Vx, Vy, x, y, m, t = integration(U_i_14[0][0], U_i_14[1][0], ind.dt, T_1 + dt1 + dt2, T_2)
            g_14 = ind.m_0 - m

            # put it together for t1**2
            f11_dt1_der = of.partial_derivative(g_12, g_11, dt2)
            f12_dt1_der = of.partial_derivative(g_14, g_13, dt2)
            f_t1_t1_der = of.partial_derivative(f12_dt1_der, f11_dt1_der, dt1)

            # for t_2
            # f21
            U_i_21 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, T_1, T_2 - dt2 - dt1)
            Vx, Vy, x, y, m, t = integration(U_i_21[0][0], U_i_21[1][0], ind.dt, T_1, T_2 - dt2 - dt1)
            g_21 = ind.m_0 - m

            # f22
            U_i_22 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, T_1, T_2 - dt2 + dt1)
            Vx, Vy, x, y, m, t = integration(U_i_22[0][0], U_i_22[1][0], ind.dt, T_1, T_2 - dt2 + dt1)
            g_22 = ind.m_0 - m

            # f23
            U_i_23 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, T_1, T_2 + dt2 - dt1)
            Vx, Vy, x, y, m, t = integration(U_i_23[0][0], U_i_23[1][0], ind.dt, T_1, T_2 + dt2 - dt1)
            g_23 = ind.m_0 - m  #delta_m

            # f24
            U_i_24 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, T_1, T_2 + dt2 + dt1)
            Vx, Vy, x, y, m, t = integration(U_i_24[0][0], U_i_24[1][0], ind.dt, T_1, T_2 + dt2 + dt1)
            g_24 = ind.m_0 - m

            # put it together for t2**2
            f21_dt2_der = of.partial_derivative(g_22, g_21, dt1)
            f22_dt2_der = of.partial_derivative(g_24, g_23, dt1)
            f_t2_t2_der = of.partial_derivative(f22_dt2_der, f21_dt2_der, dt2)

            # ///// HESSIAN /////
            H_i = np.array([f_t1_t1_der, f_t1_t2_der, f_t2_t1_der, f_t2_t2_der])
            H_i.shape = (2, 2)
            print(H_i)
            print('50%')

            # iteration formula right here
            X_new = of.iteration_formula_LM(X_new, H_i, alpha_i, I, grad_g)
            # debug
            print(X_new)
            Vx, Vy, x, y, m, t = integration(U_i_new[0][0], U_i_new[1][0], ind.dt, X_new[0][0], X_new[1][0], True)
            print('g = ', ind.m_0 - m)
            print('Vx = ', Vx, '\nVy = ', Vy, '\nx = ', x, '\ny = ', y, '\nm = ', m, '\nt = ', t)
            message = hp.visualisation('MS movement')
            print(message)

            # recalculating g
            '''U_i_pre_new = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, X_pre_new[0][0], X_pre_new[1][0])
            Vx, Vy, x, y, m, t = integration(U_i_pre_new[0][0], U_i_pre_new[1][0], ind.dt, X_pre_new[0][0], X_pre_new[1][0])
            g_new = ind.m_0 - m'''

            U_i_new = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, X_new[0][0], X_new[1][0])
            Vx, Vy, x, y, m, t = integration(U_i_new[0][0], U_i_new[1][0], ind.dt, X_new[0][0], X_new[1][0])
            g_new = ind.m_0 - m

            # checking g
            if g_new < g:
                alpha_i *= C_1
                '''U_i_new = U_i_pre_new
                X_new = X_pre_new'''
            else:
                '''X_new_while = X_new
                U_i_new_while = U_i_new'''
                while g_new > g:
                    alpha_i *= C_2
                    X_new = of.iteration_formula_LM(X_new, H_i, alpha_i, I, grad_g)

                    U_i_new = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, X_new[0][0], X_new[1][0])
                    Vx, Vy, x, y, m, t = integration(U_i_new[0][0], U_i_new[1][0], ind.dt, X_new[0][0], X_new[1][0])
                    g_new = ind.m_0 - m
                '''X_new = X_new_while
                U_i_new = U_i_new_while'''
                '''if round(g_new, 3) != round(g, 3):
                    alpha_i *= ind.C_1'''

            print('80%')

            # recalculating grad(g)
            # t1
            U_i_recalc_grad_11 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, X_new[0][0] - dt1, X_new[1][0])
            Vx, Vy, x, y, m, t = integration(U_i_recalc_grad_11[0][0], U_i_recalc_grad_11[1][0], ind.dt, X_new[0][0] - dt1, X_new[1][0])
            g_recalc_grad_11 = ind.m_0 - m

            U_i_recalc_grad_12 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, X_new[0][0] + dt1, X_new[1][0])
            Vx, Vy, x, y, m, t = integration(U_i_recalc_grad_12[0][0], U_i_recalc_grad_12[1][0], ind.dt, X_new[0][0] + dt1, X_new[1][0])
            g_recalc_grad_12 = ind.m_0 - m

            g_recalc_grad_t1 = of.partial_derivative(g_recalc_grad_12, g_recalc_grad_11, dt1)

            # t2
            U_i_recalc_grad_21 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, X_new[0][0], X_new[1][0] - dt2)
            Vx, Vy, x, y, m, t = integration(U_i_recalc_grad_21[0][0], U_i_recalc_grad_21[1][0], ind.dt, X_new[0][0], X_new[1][0] - dt2)
            g_recalc_grad_21 = ind.m_0 - m

            U_i_recalc_grad_22 = boundary_problem(U_i_new[0][0], U_i_new[1][0], ind.dt, X_new[0][0], X_new[1][0] + dt2)
            Vx, Vy, x, y, m, t = integration(U_i_recalc_grad_22[0][0], U_i_recalc_grad_22[1][0], ind.dt, X_new[0][0], X_new[1][0] + dt2)
            g_recalc_grad_22 = ind.m_0 - m

            g_recalc_grad_t2 = of.partial_derivative(g_recalc_grad_22, g_recalc_grad_21, dt2)

            grad_g = np.array([g_recalc_grad_t1, g_recalc_grad_t2])
            grad_g.shape = (2, 1)
            print(grad_g)

            iterations += 1

            print('100%')
            print('Iterations =', iterations)

        return X_new

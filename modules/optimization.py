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

    @staticmethod
    def _iteration(X_i, H_i, alpha_i, grad_g):
        return X_i - np.linalg.inv(H_i + alpha_i * np.eye(2)) @ grad_g

    def _grad(self, t_1, t_2):
        ic("LM GRAD ENTRY")
        dt1 = 0.01
        dt2 = 0.01
        M_0 = self.math_model._M_0

        self._bp_solver.reset(t_1 - dt1, t_2)
        U_i, m11 = self._bp_solver.solve()
        ic("LM GRAD 1 BP SOLVED")

        self._bp_solver.reset(t_1 + dt1, t_2)
        U_i, m12 = self._bp_solver.solve()
        ic("LM GRAD 2 BP SOLVED")

        self._bp_solver.reset(t_1, t_2 - dt2)
        U_i, m21 = self._bp_solver.solve()
        ic("LM GRAD 3 BP SOLVED")

        self._bp_solver.reset(t_1, t_2 + dt2)
        U_i, m22 = self._bp_solver.solve()
        ic("LM GRAD 4 BP SOLVED")


        grad1 = self._partial_derivate(M_0 - m12, M_0 - m11, dt1)
        grad2 = self._partial_derivate(M_0 - m22, M_0 - m21, dt2)

        grad_g_fun = np.array([
            [grad1], 
            [grad2]
        ])
        return grad_g_fun
    
    def _hessian(self, t_1, t_2):
        ic("LM HESSIAN ENTRY")
        dt1 = 0.01
        dt2 = 0.01
        M_0 = self.math_model._M_0


        # dt1dt2
        self._bp_solver.reset(t_1 - dt1, t_2 - 2 * dt2)
        U_i, m11 = self._bp_solver.solve()
        ic("LM HESSIAN 1 BP SOLVED")

        self._bp_solver.reset(t_1 + dt1, t_2 - 2 * dt2)
        U_i, m12 = self._bp_solver.solve()
        ic("LM HESSIAN 2 BP SOLVED")

        self._bp_solver.reset(t_1 - dt1, t_2 + 2 * dt2)
        U_i, m21 = self._bp_solver.solve()
        ic("LM HESSIAN 3 BP SOLVED")

        self._bp_solver.reset(t_1 + dt1, t_2 + 2 * dt2)
        U_i, m22 = self._bp_solver.solve()
        ic("LM HESSIAN 4 BP SOLVED")

        f1 = self._partial_derivate(M_0 - m12, M_0 - m11, dt1)
        f2 = self._partial_derivate(M_0 - m22, M_0 - m21, dt1)
        
        dt1dt2 = self._partial_derivate(f2, f1, dt2)


        # dt1dt1
        self._bp_solver.reset(t_1 + dt1 + dt2, t_2)
        U_i, m11 = self._bp_solver.solve()
        ic("LM HESSIAN 5 BP SOLVED")
        
        self._bp_solver.reset(t_1 + dt1 - dt2, t_2)
        U_i, m12 = self._bp_solver.solve()
        ic("LM HESSIAN 6 BP SOLVED")

        self._bp_solver.reset(t_1 - dt1 + dt2, t_2)
        U_i, m21 = self._bp_solver.solve()
        ic("LM HESSIAN 7 BP SOLVED")
        
        self._bp_solver.reset(t_1 - dt1 - dt2, t_2)
        U_i, m22 = self._bp_solver.solve()
        ic("LM HESSIAN 8 BP SOLVED")
        
        f1 = self._partial_derivate(M_0 - m12, M_0 - m11, dt1)
        f2 = self._partial_derivate(M_0 - m22, M_0 - m21, dt1)
        
        dt1dt1 = self._partial_derivate(f2, f1, dt2)

        
        # dt2dt2        
        self._bp_solver.reset(t_1, t_2 + dt1 + dt2)
        U_i, m11 = self._bp_solver.solve()
        ic("LM HESSIAN 9 BP SOLVED")
        
        self._bp_solver.reset(t_1, t_2 + dt1 - dt2)
        U_i, m12 = self._bp_solver.solve()
        ic("LM HESSIAN 10 BP SOLVED")

        self._bp_solver.reset(t_1, t_2 - dt1 + dt2)
        U_i, m21 = self._bp_solver.solve()
        ic("LM HESSIAN 11 BP SOLVED")
        
        self._bp_solver.reset(t_1, t_2 - dt1 - dt2)
        U_i, m22 = self._bp_solver.solve()
        ic("LM HESSIAN 12 BP SOLVED")
        
        f1 = self._partial_derivate(M_0 - m12, M_0 - m11, dt1)
        f2 = self._partial_derivate(M_0 - m22, M_0 - m21, dt1)
        
        dt2dt2 = self._partial_derivate(f2, f1, dt2)

        hessian = np.array([
            [dt1dt1, dt1dt2],
            [dt1dt2, dt2dt2]
        ])    
        return hessian

    def solve(self):
        ic("LM ENTRY")
        accuracy = 0.1
        alpha_i = 1000
        
        M_0 = self.math_model._M_0
        bp_solver = self._bp_solver

        X_i = np.array([
            [self._T_1],
            [self._T_2]
        ])

        #Расчет минимизируемой функции.
        U_i, m = bp_solver.solve()
        
        g_fun = M_0 - m

        error_list = []
        m_list = []
        
        while True:
            ic("LM BIG ITERATION")
            m_list.append(g_fun)
            ic(m_list)

            #Расчет градиента минимизируемой функции
            grad_g_fun = self._grad(X_i[0][0], X_i[1][0])
            ic(grad_g_fun)

            error = np.linalg.norm(grad_g_fun)
            error_list.append(error)
            ic(error_list)
            if error < accuracy:
                ic("LM SOLVED")
                X_min = X_i
                g_fun_min = g_fun
                bp_solver.reset(X_i[0][0], X_i[1][0])
                return X_min, g_fun_min
            
            H_i = self._hessian(X_i[0][0], X_i[1][0])
            ic(H_i)

            lm_small_iter_count = 0
            while True:
                ic("LM SMALL ITERATION ENTRY")
                ic(alpha_i)
                lm_small_iter_count += 1
                
                #SMALL_ITER
                #X_i+1
                X_i = self._iteration(X_i, H_i, alpha_i, grad_g_fun)
                ic(X_i)

                #Расчет новой минимизируемой функции.
                bp_solver.reset(X_i[0][0], X_i[1][0])
                U_i, m = bp_solver.solve()
                ic("LM SMALL ITTERATION BP SOLVED")
                g_fun_new = M_0 - m

                if lm_small_iter_count > 50:
                    ic(g_fun_new)
                    ic(g_fun_min)
                    raise Exception("Давай по новой.")

                if g_fun_new < g_fun:
                    #alpha_i+1
                    alpha_i *= self._C_1
                    g_fun = g_fun_new
                    break
                else:
                    alpha_i *= self._C_2

import numpy as np
import multiprocessing as mp



from icecream import ic
from typing import Callable
from modules.integration import RKSolver
from modules.boundary_problem import BPSolver, PDMixin
from modules.results import parseToResults
from modules.data import DEBUG


class LMSolver(PDMixin):
    _number_of_calc = 0
    def __init__(self, bp_solver: BPSolver, plot_generator: Callable = None):
        self._bp_solver = bp_solver
        self._rk_solver = bp_solver._rk_solver
        self._plot_generator = plot_generator
        self.math_model = bp_solver.math_model
        self._T_1 = self.math_model.T_1
        self._T_2 = self.math_model.T_2
        self._C_1 = 0.1
        self._C_2 = 10

    @staticmethod
    def _iteration(X_i, H_i, alpha_i, grad_g):
        addition = np.linalg.inv(H_i + alpha_i * np.eye(2)) @ grad_g
        ic(addition)
        return X_i - addition
        # return X_i - np.linalg.inv(H_i + alpha_i * np.eye(2)) @ grad_g

    def _grad(self, t_1, t_2):
        ic("LM GRAD ENTRY")
        dt1 = 0.1
        dt2 = 0.1
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
        dt = 0.1
        M_0 = self.math_model._M_0

        #bps
        self._bp_solver.reset(t_1 - dt, t_2 - dt)
        U_i, m_t1mdt_t2mdt = self._bp_solver.solve()
        ic("LM HESSIAN 1 BP SOLVED")
        
        self._bp_solver.reset(t_1 + dt, t_2 - dt)
        U_i, m_t1pdt_t2mdt = self._bp_solver.solve()
        ic("LM HESSIAN 2 BP SOLVED")

        self._bp_solver.reset(t_1 - dt, t_2 + dt)
        U_i, m_t1mdt_t2pdt = self._bp_solver.solve()
        ic("LM HESSIAN 3 BP SOLVED")

        self._bp_solver.reset(t_1 + dt, t_2 + dt)
        U_i, m_t1pdt_t2pdt = self._bp_solver.solve()
        ic("LM HESSIAN 4 BP SOLVED")

        self._bp_solver.reset(t_1, t_2 - dt)
        U_i, m_t1_t2mdt = self._bp_solver.solve()
        ic("LM HESSIAN 5 BP SOLVED")

        self._bp_solver.reset(t_1, t_2 + dt)
        U_i, m_t1_t2pdt = self._bp_solver.solve()
        ic("LM HESSIAN 6 BP SOLVED")

        self._bp_solver.reset(t_1 - dt, t_2)
        U_i, m_t1mdt_t2 = self._bp_solver.solve()
        ic("LM HESSIAN 7 BP SOLVED")

        self._bp_solver.reset(t_1 + dt, t_2)
        U_i, m_t1pdt_t2 = self._bp_solver.solve()
        ic("LM HESSIAN 8 BP SOLVED")

        self._bp_solver.reset(t_1, t_2)
        U_i, m_t1_t2 = self._bp_solver.solve()
        ic("LM HESSIAN 9 BP SOLVED")

        dt1dt1 = ((M_0 - m_t1pdt_t2) - 2*(M_0 - m_t1_t2) + (M_0 - m_t1mdt_t2)) / dt**2
        dt2dt2 = ((M_0 - m_t1_t2pdt) - 2*(M_0 - m_t1_t2) + (M_0 - m_t1_t2mdt)) / dt**2
        dt1dt2 = ((M_0 - m_t1pdt_t2pdt) - (M_0 - m_t1mdt_t2pdt) - (M_0 - m_t1pdt_t2mdt) + (M_0 - m_t1mdt_t2mdt)) / (4 * dt**2)

        hessian = np.array([
            [dt1dt1, dt1dt2],
            [dt1dt2, dt2dt2]
        ])

        return hessian
    
    def solve(self):
        ic("LM ENTRY")
        accuracy = 0.01
        alpha_i = 0.001
        
        M_0 = self.math_model._M_0
        bp_solver = self._bp_solver

        X_i = np.array([
            [self._T_1],
            [self._T_2]
        ])

        if self._plot_generator:
            self._q = mp.Queue()
            self._p = mp.Process(target=self._plot_generator, args=(self._q,))
            self._p.start()

        #Расчет минимизируемой функции.
        U_i, m = bp_solver.solve()

        if self._plot_generator:
            if self._q:
                t_values, state_vector_values = self._rk_solver.solve()
                results = parseToResults(t_values, state_vector_values)
                self._q.put(results)

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

                if self._plot_generator: 
                    if self._p:
                        self._p.terminate()
                        self._p = None
                        self._q = None
                return X_min, g_fun_min
            
            H_i = self._hessian(X_i[0][0], X_i[1][0])
            ic(H_i)
            
  
            while True:
                ic("LM ITERATION FORMULA ENTRY")
                ic(alpha_i)
                X_i = self._iteration(X_i, H_i, alpha_i, grad_g_fun)
                ic(X_i)

                #Расчет новой минимизируемой функции.
                bp_solver.reset(X_i[0][0], X_i[1][0])   
                U_i, m = bp_solver.solve()

                if self._plot_generator:
                    if self._q:
                        t_values, state_vector_values = self._rk_solver.solve()
                        results = parseToResults(t_values, state_vector_values)
                        self._q.put(results)

                ic("LM ITTERATION FORMULA BP SOLVED")
                g_fun_new = M_0 - m


                if g_fun_new < g_fun:
                    alpha_i *= self._C_1
                    g_fun = g_fun_new
                    break
                else:
                    alpha_i *= self._C_2
                

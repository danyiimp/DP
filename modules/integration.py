import numpy as np

from icecream import ic
from typing import Callable
from functools import partial

from modules.math_model import MathModel
import modules.data as d


def integrateFunction(math_model: MathModel) -> Callable:
    """
    Функция, которая передает параметр math_model в функцию realIntegrateFunction,
    и возвращает ее же, только с параметром math_model внутри
    """
    def realIntegrateFunction(t: float, state_vector: np.ndarray, math_model: MathModel) -> np.ndarray:
        """
        Функция, используемая классом RK для интегрирования.
        """
        new_state_vector = np.zeros((5,))
        new_state_vector[0] = math_model.dX_dT(state_vector[2])
        new_state_vector[1] = math_model.dY_dT(state_vector[3])
        new_state_vector[2] = math_model.dVx_dT(t, state_vector[0], state_vector[1], state_vector[4])
        new_state_vector[3] = math_model.dVy_dT(t, state_vector[0], state_vector[1], state_vector[4])
        new_state_vector[4] = math_model.dM_dT(t)
        return new_state_vector
    return partial(realIntegrateFunction, math_model=math_model)

class RKSolver():
    #Количество решений этим методом. Для нейминга progressBar
    _number_of_calc = 0
    def __init__(self, fun, math_model: MathModel, H_MS: float, y0, critical_times: list, t0=d.T_0, t_bound=d.T_END, dt=d.DELTA_T):
        #Параметры для реинициализации
        self.__reset_fun = fun
        self.__reset_H_MS = H_MS
        self.__reset_t0 = t0
        self.__reset_y0 = y0
        self.__reset_critical_times = critical_times
        self.__reset_t_bound = t_bound
        self.__dt = dt

        fun = fun(math_model)
        self.fun = fun
        self.math_model = math_model
        self.initial_dt = dt
        self.dt = dt
        self._H_MS = H_MS
        self.critical_times = critical_times

        self.t = t0
        self.y_old = y0
        self.y = y0
        self.t_bound = t_bound
        self._step_back_allowed = False
        self.status = "running"

    def step(self):
        self.y_old = self.y

        k1 = self.fun(self.t, self.y_old)
        k2 = self.fun(self.t + self.dt/2, self.y_old + self.dt/2 * k1)
        k3 = self.fun(self.t + self.dt/2, self.y_old + self.dt/2 * k2)
        k4 = self.fun(self.t + self.dt, self.y_old + self.dt * k3)

        self.y = self.y_old + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.t += self.dt 

        self.dt_prev = self.dt
        self._step_back_allowed = True

        if np.isclose(self.t, self.t_bound, atol=1e-8):
            self.status = "finished"

    def _progressBar(self, current_step: int, total_steps: int, update_step_count=1000):
        """
        Вообще это универсальный метод, но чтобы не плодить сущности, является методом RK
        """
        return
        if current_step % update_step_count != 0 and current_step != total_steps and current_step != 1:
            return
        progress = current_step / total_steps
        bar_length = 20 
        block = int(round(bar_length * progress))
        text = "\rRK №{0} progress: [{1}] {2:.1f}%".format(
            str(RKSolver._number_of_calc).zfill(2),
            "#" * block + "-" * (bar_length - block), 
            progress * 100
        )
        
        print(text, end='')
        if progress == 1:
            print()

    def step_reduce(self, value_of_reduction: float = 10):
        self.dt /= value_of_reduction

    def step_back(self):
        if self._step_back_allowed:
            self.y = self.y_old
            self.t = self.t - self.dt_prev
            self._step_back_allowed = False
        else:
            raise Exception("Step back is not allowed")

    def set_step(self, step_size: float):
        self.dt = step_size

    def solve(self, accuracy: float = 10**(-8), stop_on_boundary: bool = True):
        """
        :param stop_on_boundary: остановить интегрирование при достижении граничных условий
        """
        RKSolver._number_of_calc += 1
        
        #Инициализация списков для хранения результатов
        t_values = []
        state_vector_values = []
        cr_times = iter(self.critical_times)
        cr_t = next(cr_times, None)
        
        while self.status == "running":
            #Реализация уменьшения шага для повышения точности конечной скорости
            if not self.math_model.off_thrust:
                stepped_over, deviation = self.math_model.endCondition(self._H_MS, self.y[2], self.y[3], self.t)
                
                if deviation < accuracy:
                    if stop_on_boundary:
                        self.status == "finished"
                        self._progressBar(1, 1)
                        break

                    #Возврат на норм шаг
                    step_to_initial_step = self.initial_dt - self.t % self.initial_dt
                    self.set_step(step_to_initial_step)
                    self.step()
                    t_values.append(self.t)
                    state_vector_values.append(self.y)

                    self.set_step(self.initial_dt)

                    self.math_model.off_thrust = True
                    print()
                    ic(self.math_model.off_thrust, self.t)
                    continue

                if stepped_over:
                    self.step_back()
                    self.step_reduce()
                    t_values.pop()
                    state_vector_values.pop()
                    continue

            if cr_t:
                step_to_cr_t = cr_t - self.t
                if not np.isclose(self.t, cr_t) and step_to_cr_t < self.dt:
                    self.set_step(step_to_cr_t)
                    self.step()
                    t_values.append(self.t)
                    state_vector_values.append(self.y)

                    step_to_initial_step = self.initial_dt - self.t % self.initial_dt
                    self.set_step(step_to_initial_step)
                    self.step()
                    t_values.append(self.t)
                    state_vector_values.append(self.y)

                    self.set_step(self.initial_dt)
                    cr_t = next(cr_times, None)
                    continue
            
            self.step()

            t_values.append(self.t)
            state_vector_values.append(self.y)

            self._progressBar(int(self.t / self.dt), int(self.t_bound / self.dt))  
        return t_values, state_vector_values
    
    def reset(self, dpitch_dt: float = None, pitch2: float = None, t_1: float = None, t_2: float = None):
        math_model = self.math_model
        
        #Установка новых параметров в мат модель
        if t_1 is not None and t_2 is not None:
            math_model.T_1 = t_1
            math_model.T_2 = t_2
            self.__reset_critical_times = [t_1, t_2]

        #Установка новых параметров в мат модель
        if dpitch_dt is not None and pitch2 is not None:
            math_model.DPITCH_DT = dpitch_dt
            math_model.PITCH_2 = pitch2

        self.__init__(self.__reset_fun, math_model, self.__reset_H_MS, self.__reset_y0, self.__reset_critical_times, self.__reset_t0, self.__reset_t_bound, self.__dt)
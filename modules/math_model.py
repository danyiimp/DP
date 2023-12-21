from numpy import pi, arcsin, cos, sin
from icecream import ic

import modules.data as d

class MathModel():
    def __init__(self, M_0: float, M_FUEL_MAX: float, W_EFF: float, THRUST: float, DPITCH_DT: float, PITCH_2: float, T_0: float, T_1: float, T_2: float, T_3: float, T_END: float, T_VERTICAL: d.T_VERTICAL, MU_BODY: d.MU_MOON, R_BODY: d.R_MOON) -> None:
        self._M_0 = M_0
        self._M_FUEL_MAX = M_FUEL_MAX
        
        self._W_EFF = W_EFF
        
        self._THRUST = THRUST
        self._DPITCH_DT = DPITCH_DT
        self._PITCH_2 = PITCH_2
        
        self._T_0 = T_0
        self._T_VERTICAL = T_VERTICAL
        self._T_1 = T_1
        self._T_2 = T_2
        self._T_3 = T_3
        self._T_END = T_END

        self._MU_BODY = MU_BODY
        self._R_BODY = R_BODY 

        self.off_thrust = False

    """
    Геттеры и сеттеры
    """

    @property
    def T_1(self) -> float:
        return self._T_1
    
    @T_1.setter
    def T_1(self, value: float) -> None:
        self._T_1 = value

    @property
    def T_2(self) -> float:
        return self._T_2
    
    @T_2.setter
    def T_2(self, value: float) -> None:
        self._T_2 = value

    @property
    def DPITCH_DT(self) -> float:
        return self._DPITCH_DT
    
    @DPITCH_DT.setter
    def DPITCH_DT(self, value: float) -> None:
        self._DPITCH_DT = value

    @property
    def PITCH_2(self) -> float:
        return self._PITCH_2
    
    @PITCH_2.setter
    def PITCH_2(self, value: float) -> None:
        self._PITCH_2 = value

    # @property
    # def off_thrust(self) -> bool | None:
    #     return self.off_thrust

    # @off_thrust.setter
    # def off_thrust(self, value: bool) -> None:
    #     self.off_thrust = value

    """
    Математическая модель
    """

    def getBodyGA_X(self, x: float, y: float) -> float:
        """
        :param x: координата по оси X
        :param y: координата по оси Y
        :return: проекция ускорения свободного падения на поверхности Луны на ось X стартовой СК
        """
        return -self._MU_BODY * x / (x**2 + (self._R_BODY + y)**2)**(3/2)

    def getBodyGA_Y(self, x: float, y: float) -> float:
        """
        :param x: координата по оси X
        :param y: координата по оси Y
        :return: проекция ускорения свободного падения на поверхности Луны на ось Y стартовой СК
        """
        return -self._MU_BODY * (self._R_BODY + y) / (x**2 + (self._R_BODY + y)**2)**(3/2)

    def getPitch(self, t: float) -> float:
        """
        :param t: текущее время
        :return: угол тангажа в радианах
        """
        if 0 <= t <= self._T_VERTICAL:
            return pi / 2
        elif self._T_VERTICAL < t <= self._T_1:
            return pi / 2 + self._DPITCH_DT * (t - self._T_VERTICAL)
        elif self._T_2 <= t <= self._T_END:
            return self._PITCH_2
        else:
            return 0
        
    def getThrust(self, t: float) -> float:
        """
        :param t: текущее время
        :return: тяга двигателя
        """
        if self.off_thrust:
            return 0
        if self._T_0 <= t <= self._T_1 or self._T_2 < t <= self._T_3:
            return self._THRUST
        else:
            return 0
        
    def getRadiusVectorValue(self, x: float, y: float) -> float:
        return (x**2 + (y + self._R_BODY)**2)**(1/2)

    def getFlightPathAngle(self, x: float, y: float, v_x: float, v_y: float) -> float:
        """
        Для граничного условия, flightPathAngle(t_k) = 0
        где t_k время окончания работы ДУ после 2-го запуска
        :return: угол наклона траектории
        """
        velocity_value = (v_x**2 + v_y**2)**(1/2) 
        r = self.getRadiusVectorValue(x, y)

        return arcsin((x * v_x + (y + self._R_BODY) * v_y) / (r * velocity_value))
    
    def getMaxTimeOfThrust(self) -> float:
        """
        :return: максимальное время работы ДУ
        """
        return self._M_FUEL_MAX / (self._THRUST / self._W_EFF)

    def getV_MS(self, H_MS: float) -> float:
        """
        :param H_MS: высота круговой целевой орбиты
        :return: скорость на круговой целевой орбите в км/c
        """
        R_MS = self._R_BODY + H_MS
        return (self._MU_BODY / R_MS)**(1/2)
    
    def endCondition(self, H_MS: float, v_x: float, v_y: float, t: float) -> tuple:
        """
        :return: True если конец траектории, False если нет
        """
        V_MS = self.getV_MS(H_MS)
        velocity_value = (v_x**2 + v_y**2)**(1/2)       
        if velocity_value < V_MS:
            return (False, abs(V_MS - velocity_value))
        else:
            return (True, abs(V_MS - velocity_value))

    def dM_dT(self, t: float) -> float:
        """
        :param t: текущее время
        :return: изменение массы
        """
        return -self.getThrust(t) / self._W_EFF

        
    """
    Уравнения движения
    """

    def dVx_dT(self, t: float, x: float, y: float, m: float) -> float:
        """
        :param x: координата по оси X
        :param y: координата по оси Y
        :param m: текущая масса
        :param t: текущее время
        :return: производная скорости на оси X по времени
        """
        return self.getThrust(t) * cos(self.getPitch(t)) / m + self.getBodyGA_X(x, y)

    def dVy_dT(self, t: float, x: float, y: float, m: float) -> float:
        """
        :param x: координата по оси X
        :param y: координата по оси Y
        :param m: текущая масса
        :param t: текущее время
        :return: производная скорости по оси Y по времени
        """
        return self.getThrust(t) * sin(self.getPitch(t)) / m + self.getBodyGA_Y(x, y)

    def dX_dT(self, v_x: float):
        return v_x

    def dY_dT(self, v_y: float):
        return v_y

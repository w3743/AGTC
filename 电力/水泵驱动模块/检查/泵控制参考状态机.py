"""水泵控制状态参考模型；仅供要求澄清与单元测试，不是量产固件。

泵的精确型号、允许的BUS幅值PWM、堵转电流和制动参数尚未确认，故运行资格仍关闭。
模型中的22.5–37.8 V是9S理论范围；制动阈值仅由测试夹具注入。5.19 A名义斩波值
是候选点，不作为安全上限，也不被状态机用于限制制动电流。

运行：python -X utf8 电力/水泵驱动模块/检查/泵控制参考状态机.py
"""

from dataclasses import dataclass
from enum import Enum
import unittest


class State(str, Enum):
    RESET_COAST = "复位默认滑行"
    SLEEP = "休眠/滑行"
    DRIVE = "单向PWM驱动"
    BRAKE = "高侧短接制动"
    FAULT_COAST = "锁存故障/滑行"


class IPropiDiag(str, Enum):
    # 由上层ADC有效性/标定检查提供；此模型不臆造原始电压阈值。
    OK = "正常"
    OPEN = "开路"
    SHORT = "短路"
    SATURATED = "饱和"
    INVALID = "未通过有效性检查"


@dataclass(frozen=True)
class Config:
    # 这里只采用项目给定的9S理论边界；整车/BMS实际放行边界及测量裕量未定。
    bus_min_v: float = 22.5
    bus_max_v: float = 37.8
    max_average_voltage_v: float = 21.0
    duty_ceiling: float = 0.95
    pwm_hz: int = 20_000
    # 必须由实泵实测/原厂资料给定，不提供产品默认值。
    brake_done_current_a: float = 0.0
    brake_done_dwell_s: float = 0.0


@dataclass(frozen=True)
class DriverPins:
    n_sleep: int
    in1: int
    in2: int | str
    output_mode: str


class PumpControllerModel:
    def __init__(self, config: Config):
        if config.brake_done_current_a <= 0 or config.brake_done_dwell_s <= 0:
            raise ValueError("制动电流阈值和持续时间必须显式给正值；产品值仍待泵参数确认")
        self.config = config
        self.state = State.RESET_COAST
        self._below_current_s = 0.0
        self._duty = 0.0

    def _bus_ok(self, bus_v: float) -> bool:
        return self.config.bus_min_v <= bus_v <= self.config.bus_max_v

    def duty(self, speed_u: float, bus_v: float) -> float:
        if not 0.0 <= speed_u <= 1.0:
            raise ValueError("速度命令u必须在0..1")
        if not self._bus_ok(bus_v):
            raise ValueError("BUS不在当前候选窗口")
        return speed_u * min(self.config.duty_ceiling,
                             self.config.max_average_voltage_v / bus_v)

    def step(
        self,
        dt_s: float,
        *,
        bus_v: float,
        ipropi: IPropiDiag = IPropiDiag.OK,
        current_a: float | None = None,
        can_alive: bool = False,
        command_u: float | None = None,
        arm: bool = False,
        boot_ready: bool = False,
        explicit_rearm: bool = False,
        driver_fault: bool = False,
        mcu_reset: bool = False,
    ) -> State:
        if dt_s < 0:
            raise ValueError("dt_s不能为负")
        if mcu_reset:
            # R_SLEEP/R_IN1_PD/R_IN2_PD均下拉；复位/掉电为nSLEEP=0、输出高阻。
            self.state = State.RESET_COAST
            self._below_current_s = 0.0
            self._duty = 0.0
            return self.state

        unhealthy = (not self._bus_ok(bus_v) or ipropi is not IPropiDiag.OK
                     or driver_fault)
        if unhealthy and self.state is not State.RESET_COAST:
            self.state = State.FAULT_COAST
            self._below_current_s = 0.0
            self._duty = 0.0
            return self.state

        if self.state is State.FAULT_COAST:
            if explicit_rearm and can_alive and not unhealthy:
                self.state = State.SLEEP
            return self.state

        if self.state is State.RESET_COAST:
            if boot_ready:
                self.state = State.SLEEP
            return self.state

        if self.state is State.SLEEP:
            if arm and can_alive and command_u is not None:
                if command_u <= 0:
                    self.state = State.BRAKE
                    self._below_current_s = 0.0
                else:
                    self._duty = self.duty(command_u, bus_v)
                    self.state = State.DRIVE
            return self.state

        if self.state is State.DRIVE:
            if not can_alive or (command_u is not None and command_u <= 0):
                self.state = State.BRAKE
                self._below_current_s = 0.0
            elif command_u is not None:
                self._duty = self.duty(command_u, bus_v)
            return self.state

        if self.state is State.BRAKE:
            if current_a is None:
                self.state = State.FAULT_COAST
                self._below_current_s = 0.0
                return self.state
            if abs(current_a) <= self.config.brake_done_current_a:
                self._below_current_s += dt_s
                if self._below_current_s >= self.config.brake_done_dwell_s:
                    self.state = State.SLEEP
                    self._below_current_s = 0.0
            else:
                self._below_current_s = 0.0
            # BRAKE态不因CAN恢复或新速度值自动回到DRIVE。
            return self.state

        return self.state

    def pins(self) -> DriverPins:
        if self.state in (State.RESET_COAST, State.SLEEP, State.FAULT_COAST):
            return DriverPins(0, 0, 0, "高阻/滑行")
        if self.state is State.BRAKE:
            return DriverPins(1, 1, 1, "高侧短接制动")
        return DriverPins(1, 1, "反相PWM", "正向驱动脉冲/高侧慢衰减")


class PumpStateMachineTests(unittest.TestCase):
    def setUp(self):
        # 仅为状态转换测试夹具，0.5 A/0.2 s不代表水泵产品要求。
        self.model = PumpControllerModel(Config(brake_done_current_a=0.5,
                                                brake_done_dwell_s=0.2))
        self.model.step(0, bus_v=24, boot_ready=True)

    def arm_drive(self):
        self.model.step(0, bus_v=24, can_alive=True, arm=True, command_u=0.5)
        self.assertEqual(self.model.state, State.DRIVE)

    def test_bus_voltage_feedforward_keeps_average_command_under_21v(self):
        self.assertAlmostEqual(self.model.duty(1, 22.5), 21 / 22.5)
        self.assertAlmostEqual(self.model.duty(1, 37.8), 21 / 37.8)
        self.assertAlmostEqual(self.model.duty(0.5, 37.8), 0.5 * 21 / 37.8)

    def test_power_up_waits_for_explicit_enable(self):
        model = PumpControllerModel(Config(brake_done_current_a=0.5,
                                           brake_done_dwell_s=0.2))
        self.assertEqual(model.state, State.RESET_COAST)
        self.assertEqual(model.pins().n_sleep, 0)
        model.step(0, bus_v=24, can_alive=True, boot_ready=True,
                   command_u=0.7, arm=False)
        self.assertEqual(model.state, State.SLEEP)
        model.step(0, bus_v=24, can_alive=True, command_u=0.7, arm=False)
        self.assertEqual(model.state, State.SLEEP)

    def test_reset_default_can_timeout_brake_and_current_dwell_to_sleep(self):
        self.assertEqual(self.model.pins(), DriverPins(0, 0, 0, "高阻/滑行"))
        self.arm_drive()
        self.model.step(0.01, bus_v=24, can_alive=False, current_a=2.0)
        self.assertEqual(self.model.state, State.BRAKE)
        self.assertEqual(self.model.pins(), DriverPins(1, 1, 1, "高侧短接制动"))
        self.model.step(0.1, bus_v=24, can_alive=True, command_u=0.8, current_a=0.4)
        self.assertEqual(self.model.state, State.BRAKE)  # 一次低电流样本不足
        self.model.step(0.1, bus_v=24, can_alive=True, command_u=0.8, current_a=0.4)
        self.assertEqual(self.model.state, State.SLEEP)
        self.model.step(0, bus_v=24, mcu_reset=True)
        self.assertEqual(self.model.state, State.RESET_COAST)
        self.assertEqual(self.model.pins().n_sleep, 0)

    def test_zero_command_means_brake_not_zero_current(self):
        self.arm_drive()
        self.model.step(0.01, bus_v=24, can_alive=True, command_u=0, current_a=1.2)
        self.assertEqual(self.model.state, State.BRAKE)
        self.assertEqual(self.model.pins().in2, 1)

    def test_ipropi_diagnostics_and_bus_fault_latch_coast_until_rearm(self):
        for diag in (IPropiDiag.OPEN, IPropiDiag.SHORT,
                     IPropiDiag.SATURATED, IPropiDiag.INVALID):
            with self.subTest(diag=diag):
                model = PumpControllerModel(Config(brake_done_current_a=0.5,
                                                   brake_done_dwell_s=0.2))
                model.step(0, bus_v=24, boot_ready=True)
                model.step(0, bus_v=24, can_alive=True, arm=True, command_u=0.4)
                model.step(0, bus_v=24, ipropi=diag, can_alive=True)
                self.assertEqual(model.state, State.FAULT_COAST)
                model.step(0, bus_v=24, can_alive=True, command_u=0.8)
                self.assertEqual(model.state, State.FAULT_COAST)
                model.step(0, bus_v=24, can_alive=True, explicit_rearm=True)
                self.assertEqual(model.state, State.SLEEP)

        model = PumpControllerModel(Config(brake_done_current_a=0.5,
                                           brake_done_dwell_s=0.2))
        model.step(0, bus_v=24, boot_ready=True)
        model.step(0, bus_v=24, can_alive=True, arm=True, command_u=0.4)
        model.step(0, bus_v=24, driver_fault=True, can_alive=True)
        self.assertEqual(model.state, State.FAULT_COAST)
        self.assertEqual(model.pins().n_sleep, 0)

        for abnormal_bus in (22.4, 37.9):
            with self.subTest(bus_v=abnormal_bus):
                model = PumpControllerModel(Config(brake_done_current_a=0.5,
                                                   brake_done_dwell_s=0.2))
                model.step(0, bus_v=24, boot_ready=True)
                model.step(0, bus_v=24, can_alive=True, arm=True, command_u=0.4)
                model.step(0, bus_v=abnormal_bus, can_alive=True)
                self.assertEqual(model.state, State.FAULT_COAST)
                self.assertEqual(model.pins().n_sleep, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

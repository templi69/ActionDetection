"""
robot_interface.py
Member 4 - Deliverable: Robot Integration
Defines a common interface for controlling a robot (or a simulated stand-in),
so the decision engine never needs to know whether it's talking to real
hardware or a simulation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time


class RobotState(Enum):
    IDLE = "idle"
    FOLLOWING = "following"
    APPROACHING = "approaching"
    ASSISTING = "assisting"
    STOPPED = "stopped"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RobotStatus:
    state: RobotState = RobotState.IDLE
    linear_speed: float = 0.0     # m/s, +forward / -backward
    angular_speed: float = 0.0    # rad/s, +left / -right
    last_command: str = ""
    last_updated: float = field(default_factory=time.time)


class RobotController(ABC):
    """Abstract base so real hardware and simulation share one interface."""

    def __init__(self):
        self.status = RobotStatus()
        self._emergency = False

    @abstractmethod
    def _send_move(self, linear: float, angular: float):
        ...

    @abstractmethod
    def _send_stop(self):
        ...

    @abstractmethod
    def _send_alert(self, message: str):
        ...

    # ---- Public, safety-checked API used by the decision engine ----

    def move(self, linear: float, angular: float, reason: str = ""):
        if self._emergency:
            return  # ignore all motion commands until cleared
        linear = max(-1.0, min(1.0, linear))
        angular = max(-1.0, min(1.0, angular))
        self._send_move(linear, angular)
        self.status.linear_speed = linear
        self.status.angular_speed = angular
        self.status.last_command = f"move(linear={linear:.2f}, angular={angular:.2f}) [{reason}]"
        self.status.last_updated = time.time()

    def stop(self, reason: str = ""):
        self._send_stop()
        self.status.linear_speed = 0.0
        self.status.angular_speed = 0.0
        self.status.state = RobotState.STOPPED
        self.status.last_command = f"stop [{reason}]"
        self.status.last_updated = time.time()

    def emergency_stop(self, reason: str = ""):
        """Latching stop: ignores further move() calls until clear_emergency()."""
        self._emergency = True
        self._send_stop()
        self._send_alert(f"EMERGENCY STOP: {reason}")
        self.status.state = RobotState.EMERGENCY_STOP
        self.status.linear_speed = 0.0
        self.status.angular_speed = 0.0
        self.status.last_command = f"emergency_stop [{reason}]"
        self.status.last_updated = time.time()

    def clear_emergency(self):
        self._emergency = False
        self.status.state = RobotState.IDLE
        self.status.last_command = "emergency_cleared"

    def alert(self, message: str):
        self._send_alert(message)

    def set_state(self, state: RobotState):
        if not self._emergency:
            self.status.state = state

    @property
    def in_emergency(self) -> bool:
        return self._emergency


class SimulatedRobot(RobotController):
    """
    Console/log-based stand-in for real hardware. Tracks a fake pose so
    you can verify motion commands actually change something over time,
    and logs everything so you can replay a run for the report/demo.
    """

    def __init__(self, log_path: str = None):
        super().__init__()
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self._log_path = log_path
        self._log_file = open(log_path, "a") if log_path else None

    def _send_move(self, linear, angular):
        dt = 0.1  # assume ~10Hz control loop for the fake kinematics
        self.theta += angular * dt
        self.x += linear * dt
        self._log(
            f"MOVE linear={linear:.2f} angular={angular:.2f} "
            f"pose=({self.x:.2f},{self.y:.2f},{self.theta:.2f})"
        )

    def _send_stop(self):
        self._log("STOP")

    def _send_alert(self, message):
        self._log(f"ALERT: {message}")

    def _log(self, msg):
        line = f"[SimRobot] {msg}"
        print(line)
        if self._log_file:
            self._log_file.write(line + "\n")
            self._log_file.flush()

    def close(self):
        if self._log_file:
            self._log_file.close()


class SerialRobot(RobotController):
    """
    Skeleton for a real robot on a serial link (e.g. Arduino-driven rover).
    Fill in the exact protocol your hardware expects. Kept intentionally
    minimal so the decision engine and safety logic don't have to change
    when you swap simulation for real hardware.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        super().__init__()
        import serial  # only needed if you actually use real hardware
        self._ser = serial.Serial(port, baudrate, timeout=1)

    def _send_move(self, linear, angular):
        cmd = f"M {linear:.2f} {angular:.2f}\n"
        self._ser.write(cmd.encode())

    def _send_stop(self):
        self._ser.write(b"S\n")

    def _send_alert(self, message):
        # e.g. trigger a buzzer/LED code, or forward to a speaker module
        self._ser.write(f"A {message}\n".encode())

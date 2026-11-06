from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def embedded_energy_j(profile: dict[str, Any]) -> float:
    """Read embedded-computer energy without assuming it is incremental.

    `incremental_energy_j` is retained as a compatibility alias for Plans B/C.
    Plan A revised configurations use the semantically neutral field.
    """
    if "embedded_energy_j" in profile:
        return float(profile["embedded_energy_j"])
    if "incremental_energy_j" in profile:
        return float(profile["incremental_energy_j"])
    raise KeyError("Model profile requires embedded_energy_j or incremental_energy_j")


@dataclass(frozen=True)
class EnergyModel:
    battery_capacity_j: float
    speed_m_s: float
    move_power_w: float
    hover_power_w: float
    cell_distance_m: float
    reserve_j: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, float]) -> "EnergyModel":
        return cls(
            battery_capacity_j=float(config["battery_capacity_j"]),
            speed_m_s=float(config["speed_m_s"]),
            move_power_w=float(config["movement_power_w"]),
            hover_power_w=float(config["hover_power_w"]),
            cell_distance_m=float(config["cell_distance_m"]),
            reserve_j=float(config.get("reserve_j", 0.0)),
        )

    def __post_init__(self) -> None:
        if min(self.battery_capacity_j, self.speed_m_s, self.move_power_w,
               self.hover_power_w, self.cell_distance_m) <= 0:
            raise ValueError("Parâmetros físicos devem ser positivos.")
        if not 0 <= self.reserve_j < self.battery_capacity_j:
            raise ValueError("A reserva deve estar entre zero e a capacidade da bateria.")

    def movement(self, distance_m: float) -> tuple[float, float]:
        seconds = distance_m / self.speed_m_s
        return seconds, self.move_power_w * seconds

    def analysis(self, latency_s: float, embedded_computer_energy_j: float) -> tuple[float, float]:
        if latency_s < 0 or embedded_computer_energy_j < 0:
            raise ValueError("Latência e energia incremental não podem ser negativas.")
        return self.hover_power_w * latency_s, embedded_computer_energy_j

    def return_energy(self, distance_home_m: float) -> float:
        return self.movement(distance_home_m)[1]

    @property
    def usable_energy_j(self) -> float:
        return self.battery_capacity_j - self.reserve_j

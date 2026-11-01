"""Inventory decision layer: multi-product newsvendor and multi-period replenishment simulation."""
from probabilistic_modeling.inventory.simulation import (
    InventorySimulationResult,
    InventorySimulator,
    compute_simulation_kpis,
    resolve_target_inventory,
    simulate_inventory,
)

__all__ = [
    "InventorySimulationResult",
    "InventorySimulator",
    "compute_simulation_kpis",
    "resolve_target_inventory",
    "simulate_inventory",
]


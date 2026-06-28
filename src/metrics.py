"""Utilities for formatting inventory simulation metrics."""

from __future__ import annotations

from .inventory import InventoryMetrics


def metrics_to_dict(
    metrics: InventoryMetrics,
) -> dict[str, float]:
    """
    Convert InventoryMetrics dataclass into a
    rounded dictionary suitable for reporting.
    """

    return {
        "Fill_Rate_%": round(metrics.fill_rate * 100, 2),
        "Stockout_Rate_%": round(metrics.stockout_rate * 100, 2),
        "Total_Stockout_Volume": round(
            metrics.total_stockout_volume,
            2,
        ),
        "Avg_Ending_Inventory": round(
            metrics.average_inventory,
            2,
        ),
        "Avg_Order_Quantity": round(
            metrics.average_order_quantity,
            2,
        ),
        "Holding_Cost_$": round(
            metrics.holding_cost,
            2,
        ),
        "Ordering_Cost_$": round(
            metrics.ordering_cost,
            2,
        ),
        "Total_Cost_$": round(
            metrics.total_inventory_cost,
            2,
        ),
    }
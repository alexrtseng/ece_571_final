"""
Battery arbitrage optimization and batch evaluation utilities.

The LP formulation is preserved exactly from the original project.
All battery parameters are fixed to a 1 MW / 4 MWh (4-hour) battery.
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from tqdm import tqdm

# ── 4-hour battery constants ───────────────────────────────────────────────────
CAPACITY_MWH              = 4.0    # 4-hour battery at 1 MW
MAX_CHARGE_MW             = 1.0
MAX_DISCHARGE_MW          = 1.0
INITIAL_CHARGE_MWH        = 2.0   # start at 50% SOC
IN_EFFICIENCY             = 0.98
OUT_EFFICIENCY            = 0.98
SELF_DISCHARGE_PER_HOUR   = 0.0


def deterministic_arbitrage_opt(
    prices_df: pd.DataFrame,
    require_equivalent_soe: bool = True,
    use_barrier: bool = True,
) -> tuple[pd.DataFrame, float]:
    """
    Solve the battery arbitrage LP for a single price trace.

    prices_df must have:
      - a pd.DatetimeIndex
      - a column named 'lmp' ($/MWh)

    Returns (result_df, objective_value) where objective_value is
    total revenue in $ (for the given MW scale over the day).
    """
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        raise ValueError("prices_df must be indexed by a DatetimeIndex")
    if "lmp" not in prices_df.columns:
        raise ValueError("prices_df must contain an 'lmp' column")
    prices_df = prices_df.sort_index()

    capacity_mwh              = float(CAPACITY_MWH)
    max_charge_mw             = float(MAX_CHARGE_MW)
    max_discharge_mw          = float(MAX_DISCHARGE_MW)
    initial_charge_mwh        = float(INITIAL_CHARGE_MWH)
    in_efficiency             = float(IN_EFFICIENCY)
    out_efficiency            = float(OUT_EFFICIENCY)
    self_discharge_percent_per_hour = float(SELF_DISCHARGE_PER_HOUR)

    model = gp.Model("battery_arbitrage")
    model.Params.OutputFlag = 0
    if use_barrier:
        model.Params.Method         = 2   # Barrier
        model.Params.Presolve       = 2   # Aggressive presolve
        model.Params.BarHomogeneous = 1   # Homogeneous form

    times = list(prices_df.index)
    T = len(times)
    if T < 2:
        raise ValueError("Need at least two timestamps to define time intervals")

    soe: list[gp.Var] = [model.addVar(lb=0.0, ub=capacity_mwh, name="soe_0")]
    model.addConstr(soe[0] == initial_charge_mwh, name="init_soe")
    charge: list[gp.Var] = []
    discharge: list[gp.Var] = []

    obj_expr = gp.LinExpr()
    for t in range(T - 1):
        dt_hours = (times[t + 1] - times[t]).total_seconds() / 3600.0
        c = model.addVar(lb=0.0, ub=max_charge_mw,    name=f"charge_{t}")
        d = model.addVar(lb=0.0, ub=max_discharge_mw, name=f"discharge_{t}")
        charge.append(c)
        discharge.append(d)

        next_soe = model.addVar(lb=0.0, ub=capacity_mwh, name=f"soe_{t + 1}")
        soe.append(next_soe)

        model.addConstr(
            next_soe
            == soe[t]
            + (
                c * in_efficiency
                - d / out_efficiency
                - soe[t] * self_discharge_percent_per_hour
            )
            * dt_hours,
            name=f"soe_dyn_{t}",
        )

        price = float(prices_df.iloc[t]["lmp"])
        obj_expr += price * (d - c) * dt_hours

    if require_equivalent_soe:
        model.addConstr(soe[-1] == initial_charge_mwh, name="final_soe_equal_init")

    model.setObjective(obj_expr, gp.GRB.MAXIMIZE)
    model.optimize()

    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Optimization did not find optimal solution (status {model.Status})")

    result_df = pd.DataFrame(
        {
            "state_of_energy_mwh": [v.X for v in soe],
            "charge_mw":           [v.X for v in charge] + [0.0],
            "discharge_mw":        [v.X for v in discharge] + [0.0],
        },
        index=times,
    )
    return result_df, model.ObjVal


# ── Helpers ───────────────────────────────────────────────────────────────────

def prices_to_df(prices_288: np.ndarray, anchor_date: str = "2024-01-01") -> pd.DataFrame:
    """
    Convert a (288,) array of 5-min RT prices into the prices_df format.
    The absolute date is arbitrary; only time-differences matter for the LP.
    """
    index = pd.date_range(anchor_date, periods=288, freq="5min", tz="UTC")
    return pd.DataFrame({"lmp": prices_288.astype(float)}, index=index)


def batch_revenue(
    prices_array: np.ndarray,
    desc: str = "optimizing",
    require_equivalent_soe: bool = True,
) -> np.ndarray:
    """
    Run the battery LP on each row of prices_array (N, 288).
    Returns revenues array of shape (N,).
    """
    revenues = []
    for row in tqdm(prices_array, desc=desc, leave=False):
        _, rev = deterministic_arbitrage_opt(
            prices_to_df(row),
            require_equivalent_soe=require_equivalent_soe,
        )
        revenues.append(rev)
    return np.array(revenues, dtype=np.float64)

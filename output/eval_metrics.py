import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 52) -> float:
    if returns.std(ddof=1) == 0 or len(returns) < 2:
        return float("nan")
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=1))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 52) -> float:
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    if not downside_std or downside_std == 0 or np.isnan(downside_std):
        return float("nan")
    return float(np.sqrt(periods_per_year) * returns.mean() / downside_std)


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    return float(drawdown.min())


def summarize(curve: pd.DataFrame, label: str) -> dict:
    returns = curve["actual_pnl"]
    return {
        "strategy": label,
        "total_pnl": float(curve["cumulative_pnl"].iloc[-1]) if len(curve) else float("nan"),
        "n_periods": len(curve),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(curve["cumulative_pnl"]),
        "win_rate_periods": float((returns > 0).mean()) if len(returns) else float("nan"),
    }
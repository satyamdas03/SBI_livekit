# src/calc_tools.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
from math import pow

Compounding = Literal["quarterly", "monthly", "half-yearly", "yearly"]

def _n_per_year(kind: Compounding) -> int:
    return {"monthly": 12, "quarterly": 4, "half-yearly": 2, "yearly": 1}[kind]

@dataclass
class FDMaturity:
    principal: float
    annual_rate_pct: float
    years: float
    compounding: Compounding = "quarterly"

    def maturity(self) -> dict:
        n = _n_per_year(self.compounding)
        r = self.annual_rate_pct / 100.0
        A = self.principal * pow(1.0 + r / n, n * self.years)
        return {
            "principal": round(self.principal, 2),
            "rate_pct": round(self.annual_rate_pct, 4),
            "tenor_years": round(self.years, 4),
            "compounding": self.compounding,
            "maturity_amount": round(A, 2),
            "interest_earned": round(A - self.principal, 2),
        }

def premature_effective_rate(
    applicable_rate_pct: float,
    contracted_rate_pct: float,
    deposit_amount: float,
) -> float:
    """Apply SBI retail TD premature penalty rule.
    Penalty below the lower of (applicable rate for the run period) or (contracted rate):
        - 0.50% if amount < 5 lakh
        - 1.00% if amount >= 5 lakh and < 5 crore
    """
    base = min(applicable_rate_pct, contracted_rate_pct)
    penalty = 0.50 if deposit_amount < 5e5 else 1.00
    eff = max(0.0, base - penalty)
    return eff

@dataclass
class TDSEstimate:
    # Basic 194A estimate for banks (simplified).
    annual_interest: float
    senior_citizen: bool = False
    has_pan: bool = True

    def tds(self) -> dict:
        threshold = 50000.0 if self.senior_citizen else 40000.0
        rate = 0.10 if self.has_pan else 0.20
        liable = max(0.0, self.annual_interest - threshold)
        tds_amt = round(liable * rate, 2)
        return {
            "threshold": threshold,
            "tds_rate": rate,
            "interest_estimate": round(self.annual_interest, 2),
            "tds_payable": tds_amt,
            "note": "Thresholds per Sec 194A; subject to Form 15G/15H and bank-wise computation.",
        }
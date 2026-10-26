"""
Retention Campaign Financial Economics and Cost-Benefit Matrix.
Defines explicit cost parameters (intervention offer cost, save rate, customer CLV),
confusion matrix utility payouts, and net financial value calculations.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

from src.config import get_settings


@dataclass
class CostBenefitMatrix:
    """
    Financial parameters for proactive customer churn retention campaigns.

    Attributes:
        offer_cost: Direct cost of proactive retention outreach (discounts, perk, concierge).
        save_rate: Probability that an at-risk customer is successfully saved by the offer.
        customer_clv: Net present Customer Lifetime Value (CLV) preserved if retained.
        horizon_months: Planning horizon for retained revenue (months).
    """
    offer_cost: float = 35.0
    save_rate: float = 0.22
    customer_clv: float = 450.0
    horizon_months: int = 12

    @property
    def value_true_positive(self) -> float:
        """
        Net expected value gained from intervening on an actual churner:
        Value = (Save Rate * CLV) - Offer Cost
        """
        return (self.save_rate * self.customer_clv) - self.offer_cost

    @property
    def value_false_positive(self) -> float:
        """
        Cost incurred from unnecessarily intervening on a loyal customer who would not have churned:
        Value = -Offer Cost
        """
        return -self.offer_cost

    @property
    def value_false_negative(self) -> float:
        """
        Opportunity loss of failing to intervene on a churner (status quo = 0 net change relative to baseline).
        """
        return 0.0

    @property
    def value_true_negative(self) -> float:
        """
        Cost of correct non-intervention (status quo = 0).
        """
        return 0.0

    @property
    def theoretical_optimal_threshold(self) -> float:
        """
        Bayes decision theoretical cutoff where expected net value of intervening equals zero:
        p * (Save Rate * CLV) - Offer Cost = 0 => p* = Offer Cost / (Save Rate * CLV)
        """
        denom = self.save_rate * self.customer_clv
        if denom <= 0:
            return 0.50
        cutoff = self.offer_cost / denom
        return float(np.clip(cutoff, 0.05, 0.95))

    def calculate_net_savings_from_cm(
        self,
        true_positives: int,
        false_positives: int,
    ) -> float:
        """
        Calculates realized campaign net profit/savings from discrete confusion matrix counts:
        Net Savings = (TP * V_TP) + (FP * V_FP)
        """
        net_saved = (true_positives * self.value_true_positive) + (false_positives * self.value_false_positive)
        return float(net_saved)

    def calculate_expected_value_per_customer(
        self,
        predicted_probability: float,
        individual_clv: Optional[float] = None,
    ) -> float:
        """
        Calculates individual expected net financial value if targeted with retention offer:
        EV = P(Churn) * Save Rate * CLV - Offer Cost
        """
        clv = individual_clv if individual_clv is not None else self.customer_clv
        return float((predicted_probability * self.save_rate * clv) - self.offer_cost)

"""
EDA Visualization Engine for IBM Telco Customer Churn.
Generates reproducible publication-grade static figures (matplotlib/seaborn)
and interactive Plotly HTML dashboards.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Headless backend for reliable server execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

from src.config import BASE_DIR, get_settings


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Set standard publication aesthetic
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 150,
})

# Curated palette
RETAINED_COLOR = "#2b5c8f"  # Deep corporate navy
CHURN_COLOR = "#d9534f"     # Vibrant risk coral
ACCENT_COLOR = "#3caea3"    # Teal accent
NEUTRAL_COLOR = "#f4f6f9"   # Light slate


class EDAVisualizer:
    """
    Renders and persists publication-grade visualizations for the Telco Churn EDA.
    """

    def __init__(self, df: pd.DataFrame, output_dir: Optional[Union[str, Path]] = None):
        self.df = df.copy()
        settings = get_settings()
        self.output_dir = (
            Path(output_dir) if output_dir else BASE_DIR / "artifacts" / "figures"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_target_distribution(self) -> Path:
        """
        Chart 1: Target Variable (Churn) Class Balance.
        Dual panel: Count bar chart with percentages and a Donut distribution.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        counts = self.df["churn"].value_counts()
        total = len(self.df)

        # Left: Countplot with explicit annotations
        colors = [RETAINED_COLOR, CHURN_COLOR]
        bars = axes[0].bar(counts.index, counts.values, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
        axes[0].set_title("Customer Churn Volume (Total = 7,043)")
        axes[0].set_ylabel("Customer Count")
        axes[0].set_xlabel("Churn Status")
        axes[0].set_ylim(0, max(counts.values) * 1.15)

        for bar in bars:
            height = bar.get_height()
            pct = 100.0 * height / total
            axes[0].annotate(
                f"{height:,}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # Right: Donut Chart
        wedges, texts, autotexts = axes[1].pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
            textprops=dict(fontweight="bold"),
        )
        plt.setp(autotexts, size=11, weight="bold", color="white")
        axes[1].text(0, 0, f"Churn Rate\n{100.0 * counts['Yes'] / total:.1f}%", ha="center", va="center", fontsize=11, fontweight="bold")
        axes[1].set_title("Portfolio Retention vs Churn Split")

        fig.suptitle("IBM Telco Churn: Target Class Balance & Imbalance Ratio (2.77 : 1)", fontsize=13)
        plt.tight_layout()

        out_path = self.output_dir / "01_target_distribution.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated target distribution chart: {out_path}")
        return out_path

    def plot_tenure_dynamics(self) -> Path:
        """
        Chart 2: Tenure Dynamics & Early Hazard Decay.
        Dual panel: KDE distributions split by Churn and Tenure Cohort Churn Rates.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: KDE plot showing bimodal distribution
        sns.kdeplot(
            data=self.df,
            x="tenure",
            hue="churn",
            palette={"No": RETAINED_COLOR, "Yes": CHURN_COLOR},
            fill=True,
            common_norm=False,
            alpha=0.45,
            linewidth=2,
            ax=axes[0],
        )
        axes[0].set_title("Tenure Density: Early Hazard Peak vs Long-Term Loyalty")
        axes[0].set_xlabel("Tenure (Months with Company)")
        axes[0].set_ylabel("Kernel Density Estimate (KDE)")
        axes[0].axvline(12, color="gray", linestyle="--", alpha=0.7, label="1-Year Cliff (Month 12)")
        axes[0].legend(title="Churn", loc="upper right")

        # Right: Tenure Cohort Churn Rate
        tenure_bins = pd.cut(
            self.df["tenure"],
            bins=[-1, 6, 12, 24, 48, 72],
            labels=["0 - 6m", "7 - 12m", "13 - 24m", "25 - 48m", "49 - 72m"],
        )
        cohort_stats = (
            self.df.groupby(tenure_bins, observed=False)["churn_label"]
            .agg(churn_rate=lambda s: 100.0 * s.mean(), count="count")
            .reset_index()
        )

        bars = axes[1].bar(
            cohort_stats["tenure"].astype(str),
            cohort_stats["churn_rate"],
            color=["#d9534f", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"],
            edgecolor="black",
            linewidth=0.8,
        )
        axes[1].plot(
            cohort_stats["tenure"].astype(str),
            cohort_stats["churn_rate"],
            color="#2c3e50",
            marker="o",
            linewidth=2.5,
            markersize=7,
        )
        axes[1].set_title("Churn Rate by Tenure Cohort (% Churned)")
        axes[1].set_xlabel("Tenure Cohort")
        axes[1].set_ylabel("Cohort Churn Rate (%)")
        axes[1].set_ylim(0, 60)

        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        fig.suptitle("Tenure Dynamics: Customer Survival Curve & Retention Stabilization", fontsize=13)
        plt.tight_layout()

        out_path = self.output_dir / "02_tenure_distribution_and_churn.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated tenure dynamics chart: {out_path}")
        return out_path

    def plot_monthly_charges_distribution(self) -> Path:
        """
        Chart 3: Monthly Charges vs Churn & Contract Term.
        Dual panel: Violin/Box distribution and Monthly Charges across Contracts.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Box/Violin plot for Monthly Charges by Churn
        sns.boxplot(
            data=self.df,
            x="churn",
            y="monthly_charges",
            hue="churn",
            legend=False,
            palette={"No": RETAINED_COLOR, "Yes": CHURN_COLOR},
            width=0.45,
            boxprops=dict(alpha=0.8),
            ax=axes[0],
        )
        # Overlay points summary
        medians = self.df.groupby("churn")["monthly_charges"].median()
        axes[0].set_title("Monthly Recurring Revenue: Churned vs Retained")
        axes[0].set_xlabel("Churn Status")
        axes[0].set_ylabel("Monthly Charges ($)")
        axes[0].text(0, medians["No"] + 2, f"Median: ${medians['No']:.2f}", ha="center", fontweight="bold", color="white")
        axes[0].text(1, medians["Yes"] + 2, f"Median: ${medians['Yes']:.2f}", ha="center", fontweight="bold", color="white")

        # Right: Monthly charges by Contract Term and Churn
        sns.barplot(
            data=self.df,
            x="contract",
            y="monthly_charges",
            hue="churn",
            palette={"No": RETAINED_COLOR, "Yes": CHURN_COLOR},
            errorbar=None,
            ax=axes[1],
        )
        axes[1].set_title("Average Monthly Charges by Contract Structure & Churn")
        axes[1].set_xlabel("Contract Term")
        axes[1].set_ylabel("Average Monthly Charges ($)")
        axes[1].legend(title="Churn", loc="upper left")

        fig.suptitle("Price Sensitivity & Revenue Exposure: Churned Customers Pay Higher Monthly Bills", fontsize=13)
        plt.tight_layout()

        out_path = self.output_dir / "03_monthly_charges_distribution.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated monthly charges distribution chart: {out_path}")
        return out_path

    def plot_contract_churn_comparison(self) -> Path:
        """
        Chart 4: Contract Commitment Horizon Comparison.
        Displays total volume and churn rate across Month-to-Month, 1-Year, and 2-Year contracts.
        """
        fig, ax1 = plt.subplots(figsize=(10, 5))

        contract_stats = (
            self.df.groupby("contract")
            .agg(
                total_customers=("customer_id", "count"),
                churn_rate=("churn_label", lambda s: 100.0 * s.mean()),
            )
            .reindex(["Month-to-month", "One year", "Two year"])
            .reset_index()
        )

        x = np.arange(len(contract_stats))
        width = 0.4

        # Primary Axis: Customer Volume
        bars = ax1.bar(
            x,
            contract_stats["total_customers"],
            width,
            color=RETAINED_COLOR,
            alpha=0.85,
            edgecolor="black",
            label="Total Customer Volume",
        )
        ax1.set_ylabel("Customer Volume (Accounts)", color=RETAINED_COLOR, fontweight="bold")
        ax1.set_xlabel("Contract Commitment Term")
        ax1.set_xticks(x)
        ax1.set_xticklabels(contract_stats["contract"], fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=RETAINED_COLOR)
        ax1.set_ylim(0, 4500)

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f"{height:,} accts",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                color=RETAINED_COLOR,
            )

        # Secondary Axis: Churn Rate
        ax2 = ax1.twinx()
        lines = ax2.plot(
            x,
            contract_stats["churn_rate"],
            color=CHURN_COLOR,
            marker="o",
            linewidth=3,
            markersize=9,
            label="Churn Rate (%)",
        )
        ax2.set_ylabel("Churn Rate (%)", color=CHURN_COLOR, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=CHURN_COLOR)
        ax2.set_ylim(0, 55)

        for i, val in enumerate(contract_stats["churn_rate"]):
            ax2.annotate(
                f"{val:.1f}% Churn",
                xy=(i, val),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontweight="bold",
                color=CHURN_COLOR,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=CHURN_COLOR, alpha=0.9),
            )

        plt.title("Contract Term Impact: Month-to-Month (42.7%) vs 1-Year (11.3%) vs 2-Year (2.8%)", pad=15)
        plt.tight_layout()

        out_path = self.output_dir / "04_contract_type_churn.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated contract churn comparison chart: {out_path}")
        return out_path

    def plot_services_and_addons_matrix(self) -> Path:
        """
        Chart 5: Service Configurations & Add-On Stacking Analysis.
        4-Panel Grid: Internet Service, Tech Support, Online Security, and Add-on Stacking curve.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel A: Internet Service
        net_stats = (
            self.df.groupby("internet_service")["churn_label"]
            .apply(lambda s: 100.0 * s.mean())
            .reindex(["DSL", "Fiber optic", "No"])
        )
        axes[0, 0].bar(net_stats.index, net_stats.values, color=["#3498db", "#e74c3c", "#2ecc71"], edgecolor="black")
        axes[0, 0].set_title("A. Internet Service Churn Risk (Fiber Optic Hazard)")
        axes[0, 0].set_ylabel("Churn Rate (%)")
        axes[0, 0].set_ylim(0, 50)
        for i, v in enumerate(net_stats.values):
            axes[0, 0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

        # Panel B: Tech Support
        ts_stats = (
            self.df.groupby("tech_support")["churn_label"]
            .apply(lambda s: 100.0 * s.mean())
            .reindex(["No", "Yes", "No internet service"])
        )
        axes[0, 1].bar(ts_stats.index, ts_stats.values, color=["#e74c3c", "#2ecc71", "#95a5a6"], edgecolor="black")
        axes[0, 1].set_title("B. Tech Support Subscription Impact")
        axes[0, 1].set_ylabel("Churn Rate (%)")
        axes[0, 1].set_ylim(0, 50)
        for i, v in enumerate(ts_stats.values):
            axes[0, 1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

        # Panel C: Online Security
        sec_stats = (
            self.df.groupby("online_security")["churn_label"]
            .apply(lambda s: 100.0 * s.mean())
            .reindex(["No", "Yes", "No internet service"])
        )
        axes[1, 0].bar(sec_stats.index, sec_stats.values, color=["#e74c3c", "#2ecc71", "#95a5a6"], edgecolor="black")
        axes[1, 0].set_title("C. Online Security Add-On Impact")
        axes[1, 0].set_ylabel("Churn Rate (%)")
        axes[1, 0].set_ylim(0, 50)
        for i, v in enumerate(sec_stats.values):
            axes[1, 0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

        # Panel D: Feature Stacking (Add-on counts vs Churn)
        addons = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]
        self.df["addon_count"] = 0
        for a in addons:
            self.df["addon_count"] += (self.df[a] == "Yes").astype(int)

        stack_stats = self.df.groupby("addon_count")["churn_label"].apply(lambda s: 100.0 * s.mean())
        axes[1, 1].plot(stack_stats.index, stack_stats.values, marker="s", color="#8e44ad", linewidth=2.5, markersize=8)
        axes[1, 1].bar(stack_stats.index, stack_stats.values, color="#8e44ad", alpha=0.3, edgecolor="black")
        axes[1, 1].set_title("D. Add-On Feature Stacking Elasticity (0 to 6 Services)")
        axes[1, 1].set_xlabel("Number of Subscribed Add-Ons")
        axes[1, 1].set_ylabel("Churn Rate (%)")
        axes[1, 1].set_ylim(0, 60)
        for x, y in zip(stack_stats.index, stack_stats.values):
            axes[1, 1].text(x, y + 2, f"{y:.1f}%", ha="center", fontweight="bold")

        fig.suptitle("Service Combinations & Product Stacking: Protection Add-Ons Dramatically Mitigate Churn", fontsize=14)
        plt.tight_layout()

        out_path = self.output_dir / "05_services_and_addons_churn.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated services matrix chart: {out_path}")
        return out_path

    def plot_payment_method_friction(self) -> Path:
        """
        Chart 6: Payment Method & Invoicing Friction.
        Dual panel: Churn rate by Payment Disbursement vehicle and Paperless Billing split.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Payment Method
        pm_stats = (
            self.df.groupby("payment_method")["churn_label"]
            .apply(lambda s: 100.0 * s.mean())
            .sort_values(ascending=False)
        )
        bars = axes[0].barh(
            pm_stats.index,
            pm_stats.values,
            color=["#d9534f" if "Electronic check" in idx else "#2b5c8f" for idx in pm_stats.index],
            edgecolor="black",
        )
        axes[0].set_title("Payment Method Churn Risk (Electronic Check Vulnerability)")
        axes[0].set_xlabel("Churn Rate (%)")
        axes[0].set_xlim(0, 55)

        for bar in bars:
            width = bar.get_width()
            axes[0].annotate(
                f" {width:.1f}%",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontweight="bold",
            )

        # Right: Paperless Billing
        pb_stats = self.df.groupby("paperless_billing")["churn_label"].apply(lambda s: 100.0 * s.mean())
        axes[1].bar(
            pb_stats.index,
            pb_stats.values,
            color=[RETAINED_COLOR, CHURN_COLOR],
            width=0.45,
            edgecolor="black",
        )
        axes[1].set_title("Paperless Billing vs Standard Invoicing")
        axes[1].set_ylabel("Churn Rate (%)")
        axes[1].set_xlabel("Paperless Billing Subscribed")
        axes[1].set_ylim(0, 45)

        for i, val in enumerate(pb_stats.values):
            axes[1].text(i, val + 1, f"{val:.1f}%", ha="center", fontweight="bold")

        fig.suptitle("Payment Invoicing & Transaction Friction: Automated Billing Anchors Retention", fontsize=13)
        plt.tight_layout()

        out_path = self.output_dir / "06_payment_method_friction.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated payment friction chart: {out_path}")
        return out_path

    def plot_correlation_matrix(self) -> Path:
        """
        Chart 7: Continuous Feature Correlation Heatmap.
        Calculates Spearman rank correlation matrix for tenure, charges, and churn_label.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        cols = ["tenure", "monthly_charges", "total_charges", "churn_label"]
        corr = self.df[cols].corr(method="spearman")

        labels = ["Tenure", "Monthly Charges", "Total Charges", "Churn (Binary)"]
        sns.heatmap(
            corr,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            vmin=-0.5,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.8,
            cbar_kws={"label": "Spearman Rank Correlation"},
            ax=ax,
        )
        ax.set_title("Spearman Rank Correlation Heatmap (Continuous Attributes vs Churn)", pad=15)
        plt.tight_layout()

        out_path = self.output_dir / "07_correlation_matrix.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated correlation heatmap: {out_path}")
        return out_path

    def build_interactive_html_dashboard(self) -> Path:
        """
        Chart 8: Interactive Plotly HTML Executive Dashboard.
        Self-contained HTML file for rich visual inspection and dashboard embedding.
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Tenure vs Monthly Charges (Colored by Churn)",
                "Churn Rate by Contract Type",
                "Churn Rate by Internet Service",
                "Churn Rate by Payment Method",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        # 1. Scatter: Tenure vs Monthly Charges
        sample_subset = self.df.sample(n=min(1200, len(self.df)), random_state=42)
        fig.add_trace(
            go.Scatter(
                x=sample_subset[sample_subset["churn"] == "No"]["tenure"],
                y=sample_subset[sample_subset["churn"] == "No"]["monthly_charges"],
                mode="markers",
                name="Retained (No)",
                marker=dict(color=RETAINED_COLOR, size=5, opacity=0.6),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sample_subset[sample_subset["churn"] == "Yes"]["tenure"],
                y=sample_subset[sample_subset["churn"] == "Yes"]["monthly_charges"],
                mode="markers",
                name="Churned (Yes)",
                marker=dict(color=CHURN_COLOR, size=6, opacity=0.7),
            ),
            row=1,
            col=1,
        )

        # 2. Bar: Contract Churn Rate
        c_stats = self.df.groupby("contract")["churn_label"].mean() * 100
        fig.add_trace(
            go.Bar(
                x=c_stats.index,
                y=c_stats.values,
                name="Contract Churn %",
                marker_color=[CHURN_COLOR, ACCENT_COLOR, RETAINED_COLOR],
            ),
            row=1,
            col=2,
        )

        # 3. Bar: Internet Service Churn Rate
        net_stats = self.df.groupby("internet_service")["churn_label"].mean() * 100
        fig.add_trace(
            go.Bar(
                x=net_stats.index,
                y=net_stats.values,
                name="Internet Churn %",
                marker_color=["#3498db", CHURN_COLOR, "#2ecc71"],
            ),
            row=2,
            col=1,
        )

        # 4. Bar: Payment Method Churn Rate
        pm_stats = self.df.groupby("payment_method")["churn_label"].mean() * 100
        fig.add_trace(
            go.Bar(
                x=pm_stats.index,
                y=pm_stats.values,
                name="Payment Churn %",
                marker_color=CHURN_COLOR,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title_text="Customer Churn Intelligence: Interactive Visual Explorer",
            height=850,
            showlegend=False,
            template="plotly_white",
        )

        html_path = self.output_dir / "interactive_churn_explorer.html"
        fig.write_html(str(html_path))
        logger.info(f"Generated interactive Plotly HTML dashboard: {html_path}")
        return html_path

    def render_all(self) -> Dict[str, Path]:
        """
        Renders and returns all 8 visualization assets.
        """
        paths = {
            "target_distribution": self.plot_target_distribution(),
            "tenure_dynamics": self.plot_tenure_dynamics(),
            "monthly_charges": self.plot_monthly_charges_distribution(),
            "contract_churn": self.plot_contract_churn_comparison(),
            "services_addons": self.plot_services_and_addons_matrix(),
            "payment_friction": self.plot_payment_method_friction(),
            "correlation_matrix": self.plot_correlation_matrix(),
            "interactive_html": self.build_interactive_html_dashboard(),
        }
        return paths

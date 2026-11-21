"""
Основний скрипт аналізу продажів і комерційної діяльності підприємства
(на основі набору даних Sample Superstore та пов'язаних довідників).

Запуск:
    python src/run_analysis.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Налаштування стилю графіків (підтримка кирилиці та висока якість)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
sns.set_theme(style="whitegrid", palette="Set2", font="DejaVu Sans")


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "results" / "figures"
EXPORTS_DIR = BASE_DIR / "results" / "exports"

for d in (PROCESSED_DIR, FIGURES_DIR, EXPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Завантаження основного набору даних і двох довідкових таблиць."""
    sales_df = pd.read_csv(RAW_DIR / "SampleSuperstore.csv")
    managers_df = pd.read_csv(RAW_DIR / "regional_managers.csv")
    shipping_df = pd.read_csv(RAW_DIR / "shipping_modes.csv")
    return sales_df, managers_df, shipping_df


def clean_and_enrich(
    sales_df: pd.DataFrame,
    managers_df: pd.DataFrame,
    shipping_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Очищення даних, перевірка викидів, об'єднання таблиць і створення ознак."""
    report = {}
    report["raw_rows"] = int(sales_df.shape[0])
    report["raw_cols"] = int(sales_df.shape[1])
    report["missing_before"] = int(sales_df.isna().sum().sum())
    report["duplicates_before"] = int(sales_df.duplicated().sum())

    # 1. Видалення повних дублікатів
    df = sales_df.drop_duplicates().copy()
    report["clean_rows"] = int(df.shape[0])
    report["duplicates_removed"] = report["raw_rows"] - report["clean_rows"]

    # 2. Корекція типів даних: поштовий індекс -> 5-значний рядок із ведучими нулями
    df["Postal Code"] = df["Postal Code"].astype(str).str.zfill(5)

    # 3. Перевірка некоректних значень (Sales <= 0, Quantity <= 0, Discount поза [0, 1])
    report["invalid_sales"] = int((df["Sales"] <= 0).sum())
    report["invalid_quantity"] = int((df["Quantity"] <= 0).sum())
    report["invalid_discount"] = int(((df["Discount"] < 0) | (df["Discount"] > 1)).sum())

    # 4. Оцінка потенційних викидів за методом IQR (без сліпого видалення)
    outliers_info = {}
    for col in ["Sales", "Profit", "Quantity"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cnt = int(((df[col] < lower) | (df[col] > upper)).sum())
        outliers_info[col] = {
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lower": round(float(lower), 2),
            "upper": round(float(upper), 2),
            "count": cnt,
            "pct": round(cnt / len(df) * 100, 2),
        }
    report["outliers_iqr"] = outliers_info

    # 5. Об'єднання пов'язаних таблиць (2 обґрунтовані merge)
    df = df.merge(managers_df, on="Region", how="left")
    df = df.merge(shipping_df, on="Ship Mode", how="left")
    report["missing_after_merge"] = int(df.isna().sum().sum())

    # 6. Створення нових аналітичних ознак (5 ознак)
    # 6.1. Маржинальність замовлення (%)
    df["Profit_Margin_Pct"] = np.round((df["Profit"] / df["Sales"]) * 100, 2)

    # 6.2. Ціна за одиницю товару ($)
    df["Unit_Price"] = np.round(df["Sales"] / df["Quantity"], 2)

    # 6.3. Статус прибутковості (Прибуткове / Беззбиткове / Збиткове)
    df["Profit_Status"] = np.where(
        df["Profit"] > 0,
        "Прибуткове",
        np.where(df["Profit"] < 0, "Збиткове", "Беззбиткове"),
    )

    # 6.4. Категорія величини замовлення за сумою продажу (pd.cut)
    df["Order_Size"] = pd.cut(
        df["Sales"],
        bins=[0, 50, 200, 1000, np.inf],
        labels=["Малий (до $50)", "Середній ($50–200)", "Великий ($200–1000)", "Преміум (>$1000)"],
    )

    # 6.5. Рівень знижки (pd.cut)
    df["Discount_Tier"] = pd.cut(
        df["Discount"],
        bins=[-0.01, 0.0, 0.2, 0.4, 1.0],
        labels=["Без знижки (0%)", "Помірна (1–20%)", "Середня (21–40%)", "Висока (>40%)"],
    )

    # 6.6. Оціночна собівартість (Cost = Sales - Profit)
    df["Estimated_Cost"] = np.round(df["Sales"] - df["Profit"], 2)

    report["final_cols"] = int(df.shape[1])

    # Збереження очищеного набору для Tableau Public
    df.to_csv(PROCESSED_DIR / "superstore_cleaned.csv", index=False, encoding="utf-8-sig")
    return df, report


def run_statistical_analysis(df: pd.DataFrame, report: dict) -> dict:
    """Описова статистика, групування, ранжування, зведені таблиці та кореляції."""
    num_cols = ["Sales", "Quantity", "Discount", "Profit", "Profit_Margin_Pct", "Unit_Price"]
    desc = df[num_cols].describe().T
    desc["median"] = df[num_cols].median()
    desc = desc[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]].round(2)
    desc.to_csv(EXPORTS_DIR / "descriptive_stats.csv", encoding="utf-8-sig")

    # Загальні KPI
    report["kpi"] = {
        "total_sales": round(float(df["Sales"].sum()), 2),
        "total_profit": round(float(df["Profit"].sum()), 2),
        "overall_margin_pct": round(float(df["Profit"].sum() / df["Sales"].sum() * 100), 2),
        "total_quantity": int(df["Quantity"].sum()),
        "avg_order_sales": round(float(df["Sales"].mean()), 2),
        "median_order_sales": round(float(df["Sales"].median()), 2),
        "avg_profit": round(float(df["Profit"].mean()), 2),
        "median_profit": round(float(df["Profit"].median()), 2),
        "avg_discount_pct": round(float(df["Discount"].mean() * 100), 2),
        "profitable_orders": int((df["Profit"] > 0).sum()),
        "loss_orders": int((df["Profit"] < 0).sum()),
        "breakeven_orders": int((df["Profit"] == 0).sum()),
        "loss_orders_pct": round(float((df["Profit"] < 0).mean() * 100), 2),
    }

    # Групування за категоріями
    cat_summary = (
        df.groupby("Category")
        .agg(
            Orders=("Sales", "count"),
            Total_Sales=("Sales", "sum"),
            Avg_Sales=("Sales", "mean"),
            Total_Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean"),
            Total_Quantity=("Quantity", "sum"),
        )
        .reset_index()
    )
    cat_summary["Profit_Margin_Pct"] = (cat_summary["Total_Profit"] / cat_summary["Total_Sales"] * 100).round(2)
    cat_summary["Avg_Discount_Pct"] = (cat_summary["Avg_Discount"] * 100).round(2)
    cat_summary["Sales_Share_Pct"] = (cat_summary["Total_Sales"] / cat_summary["Total_Sales"].sum() * 100).round(2)
    cat_summary = cat_summary.sort_values("Total_Profit", ascending=False).round(2)
    cat_summary.to_csv(EXPORTS_DIR / "category_summary.csv", index=False, encoding="utf-8-sig")
    report["category_summary"] = cat_summary.to_dict(orient="records")

    # Групування та ранжування за підкатегоріями
    sub_summary = (
        df.groupby(["Category", "Sub-Category"])
        .agg(
            Orders=("Sales", "count"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean"),
            Total_Quantity=("Quantity", "sum"),
        )
        .reset_index()
    )
    sub_summary["Profit_Margin_Pct"] = (sub_summary["Total_Profit"] / sub_summary["Total_Sales"] * 100).round(2)
    sub_summary["Avg_Discount_Pct"] = (sub_summary["Avg_Discount"] * 100).round(2)
    sub_summary["Profit_Rank"] = sub_summary["Total_Profit"].rank(ascending=False).astype(int)
    sub_summary = sub_summary.sort_values("Total_Profit", ascending=False).round(2)
    sub_summary.to_csv(EXPORTS_DIR / "subcategory_summary.csv", index=False, encoding="utf-8-sig")
    report["top5_subcategories"] = sub_summary.head(5).to_dict(orient="records")
    report["bottom5_subcategories"] = sub_summary.tail(5).to_dict(orient="records")

    # Групування за регіонами та менеджерами (результат merge №1)
    reg_summary = (
        df.groupby(["Region", "Regional_Manager", "Regional_Hub"])
        .agg(
            Orders=("Sales", "count"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean"),
        )
        .reset_index()
    )
    reg_summary["Profit_Margin_Pct"] = (reg_summary["Total_Profit"] / reg_summary["Total_Sales"] * 100).round(2)
    reg_summary["Avg_Discount_Pct"] = (reg_summary["Avg_Discount"] * 100).round(2)
    reg_summary = reg_summary.sort_values("Total_Profit", ascending=False).round(2)
    reg_summary.to_csv(EXPORTS_DIR / "region_manager_summary.csv", index=False, encoding="utf-8-sig")
    report["region_summary"] = reg_summary.to_dict(orient="records")

    # Групування за сегментами
    seg_summary = (
        df.groupby("Segment")
        .agg(
            Orders=("Sales", "count"),
            Total_Sales=("Sales", "sum"),
            Avg_Sales=("Sales", "mean"),
            Total_Profit=("Profit", "sum"),
        )
        .reset_index()
    )
    seg_summary["Profit_Margin_Pct"] = (seg_summary["Total_Profit"] / seg_summary["Total_Sales"] * 100).round(2)
    seg_summary = seg_summary.sort_values("Total_Sales", ascending=False).round(2)
    seg_summary.to_csv(EXPORTS_DIR / "segment_summary.csv", index=False, encoding="utf-8-sig")
    report["segment_summary"] = seg_summary.to_dict(orient="records")

    # Групування за способом доставки (результат merge №2)
    ship_summary = (
        df.groupby(["Ship Mode", "SLA_Days", "Base_Shipping_Cost"])
        .agg(
            Orders=("Sales", "count"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
        )
        .reset_index()
    )
    ship_summary["Orders_Share_Pct"] = (ship_summary["Orders"] / ship_summary["Orders"].sum() * 100).round(2)
    ship_summary["Profit_Margin_Pct"] = (ship_summary["Total_Profit"] / ship_summary["Total_Sales"] * 100).round(2)
    ship_summary = ship_summary.sort_values("Orders", ascending=False).round(2)
    ship_summary.to_csv(EXPORTS_DIR / "shipping_summary.csv", index=False, encoding="utf-8-sig")
    report["shipping_summary"] = ship_summary.to_dict(orient="records")

    # Вплив рівня знижки на прибутковість
    disc_summary = (
        df.groupby("Discount_Tier", observed=False)
        .agg(
            Orders=("Sales", "count"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Avg_Profit=("Profit", "mean"),
            Avg_Margin_Pct=("Profit_Margin_Pct", "mean"),
        )
        .reset_index()
        .round(2)
    )
    disc_summary.to_csv(EXPORTS_DIR / "discount_impact.csv", index=False, encoding="utf-8-sig")
    report["discount_summary"] = disc_summary.to_dict(orient="records")

    # Топ-5 і антитоп-5 штатів за прибутком
    state_summary = (
        df.groupby("State")
        .agg(Orders=("Sales", "count"), Total_Sales=("Sales", "sum"), Total_Profit=("Profit", "sum"))
        .reset_index()
        .sort_values("Total_Profit", ascending=False)
        .round(2)
    )
    state_summary.to_csv(EXPORTS_DIR / "state_summary.csv", index=False, encoding="utf-8-sig")
    report["top5_states"] = state_summary.head(5).to_dict(orient="records")
    report["bottom5_states"] = state_summary.tail(5).to_dict(orient="records")

    # Зведена таблиця: Регіон x Категорія (сумарний прибуток)
    pivot_profit = pd.pivot_table(
        df,
        values="Profit",
        index="Region",
        columns="Category",
        aggfunc="sum",
        margins=True,
        margins_name="Разом",
    ).round(2)
    pivot_profit.to_csv(EXPORTS_DIR / "pivot_region_category_profit.csv", encoding="utf-8-sig")

    # Таблиця спряженості (crosstab): Сегмент x Статус прибутковості (%)
    ct_seg_status = pd.crosstab(df["Segment"], df["Profit_Status"], normalize="index").mul(100).round(2)
    ct_seg_status.to_csv(EXPORTS_DIR / "crosstab_segment_profit_status.csv", encoding="utf-8-sig")

    # Кореляційна матриця
    corr = df[["Sales", "Quantity", "Discount", "Profit", "Profit_Margin_Pct", "Unit_Price"]].corr().round(3)
    corr.to_csv(EXPORTS_DIR / "correlation_matrix.csv", encoding="utf-8-sig")
    report["correlations"] = {
        "sales_profit": float(corr.loc["Sales", "Profit"]),
        "discount_profit": float(corr.loc["Discount", "Profit"]),
        "discount_margin": float(corr.loc["Discount", "Profit_Margin_Pct"]),
        "quantity_sales": float(corr.loc["Quantity", "Sales"]),
    }

    with open(EXPORTS_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def build_visualizations(df: pd.DataFrame) -> None:
    """Побудова 7 змістовних графіків засобами Matplotlib та Seaborn."""
    # --- Графік 1: Розподіл обсягів продажів (лог-шкала) та прибутку ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(np.log10(df["Sales"]), bins=35, kde=True, color="#2b7bba", ax=axes[0])
    axes[0].set_title("Розподіл обсягів продажів (log10)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("log10(Продажі, USD)")
    axes[0].set_ylabel("Кількість замовлень")

    profit_clipped = df["Profit"].clip(-500, 500)
    sns.histplot(profit_clipped, bins=40, kde=True, color="#2ca02c", ax=axes[1])
    axes[1].axvline(0, color="#d62728", linestyle="--", linewidth=1.5, label="Точка беззбитковості ($0)")
    axes[1].set_title("Розподіл прибутку (інтервал -$500 ... +$500)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Прибуток, USD")
    axes[1].set_ylabel("Кількість замовлень")
    axes[1].legend()

    plt.suptitle("Розподіл ключових фінансових показників замовлень", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_sales_profit_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Графік 2: Продажі та прибуток за категоріями ---
    cat_agg = (
        df.groupby("Category")[["Sales", "Profit"]]
        .sum()
        .reset_index()
        .melt(id_vars="Category", var_name="Показник", value_name="Сума, USD")
    )
    cat_agg["Показник"] = cat_agg["Показник"].replace({"Sales": "Продажі", "Profit": "Прибуток"})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=cat_agg, x="Category", y="Сума, USD", hue="Показник", palette=["#3b82f6", "#10b981"], ax=ax)
    ax.set_title("Сумарні продажі та прибуток за категоріями товарів", fontsize=13, fontweight="bold")
    ax.set_xlabel("Категорія товарів")
    ax.set_ylabel("Сума, USD")
    for container in ax.containers:
        ax.bar_label(container, fmt="$%.0f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_category_sales_profit.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Графік 3: Рейтинг підкатегорій за прибутком ---
    sub_agg = df.groupby("Sub-Category")["Profit"].sum().sort_values()
    colors = ["#ef4444" if v < 0 else "#10b981" for v in sub_agg.values]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(sub_agg.index, sub_agg.values, color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Рейтинг підкатегорій товарів за сумарним прибутком", fontsize=13, fontweight="bold")
    ax.set_xlabel("Сумарний прибуток, USD")
    ax.set_ylabel("Підкатегорія")
    for bar in bars:
        w = bar.get_width()
        offset = 800 if w >= 0 else -800
        ha = "left" if w >= 0 else "right"
        ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"${w:,.0f}", va="center", ha=ha, fontsize=8.5)
    ax.set_xlim(sub_agg.min() - 12000, sub_agg.max() + 14000)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_subcategory_profit_ranking.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Графік 4: Вплив знижки на прибуток і маржинальність ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.scatterplot(
        data=df,
        x="Discount",
        y="Profit",
        hue="Category",
        alpha=0.55,
        s=30,
        ax=axes[0],
    )
    axes[0].axhline(0, color="#d62728", linestyle="--", linewidth=1.2)
    axes[0].set_title("Залежність прибутку замовлення від знижки", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Знижка (частка, 0.0–0.8)")
    axes[0].set_ylabel("Прибуток, USD")

    disc_avg = df.groupby("Discount_Tier", observed=False)["Profit_Margin_Pct"].mean().reset_index()
    bar_colors = ["#10b981" if v >= 0 else "#ef4444" for v in disc_avg["Profit_Margin_Pct"]]
    sns.barplot(data=disc_avg, x="Discount_Tier", y="Profit_Margin_Pct", hue="Discount_Tier", palette=bar_colors, legend=False, ax=axes[1])
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_title("Середня маржинальність за рівнем знижки", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Рівень знижки")
    axes[1].set_ylabel("Середня маржинальність, %")
    axes[1].tick_params(axis="x", rotation=15)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_discount_vs_profit.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Графік 5: Boxplot маржинальності за категоріями та сегментами ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(
        data=df,
        x="Category",
        y="Profit_Margin_Pct",
        hue="Segment",
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
        ax=ax,
    )
    ax.axhline(0, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_title("Розподіл маржинальності (%) за категоріями та сегментами клієнтів", fontsize=13, fontweight="bold")
    ax.set_xlabel("Категорія товарів")
    ax.set_ylabel("Маржинальність, %")
    ax.legend(title="Сегмент")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_profit_margin_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Графік 6: Теплова карта кореляцій ---
    corr_cols = ["Sales", "Quantity", "Discount", "Profit", "Profit_Margin_Pct", "Unit_Price"]
    corr_labels = ["Продажі", "Кількість", "Знижка", "Прибуток", "Маржа (%)", "Ціна за од."]
    corr = df[corr_cols].corr()
    corr.index = corr_labels
    corr.columns = corr_labels

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Кореляційна матриця числових показників продажів", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Графік 7: Теплова карта зведеної таблиці Регіон x Категорія ---
    pivot_df = pd.pivot_table(df, values="Profit", index="Region", columns="Category", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.heatmap(pivot_df, annot=True, fmt=",.0f", cmap="YlGn", linewidths=0.6, ax=ax)
    ax.set_title("Зведена таблиця сумарного прибутку (USD): Регіон × Категорія", fontsize=13, fontweight="bold")
    ax.set_xlabel("Категорія товарів")
    ax.set_ylabel("Регіон")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_pivot_region_category_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    sales_df, managers_df, shipping_df = load_data()
    df, report = clean_and_enrich(sales_df, managers_df, shipping_df)
    report = run_statistical_analysis(df, report)
    build_visualizations(df)

    print("Аналіз успішно завершено!")
    print(f"Рядків до очищення: {report['raw_rows']}, після видалення дублікатів: {report['clean_rows']}")
    print(f"Загальний обсяг продажів: ${report['kpi']['total_sales']:,.2f}")
    print(f"Загальний прибуток: ${report['kpi']['total_profit']:,.2f} (маржа {report['kpi']['overall_margin_pct']}%)")
    print(f"Збережено 7 графіків у {FIGURES_DIR}")


if __name__ == "__main__":
    main()

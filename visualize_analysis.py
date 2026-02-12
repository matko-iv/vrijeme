"""Vizualizacija rezultata analize vremenskih modela za Budvu."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

JSON_FILE = "budva_detailed_error_analysis.json"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'primary': '#0C5DA5',
    'secondary': '#FF6B35',
    'tertiary': '#00B945',
    'quaternary': '#845EC2',
    'quinary': '#D65DB1',
    'senary': '#F9B936',
    'gray': '#7A7A7A',
    'light_gray': '#E0E0E0',
    'positive': '#C73E1D',
    'negative': '#2E86AB',
}

MODEL_PALETTE = [
    COLORS['primary'],
    COLORS['secondary'], 
    COLORS['tertiary'],
    COLORS['quaternary'],
    COLORS['quinary'],
    COLORS['senary']
]

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'normal',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})


def load_reports(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        reports = json.load(f)
    return reports


def safe_get(nested_dict, key, default=0):
    val = nested_dict.get(key, default)
    return val if val is not None else default


def get_palette(n):
    return [MODEL_PALETTE[i % len(MODEL_PALETTE)] for i in range(n)]


def plot_model_performance_overview(reports):
    models = [r['model_name'] for r in reports]
    n_models = len(models)
    colors = get_palette(n_models)

    mae_temp = [safe_get(r['overall'].get('temperature_2m', {}), 'mae') for r in reports]
    mae_wind = [safe_get(r['overall'].get('wind_speed_10m', {}), 'mae') for r in reports]
    mae_precip = [safe_get(r['overall'].get('precipitation', {}), 'mae') for r in reports]
    mae_humidity = [safe_get(r['overall'].get('relative_humidity_2m', {}), 'mae') for r in reports]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pregled Performansi Globalnih Meteoroloških Modela za Budvu', 
                 fontsize=15, fontweight='bold', y=0.995)

    ax1 = axes[0, 0]
    bars1 = ax1.barh(models, mae_temp, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax1.set_xlabel('Mean Absolute Error (°C)', fontweight='bold')
    ax1.set_title('(A) Greška Temperature', loc='left', fontweight='bold')
    ax1.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars1, mae_temp)):
        if val > 0:
            ax1.text(val + 0.05, i, f'{val:.2f}°C', va='center', fontsize=9, fontweight='bold')

    ax2 = axes[0, 1]
    bars2 = ax2.barh(models, mae_wind, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax2.set_xlabel('Mean Absolute Error (m/s)', fontweight='bold')
    ax2.set_title('(B) Greška Brzine Vjetra', loc='left', fontweight='bold')
    ax2.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars2, mae_wind)):
        if val > 0:
            ax2.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')

    ax3 = axes[1, 0]
    bars3 = ax3.barh(models, mae_precip, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax3.set_xlabel('Mean Absolute Error (mm)', fontweight='bold')
    ax3.set_title('(C) Greška Padavina', loc='left', fontweight='bold')
    ax3.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars3, mae_precip)):
        if val > 0:
            ax3.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')

    ax4 = axes[1, 1]
    bars4 = ax4.barh(models, mae_humidity, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax4.set_xlabel('Mean Absolute Error (%)', fontweight='bold')
    ax4.set_title('(D) Greška Relativne Vlažnosti', loc='left', fontweight='bold')
    ax4.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars4, mae_humidity)):
        if val > 0:
            ax4.text(val + 0.2, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_model_performance_overview.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ 01_model_performance_overview.png")
    plt.close()


def plot_systematic_bias(reports):
    models = [r['model_name'] for r in reports]

    bias_temp = [safe_get(r['overall'].get('temperature_2m', {}), 'bias') for r in reports]
    bias_wind = [safe_get(r['overall'].get('wind_speed_10m', {}), 'bias') for r in reports]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sistematske Greške (Bias) Modela', fontsize=14, fontweight='bold')

    ax1 = axes[0]
    colors_temp = [COLORS['positive'] if b > 0 else COLORS['negative'] for b in bias_temp]
    bars1 = ax1.barh(models, bias_temp, color=colors_temp, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Bias (°C)', fontweight='bold')
    ax1.set_title('(A) Temperatura', loc='left', fontweight='bold')
    ax1.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars1, bias_temp)):
        label = 'Pregrijava' if val > 0 else 'Pothlađuje'
        ax1.text(val + (0.05 if val > 0 else -0.05), i, f'{val:+.2f}°C', 
                va='center', ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')

    red_patch = mpatches.Patch(color=COLORS['positive'], alpha=0.8, label='Pregrijava (pozitivan bias)')
    blue_patch = mpatches.Patch(color=COLORS['negative'], alpha=0.8, label='Pothlađuje (negativan bias)')
    ax1.legend(handles=[red_patch, blue_patch], loc='lower right', framealpha=0.95, fontsize=8)

    ax2 = axes[1]
    colors_wind = [COLORS['positive'] if b > 0 else COLORS['negative'] for b in bias_wind]
    bars2 = ax2.barh(models, bias_wind, color=colors_wind, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Bias (m/s)', fontweight='bold')
    ax2.set_title('(B) Brzina Vjetra', loc='left', fontweight='bold')
    ax2.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars2, bias_wind)):
        ax2.text(val + (0.05 if val > 0 else -0.05), i, f'{val:+.2f}', 
                va='center', ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')

    red_patch = mpatches.Patch(color=COLORS['positive'], alpha=0.8, label='Precjenjuje')
    blue_patch = mpatches.Patch(color=COLORS['negative'], alpha=0.8, label='Potcjenjuje')
    ax2.legend(handles=[red_patch, blue_patch], loc='lower right', framealpha=0.95, fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_systematic_bias.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ 02_systematic_bias.png")
    plt.close()


def plot_seasonal_analysis(reports):
    seasons = ['Zima', 'Proleće', 'Leto', 'Jesen']
    n_models = len(reports)
    colors = get_palette(n_models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sezonska Analiza Grešaka', fontsize=14, fontweight='bold')

    x = np.arange(len(seasons))
    width = 0.75 / n_models

    ax1 = axes[0]
    for i, report in enumerate(reports):
        model_name = report['model_name']
        by_season = report.get('by_season', {})
        mae_values = [safe_get(by_season.get(s, {}).get('temperature_2m', {}), 'mae') for s in seasons]

        offset = (i - n_models/2 + 0.5) * width
        ax1.bar(x + offset, mae_values, width, label=model_name, 
               color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)

    ax1.set_xlabel('Sezona', fontweight='bold')
    ax1.set_ylabel('MAE (°C)', fontweight='bold')
    ax1.set_title('(A) Temperatura po Sezonama', loc='left', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seasons)
    ax1.legend(loc='upper left', framealpha=0.95, fontsize=8)

    ax2 = axes[1]
    for i, report in enumerate(reports):
        model_name = report['model_name']
        by_season = report.get('by_season', {})
        mae_values = [safe_get(by_season.get(s, {}).get('precipitation', {}), 'mae') for s in seasons]

        offset = (i - n_models/2 + 0.5) * width
        ax2.bar(x + offset, mae_values, width, label=model_name,
               color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)

    ax2.set_xlabel('Sezona', fontweight='bold')
    ax2.set_ylabel('MAE (mm)', fontweight='bold')
    ax2.set_title('(B) Padavine po Sezonama', loc='left', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seasons)
    ax2.legend(loc='upper left', framealpha=0.95, fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_seasonal_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ 03_seasonal_analysis.png")
    plt.close()


def plot_diurnal_cycle(reports):
    time_periods = ['Noć\n(22-06h)', 'Jutro\n(06-12h)', 'Popodne\n(12-18h)', 'Veče\n(18-22h)']
    time_keys = ['Noć', 'Jutro', 'Popodne', 'Veče']
    n_models = len(reports)
    colors = get_palette(n_models)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle('Dnevni Ciklus Grešaka Temperature', fontsize=14, fontweight='bold')

    x = np.arange(len(time_periods))
    width = 0.75 / n_models

    for i, report in enumerate(reports):
        model_name = report['model_name']
        by_time = report.get('by_time_of_day', {})
        mae_values = [safe_get(by_time.get(t, {}).get('temperature_2m', {}), 'mae') for t in time_keys]

        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, mae_values, width, label=model_name,
              color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Doba Dana', fontweight='bold')
    ax.set_ylabel('MAE Temperature (°C)', fontweight='bold')
    ax.set_title('Greške Temperature po Dobu Dana', loc='left', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(time_periods)
    ax.legend(loc='upper right', framealpha=0.95, ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_diurnal_cycle.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ 04_diurnal_cycle.png")
    plt.close()


def plot_extreme_conditions(reports):
    if not reports or 'by_wind_conditions' not in reports[0]:
        print("⚠️  Preskajem ekstremne uslove (nema podataka)")
        return

    key_conditions = ['Jak vjetar (>8 m/s)', 'BURA (SJ/SJI >8 m/s)']

    available_conditions = []
    for cond in key_conditions:
        if cond in reports[0].get('by_wind_conditions', {}):
            available_conditions.append(cond)

    if not available_conditions:
        print("⚠️  Preskajem ekstremne uslove (nema traženih uslova)")
        return

    n_models = len(reports)
    colors = get_palette(n_models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Performanse u Ekstremnim Uslovima', fontsize=14, fontweight='bold')

    x = np.arange(len(available_conditions))
    width = 0.75 / n_models

    ax1 = axes[0]
    for i, report in enumerate(reports):
        model_name = report['model_name']
        by_wind = report.get('by_wind_conditions', {})
        mae_values = [safe_get(by_wind.get(c, {}).get('temperature_2m', {}), 'mae') for c in available_conditions]

        offset = (i - n_models/2 + 0.5) * width
        ax1.bar(x + offset, mae_values, width, label=model_name,
               color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)

    ax1.set_xlabel('Uslov', fontweight='bold')
    ax1.set_ylabel('MAE (°C)', fontweight='bold')
    ax1.set_title('(A) Temperatura', loc='left', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace(' (>8 m/s)', '').replace(' (SJ/SJI >8 m/s)', '') for c in available_conditions], fontsize=9)
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=8)

    ax2 = axes[1]
    for i, report in enumerate(reports):
        model_name = report['model_name']
        by_wind = report.get('by_wind_conditions', {})
        mae_values = [safe_get(by_wind.get(c, {}).get('precipitation', {}), 'mae') for c in available_conditions]

        offset = (i - n_models/2 + 0.5) * width
        ax2.bar(x + offset, mae_values, width, label=model_name,
               color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)

    ax2.set_xlabel('Uslov', fontweight='bold')
    ax2.set_ylabel('MAE (mm)', fontweight='bold')
    ax2.set_title('(B) Padavine', loc='left', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace(' (>8 m/s)', '').replace(' (SJ/SJI >8 m/s)', '') for c in available_conditions], fontsize=9)
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_extreme_conditions.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ 05_extreme_conditions.png")
    plt.close()


def plot_model_comparison_matrix(reports):
    models = [r['model_name'] for r in reports]

    metrics_data = []
    metric_names = ['Temp\nMAE', 'Temp\nBias', 'Vjetar\nMAE', 'Vjetar\nBias', 
                    'Padavine\nMAE', 'Vlaga\nMAE']

    for report in reports:
        row = [
            safe_get(report['overall'].get('temperature_2m', {}), 'mae'),
            safe_get(report['overall'].get('temperature_2m', {}), 'bias'),
            safe_get(report['overall'].get('wind_speed_10m', {}), 'mae'),
            safe_get(report['overall'].get('wind_speed_10m', {}), 'bias'),
            safe_get(report['overall'].get('precipitation', {}), 'mae'),
            safe_get(report['overall'].get('relative_humidity_2m', {}), 'mae'),
        ]
        metrics_data.append(row)

    df = pd.DataFrame(metrics_data, columns=metric_names, index=models)

    df_norm = df.copy()
    for col in df_norm.columns:
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        if col_max > col_min:
            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(df_norm.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_yticklabels(models, fontsize=10)

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    for i in range(len(models)):
        for j in range(len(metric_names)):
            text = ax.text(j, i, f'{df.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')

    ax.set_title('Matrica Poređenja Modela (Normalizovano)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Metrika', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relativna Performansa\n(0=najbolje, 1=najgore)', rotation=270, labelpad=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_model_comparison_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ 06_model_comparison_matrix.png")
    plt.close()


def create_summary_report(reports):
    best_model = min(reports, key=lambda r: safe_get(r['overall'].get('temperature_2m', {}), 'mae'))

    report_text = f"""

Analizirano modela: {len(reports)}


NAJBOLJI MODEL: {best_model['model_name']}

   Temperatura:
     • MAE:  {safe_get(best_model['overall'].get('temperature_2m', {}), 'mae'):.2f}°C
     • RMSE: {safe_get(best_model['overall'].get('temperature_2m', {}), 'rmse'):.2f}°C
     • Bias: {safe_get(best_model['overall'].get('temperature_2m', {}), 'bias'):+.2f}°C

   Brzina vjetra:
     • MAE:  {safe_get(best_model['overall'].get('wind_speed_10m', {}), 'mae'):.2f} m/s
     • Bias: {safe_get(best_model['overall'].get('wind_speed_10m', {}), 'bias'):+.2f} m/s

   Padavine:
     • MAE:  {safe_get(best_model['overall'].get('precipitation', {}), 'mae'):.2f} mm
     • Bias: {safe_get(best_model['overall'].get('precipitation', {}), 'bias'):+.2f} mm


RANGIRANJE MODELA (po MAE temperature):

"""

    sorted_models = sorted(reports, key=lambda r: safe_get(r['overall'].get('temperature_2m', {}), 'mae'))

    for i, model in enumerate(sorted_models, 1):
        mae = safe_get(model['overall'].get('temperature_2m', {}), 'mae')
        bias = safe_get(model['overall'].get('temperature_2m', {}), 'bias')
        report_text += f"   {i}. {model['model_name']:25s} MAE: {mae:5.2f}°C   Bias: {bias:+5.2f}°C\n"

    report_text += """
"""

    with open(f'{OUTPUT_DIR}/00_SUMMARY_REPORT.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("✅ 00_SUMMARY_REPORT.txt")
    print(report_text)


def main():
    reports = load_reports(JSON_FILE)
    if not reports:
        print("Nema podataka.")
        return

    print(f"Učitano {len(reports)} modela, kreiram vizualizacije...")

    create_summary_report(reports)
    plot_model_performance_overview(reports)
    plot_systematic_bias(reports)
    plot_seasonal_analysis(reports)
    plot_diurnal_cycle(reports)
    plot_extreme_conditions(reports)
    plot_model_comparison_matrix(reports)

    print(f"Gotovo. Vizualizacije u {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

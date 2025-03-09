# visualisation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import ipywidgets as widgets
import seaborn as sns
from IPython.display import display, clear_output
from scipy.stats import skew, kurtosis, shapiro, normaltest


################################
# Trimming & Standardizing
################################

def trim_data(x, col_name):
    if x[col_name].max() - x[col_name].min() > 1:
        lower, upper = x[col_name].quantile([0.001, 0.999])
        return x[(x[col_name] >= lower) & (x[col_name] <= upper)]
    return x.copy()

def standardize_data(x):
    scaler = StandardScaler(with_std=False)
    return scaler.fit_transform(x)


################################
# Extra Normality Metrics
################################

def compute_normality_metrics(data):
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    elif isinstance(data, pd.Series):
        data = data.values
    data = np.array(data).flatten()

    skew_val = skew(data)
    kurt_val = kurtosis(data)

    if len(data) <= 5000:
        try:
            _, p_value = shapiro(data)
            test_used = "Shapiro-Wilk"
        except Exception:
            p_value = np.nan
            test_used = "Shapiro-Wilk (failed)"
    else:
        try:
            _, p_value = normaltest(data)
            test_used = "D'Agostino K-squared"
        except Exception:
            p_value = np.nan
            test_used = "D'Agostino K-squared (failed)"

    return {
        "Skewness": skew_val,
        "Excess Kurtosis": kurt_val,
        "Normality Test p-value": p_value,
        "Test Used": test_used
    }


################################
# Core Transformation
################################

def process_column_final(df, col_name, skew_threshold=1.0):
    x_full = df[[col_name]]
    
    x_trim = trim_data(x_full, col_name)
    x_trim_std = standardize_data(x_trim)

    x_full_log = np.sign(x_full) * np.log2(1 + np.abs(x_full))
    x_full_log_trim = trim_data(x_full_log, col_name)
    x_full_log_trim_std = standardize_data(x_full_log_trim)

    original_skewness = x_full[col_name].skew()
    log_skewness = x_full_log[col_name].skew()

    print(f"Skewness of '{col_name}' (original data, no trimming): {original_skewness:.4f}")
    print(f"Skewness of '{col_name}' (log-transformed, no trimming): {log_skewness:.4f}")

    if original_skewness > skew_threshold:
        x_final = x_full_log
        print(f"<Full dataset without trimming> Transformation of '{col_name}' proposed with sign-preserving log₂.")
    else:
        x_final = x_full.copy()
        print(f"<Full dataset without trimming> No log transformation of '{col_name}' proposed.")

    x_final_trim = trim_data(x_final, col_name)
    x_final_trim_std = standardize_data(x_final_trim)

    return (
        x_trim, x_trim_std,
        x_full_log_trim, x_full_log_trim_std,
        x_final_trim, x_final_trim_std
    )


################################
# Plotting with Normality Metrics
################################

def _plot_hist(ax, data, title, xlab):
    if isinstance(data, pd.DataFrame):
        hist_data = data.values.flatten()
    elif isinstance(data, np.ndarray):
        hist_data = data.flatten()
    else:
        hist_data = np.array(data).flatten()
        
    ax.hist(hist_data, bins=100, edgecolor='black')
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlab, fontsize=6)
    ax.set_ylabel("Frequency", fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=5)
    
    metrics = compute_normality_metrics(data)
    metrics_text = (f"Skew: {metrics['Skewness']:.2f}\n"
                    f"Kurtosis: {metrics['Excess Kurtosis']:.2f}\n"
                    f"Normality p: {metrics['Normality Test p-value']:.2f}")
    ax.text(0.95, 0.95, metrics_text, transform=ax.transAxes, fontsize=5,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

def plot_six_histograms(df, col_name):
    results = process_column_final(df, col_name)
    (x_trim, x_trim_std,
     x_full_log_trim, x_full_log_trim_std,
     x_final_trim, x_final_trim_std) = results

    fig, axs = plt.subplots(3, 2, figsize=(12, 5))
    
    _plot_hist(axs[0, 0], x_trim,    f"Trimmed Original ({col_name})",    "Value")
    _plot_hist(axs[0, 1], x_trim_std,       f"Std Trimmed ({col_name})",         "Std Value")

    _plot_hist(axs[1, 0], x_full_log_trim, f"Log₂ Transformed ({col_name})",    "Log₂ Value")
    _plot_hist(axs[1, 1], x_full_log_trim_std,    f"Std Log₂ Transformed ({col_name})", "Std Log₂ Value")

    _plot_hist(axs[2, 0], x_final_trim,    f"Final Data ({col_name})",    "Value")
    _plot_hist(axs[2, 1], x_final_trim_std,       f"Std Final Data ({col_name})","Std Value")

    plt.tight_layout()
    plt.show()


#######################################################
# Navigation Widgets
#######################################################

plot_output = widgets.Output()

current_index = 0
relevant_columns = []
df_for_nav = None

next_button = widgets.Button(description="Next")
back_button = widgets.Button(description="Back")

def update_display():
    """Displays the current column’s plot inside the 'plot_output' widget."""
    with plot_output:
        clear_output(wait=True)
        col = relevant_columns[current_index]
        print("Current Column:", col)
        plot_six_histograms(df_for_nav, col)

def next_column(b):
    global current_index
    current_index = (current_index + 1) % len(relevant_columns)
    update_display()

def prev_column(b):
    global current_index
    current_index = (current_index - 1) % len(relevant_columns)
    update_display()

next_button.on_click(next_column)
back_button.on_click(prev_column)

def run_navigation(df, columns):
    global df_for_nav, relevant_columns, current_index
    df_for_nav = df
    relevant_columns = columns
    current_index = 0

    display(
        widgets.VBox([
            widgets.HBox([back_button, next_button]),
            plot_output
        ])
    )
    update_display()


#######################################################
# Correlation Matrix Plot
#######################################################

def plot_correlation_matrix(df, relevant_columns):
    plt.figure(figsize=(16, 8))
    heatmap_corr = sns.heatmap(df[relevant_columns].corr(), annot=True, cmap="Blues",
                               fmt=".2f", center=0, annot_kws={"fontsize": 7})
    plt.title('Correlation Matrix of Features')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.show()
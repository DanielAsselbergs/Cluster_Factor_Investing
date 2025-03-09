"""
Module: preprocessing.py
Description: Contains functions to read, preprocess the data,
             and generate report outputs (LaTeX tables) from the raw data.
             This version consolidates several smaller helper functions into
             larger ones to simplify the code structure.
"""

import os
import pandas as pd
import numpy as np
from tabulate import tabulate

def load_and_preprocess_data(data_folder, universe=500):
    file_paths = [os.path.join(data_folder, f'chunk_{i}.parquet') for i in [1, 2]]
    
    dfs = [pd.read_parquet(fp) for fp in file_paths]
    df = pd.concat(dfs, ignore_index=False)
    df.reset_index(inplace=True)
    
    df['date'] = pd.to_datetime(df['date'])
    
    cols = df.columns.tolist()
    if len(cols) > 9:
        new_order = cols[:6] + cols[-3:] + cols[6:-3]
        df = df[new_order]
    ret_cols = [col for col in df.columns if 'RET_' in col]
    non_ret_cols = [col for col in df.columns if 'RET_' not in col]
    df = df[non_ret_cols + ret_cols]
    
    df = df.dropna(subset=['company_name'])
    df = df[df['Country'] == 'United States']
    for col in ['ison_univ', 'NTM Revenue / Employee']:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df[df['FR RBICS Name Economy'] != 'Non-Corporate']
    
    stocks_per_date = df.groupby('date')['company_name'].count()
    if (stocks_per_date < universe).any():
        print(f"Warning: Some dates have fewer than {universe} stocks.")
        print(stocks_per_date[stocks_per_date < universe].to_string())
    else:
        print(f"All dates have at least {universe} stocks.")
    
    df = df.sort_values(['date', 'Market Value'], ascending=[True, False])
    df = df.groupby('date', group_keys=False).head(universe)
    
    fill_cols = ['FR RBICS Num Economy', 'FR RBICS Name Economy',
                 'FR RBICS Num Sector', 'FR RBICS Name Sector']
    df = df.sort_values(['company_name', 'date'])
    for col in fill_cols:
        if col in df.columns:
            df[col] = df.groupby('company_name')[col].ffill().bfill()
    
    df['year_period'] = df['date'].dt.year.apply(lambda y: f"{(y // 5) * 5}-{(y // 5) * 5 + 4}")
    
    return df

def generate_reports(df, relevant_cols, output_folder=None):
    try:
        base_folder = os.path.dirname(__file__)
    except NameError:
        base_folder = os.getcwd()

    if output_folder is None:
        output_folder = os.path.abspath(os.path.join(base_folder, "..", "Output"))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mapping_info = {
        'economy': ['FR RBICS Name Economy', 'FR RBICS Num Economy'],
        'sector': ['FR RBICS Name Sector', 'FR RBICS Num Sector']
    }
    for key, cols in mapping_info.items():
        mapping = (df[cols]
                   .dropna()
                   .drop_duplicates()
                   .sort_values(cols[0])
                   .reset_index(drop=True))
        _write_latex_table(mapping, f"rbics_{key}.tex", output_folder)

    var_descr = pd.DataFrame({'Column Name': relevant_cols, 'Description': '', 'Factor': ''})
    _write_latex_table(var_descr, "columns.tex", output_folder)

    daily_counts = df.groupby(['date', 'FR RBICS Name Economy']).size().reset_index(name='Count')
    daily_counts['year_period'] = daily_counts['date'].dt.year.apply(
        lambda y: f"{(y // 5) * 5}-{(y // 5) * 5 + 4}"
    )
    average_counts = daily_counts.groupby(['year_period', 'FR RBICS Name Economy'])['Count'].mean().reset_index()
    rbics_pivot = average_counts.pivot(index='FR RBICS Name Economy', columns='year_period', values='Count')
    rbics_pivot = rbics_pivot.fillna(0).round(2)
    rbics_pivot.index.name = 'RBICS'
    _write_latex_table(rbics_pivot.reset_index(), "rbics.tex", output_folder)

    missing_by_period = df.groupby('year_period')[relevant_cols].apply(lambda x: x.isnull().mean() * 100)
    missing_period_table = missing_by_period.T.reset_index().round(2)
    _write_latex_table(missing_period_table, "missing_values_period.tex", output_folder)

    missing_by_sector = df.groupby('FR RBICS Name Economy')[relevant_cols].apply(lambda x: x.isnull().mean() * 100).round(2)
    missing_sector_table = missing_by_sector.reset_index()
    _write_latex_table(missing_sector_table, "missing_values_sector.tex", output_folder)

def _write_latex_table(df, filename, output_folder):
    output_file = os.path.join(output_folder, filename)
    latex_table = tabulate(df, headers='keys', tablefmt='latex', showindex=False)
    with open(output_file, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table written to: {output_file}")
    
def adjust_dataframe(df, relevant_columns, skew_threshold=1.0):
    df_adjusted = df.copy()
    for col in relevant_columns:
        skewness = df_adjusted[col].skew()
        if skewness > skew_threshold:
            df_adjusted[col] = np.sign(df_adjusted[col]) * np.log2(1 + np.abs(df_adjusted[col]))
            print(f"Column '{col}' transformed using sign-preserving log₂ (skewness = {skewness:.2f}).")
        else:
            print(f"Column '{col}' left unchanged (skewness = {skewness:.2f}).")
    return df_adjusted


#######################################################
# Generate Summary Statistics Report
#######################################################

def generate_summary_statistics(df, relevant_columns, output_folder=None):
    try:
        base_folder = os.path.dirname(__file__)
    except NameError:
        base_folder = os.getcwd()
        
    if output_folder is None:
        output_folder = os.path.abspath(os.path.join(base_folder, "..", "Output"))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    def kurtosis_custom(x):
        return x.kurtosis()
    
    def skewness_custom(x):
        return x.skew()
    
    def percentile_5(x):
        return x.quantile(0.05)
    
    def percentile_95(x):
        return x.quantile(0.95)
    
    summary_statistics = df[relevant_columns].agg([
        'min', 'max', 'std', 'median', 'mean', 'var',
        kurtosis_custom, skewness_custom,
        percentile_5, percentile_95
    ]).round(2)
    summary_statistics = summary_statistics.T
    
    csv_file = os.path.join(output_folder, 'FEAT_summarystats.csv')
    summary_statistics.to_csv(csv_file)
    
    _write_latex_table(summary_statistics, "summary_stats.tex", output_folder)
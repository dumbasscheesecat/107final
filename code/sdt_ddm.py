"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output" #you put output dir within a function so im putting one outside also
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping dictionaries for categorical variables
# These convert categorical labels to numeric codes for analysis
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format.
    
    Args:
        file_path: Path to the CSV file containing raw response data
        prepare_for: Type of analysis to prepare data for ('sdt' or 'delta plots')
        display: Whether to print summary statistics
        
    Returns:
        DataFrame with processed data in the requested format
    """
    # Read and preprocess data
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print("\nUnique conditions:", data['condition'].unique())
        print("Signal values:", data['signal'].unique())
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        if display:
            print("\nGrouped data:")
            print(grouped.head())
        
        # Transform into SDT format (hits, misses, false alarms, correct rejections)
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        
        data = pd.DataFrame(sdt_data)
        
        if display:
            print("\nSDT summary:")
            print(data)
            if data.empty:
                print("\nWARNING: Empty SDT summary generated!")
                print("Number of participants:", len(data['pnum'].unique()))
                print("Number of conditions:", len(data['condition'].unique()))
            else:
                print("\nSummary statistics:")
                print(data.groupby('condition').agg({
                    'hits': 'sum',
                    'misses': 'sum',
                    'false_alarms': 'sum',
                    'correct_rejections': 'sum',
                    'nSignal': 'sum',
                    'nNoise': 'sum'
                }).round(2))
    
    # Prepare data for delta plot analysis
    if prepare_for == 'delta plots':
        # Initialize DataFrame for delta plot data
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', 
                                      *[f'p{p}' for p in PERCENTILES]])
        
        # Process data for each participant and condition
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                # Get data for this participant and condition
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                
                # Calculate percentiles for overall RTs
                overall_rt = c_data['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['overall'],
                    **{f'p{p}': [np.percentile(overall_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for accurate responses
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['accurate'],
                    **{f'p{p}': [np.percentile(accurate_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for error responses
                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['error'],
                    **{f'p{p}': [np.percentile(error_rt, p)] for p in PERCENTILES}
                })])
                
        if display:
            print("\nDelta plots data:")
            print(dp_data)
            
        data = pd.DataFrame(dp_data)

    return data


def apply_hierarchical_sdt_model(data):
    """Apply a hierarchical Signal Detection Theory model using PyMC.
    
    This function implements a Bayesian hierarchical model for SDT analysis,
    allowing for both group-level and individual-level parameter estimation.
    
    Args:
        data: DataFrame containing SDT summary statistics
        
    Returns:
        PyMC model object
    """
    # Get unique participants and conditions
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())
    
    # Define the hierarchical model
    with pm.Model() as sdt_model:
        # Group-level parameters
        mean_d_prime = pm.Normal('mean_d_prime', mu=0.0, sigma=1.0, shape=C)
        stdev_d_prime = pm.HalfNormal('stdev_d_prime', sigma=1.0)
        
        mean_criterion = pm.Normal('mean_criterion', mu=0.0, sigma=1.0, shape=C)
        stdev_criterion = pm.HalfNormal('stdev_criterion', sigma=1.0)
        
        # Individual-level parameters
        d_prime = pm.Normal('d_prime', mu=mean_d_prime, sigma=stdev_d_prime, shape=(P, C))
        criterion = pm.Normal('criterion', mu=mean_criterion, sigma=stdev_criterion, shape=(P, C))
        
        # Calculate hit and false alarm rates using SDT
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)
                
        # Likelihood for signal trials
        # Note: pnum is 1-indexed in the data, but needs to be 0-indexed for the model, so we change the indexing here.  The results table will show participant numbers starting from 0, so we need to interpret the results accordingly.
        pm.Binomial('hit_obs', 
                   n=data['nSignal'], 
                   p=hit_rate[data['pnum']-1, data['condition']], 
                   observed=data['hits'])
        
        # Likelihood for noise trials
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'], 
                   p=false_alarm_rate[data['pnum']-1, data['condition']], 
                   observed=data['false_alarms'])
    
    return sdt_model

def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs.
    
    Creates a matrix of delta plots where:
    - Upper triangle shows overall RT distribution differences
    - Lower triangle shows RT differences split by correct/error responses
    
    Args:
        data: DataFrame with RT percentile data
        pnum: Participant number to plot
    """
    # Filter data for specified participant
    data = data[data['pnum'] == pnum]
    
    # Get unique conditions and create subplot matrix
    conditions = data['condition'].unique()
    n_conditions = len(conditions)
    
    # Create figure with subplots matrix
    fig, axes = plt.subplots(n_conditions, n_conditions, 
                            figsize=(4*n_conditions, 4*n_conditions))
    
    # Create output directory
    OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define marker style for plots
    marker_style = {
        'marker': 'o',
        'markersize': 10,
        'markerfacecolor': 'white',
        'markeredgewidth': 2,
        'linewidth': 3
    }
    
    # Create delta plots for each condition pair
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            # Add labels only to edge subplots
            if j == 0:
                axes[i,j].set_ylabel('Difference in RT (s)', fontsize=12)
            if i == len(axes)-1:
                axes[i,j].set_xlabel('Percentile', fontsize=12)
                
            # Skip diagonal and lower triangle for overall plots
            if i > j:
                continue
            if i == j:
                axes[i,j].axis('off')
                continue
            
            # Create masks for condition and plotting mode
            cmask1 = data['condition'] == cond1
            cmask2 = data['condition'] == cond2
            overall_mask = data['mode'] == 'overall'
            error_mask = data['mode'] == 'error'
            accurate_mask = data['mode'] == 'accurate'
            
            # Calculate RT differences for overall performance
            quantiles1 = [data[cmask1 & overall_mask][f'p{p}'] for p in PERCENTILES]
            quantiles2 = [data[cmask2 & overall_mask][f'p{p}'] for p in PERCENTILES]
            overall_delta = np.array(quantiles2) - np.array(quantiles1)
            
            # Calculate RT differences for error responses
            error_quantiles1 = [data[cmask1 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_quantiles2 = [data[cmask2 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_delta = np.array(error_quantiles2) - np.array(error_quantiles1)
            
            # Calculate RT differences for accurate responses
            accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
            
            # Plot overall RT differences
            axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
            
            # Plot error and accurate RT differences
            axes[j,i].plot(PERCENTILES, error_delta, color='red', **marker_style)
            axes[j,i].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
            axes[j,i].legend(['Error', 'Accurate'], loc='upper left')

            # Set y-axis limits and add reference line
            axes[i,j].set_ylim(bottom=-1/3, top=1/2)
            axes[j,i].set_ylim(bottom=-1/3, top=1/2)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
            axes[j,i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add condition labels
            axes[i,j].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            axes[j,i].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            plt.tight_layout()
            
    # Save the figure
    plt.savefig(OUTPUT_DIR / f'delta_plots_{pnum}.png')

#MY STUFFS

def sample_posterior(model, draws, tune, chains, target_accept): 
    with model: 
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept)
    return idata

#showing hit rate and false alarm rate for original dataset
def show_summary_sdt(data):
    sdt_df = data
    sdt_df['hit_rate'] = sdt_df['hits'] / sdt_df['nSignal']
    sdt_df['fa_rate'] = sdt_df['false_alarms'] / sdt_df['nNoise']

    sdt_df.to_csv(OUTPUT_DIR / "original_data_summary.csv", index=False)

def analyze_results(idata, data):
    print('Hierarchical SDT Model Summary')
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())

    #converge checking
    print("\nConvergence Check")
    summary = az.summary(idata, var_names=["mean_d_prime", "mean_criterion", "stdev_d_prime", "stdev_criterion"], hdi_prob=0.94)
    print(summary)

    az.plot_trace(idata, var_names=["mean_d_prime", "mean_criterion", "stdev_d_prime", "stdev_criterion"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trace_group_params.png")
    plt.close()

    az.plot_posterior(idata, var_names=["mean_d_prime", "mean_criterion", "stdev_d_prime", "stdev_criterion"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "posterior_group_params.png")
    plt.close()

    #posterior distributions, conversion to dataframe by gpt

    d = idata.posterior['d_prime'].mean(dim=('chain', 'draw')).values
    c = idata.posterior['criterion'].mean(dim=('chain', 'draw')).values
    n_participants, n_conditions = d.shape

    d_df = pd.DataFrame(d, columns=[f"d_{i}" for i in range(n_conditions)])
    d_df['pnum'] = range(1, n_participants + 1)
    c_df = pd.DataFrame(c, columns=[f"c_{i}" for i in range(n_conditions)])
    c_df['pnum'] = range(1, n_participants + 1)

    full_df = pd.merge(d_df, c_df, on='pnum')
    full_df.to_csv(OUTPUT_DIR / "participant_condition_posterior_estimates.csv", index=False)

    az.plot_forest(idata, var_names=['d_prime', 'criterion'], combined=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forest_plot_d_c.png")
    plt.close()

    #min and max false alarm and hit rate of idata
    hit_rates = 1 / (1 + np.exp(-(d - c)))  #gpt implementation 
    fa_rates = 1 / (1 + np.exp(c))          

    hr_df = pd.DataFrame(hit_rates, columns=[f"hit_{i}" for i in range(hit_rates.shape[1])])
    fa_df = pd.DataFrame(fa_rates, columns=[f"fa_{i}" for i in range(fa_rates.shape[1])])
    hr_df['pnum'] = range(1, hit_rates.shape[0] + 1)
    fa_df['pnum'] = range(1, fa_rates.shape[0] + 1)

    rates_df = pd.merge(hr_df, fa_df, on='pnum') #gpt for format
    hr_long = hr_df.melt(id_vars='pnum', var_name='condition', value_name='hit_rate')
    fa_long = fa_df.melt(id_vars='pnum', var_name='condition', value_name='fa_rate')
    hr_long['condition'] = hr_long['condition'].str.extract(r'(\d+)').astype(int)
    fa_long['condition'] = fa_long['condition'].str.extract(r'(\d+)').astype(int)

    rate_df = pd.merge(hr_long, fa_long, on=['pnum', 'condition'])

    rate_summary = rate_df.groupby("condition").agg(
        max_hit_rate=('hit_rate', 'max'),
        min_hit_rate=('hit_rate', 'min'),
        max_fa_rate=('fa_rate', 'max'),
        min_fa_rate=('fa_rate', 'min')
    )

    idxs = {
        'max_hit_rate': rate_df.groupby("condition")['hit_rate'].idxmax(),
        'min_hit_rate': rate_df.groupby("condition")['hit_rate'].idxmin(),
        'max_fa_rate': rate_df.groupby("condition")['fa_rate'].idxmax(),
        'min_fa_rate': rate_df.groupby("condition")['fa_rate'].idxmin(),
    }
    for key, idx in idxs.items():
        rate_summary[f"{key}_participant"] = rate_df.loc[idx, 'pnum'].values

    rate_summary.to_csv(OUTPUT_DIR / "max_min_posterior_rates.csv")
    rate_df.to_csv(OUTPUT_DIR / "participant_condition_posterior_rates.csv", index=False) #gpt's code to separate out to dfs for ease of viewing information


    #final summary for comparison like with stroop.py, first is discriminability
    mu_d_samples = idata.posterior['mean_d_prime']

    d_contrasts = { #this setup by gpt
        'Easy_vs_Hard': mu_d_samples.sel(mean_d_prime_dim_0=[0, 1]).mean(dim='mean_d_prime_dim_0') -
                        mu_d_samples.sel(mean_d_prime_dim_0=[2, 3]).mean(dim='mean_d_prime_dim_0'),

        'Simple_vs_Complex': mu_d_samples.sel(mean_d_prime_dim_0=[0, 2]).mean(dim='mean_d_prime_dim_0') -
                            mu_d_samples.sel(mean_d_prime_dim_0=[1, 3]).mean(dim='mean_d_prime_dim_0'),

        'Easy_Simple_vs_Easy_Complex': mu_d_samples.sel(mean_d_prime_dim_0=0) -
                                   mu_d_samples.sel(mean_d_prime_dim_0=1),

        'Easy_Simple_vs_Hard_Simple': mu_d_samples.sel(mean_d_prime_dim_0=0) -
                                  mu_d_samples.sel(mean_d_prime_dim_0=2),

        'Hard_Simple_vs_Hard_Complex': mu_d_samples.sel(mean_d_prime_dim_0=2) -
                                   mu_d_samples.sel(mean_d_prime_dim_0=3),

        'Easy_Complex_vs_Hard_Complex': mu_d_samples.sel(mean_d_prime_dim_0=1) -
                                    mu_d_samples.sel(mean_d_prime_dim_0=3)
    }
    d_idata = az.from_dict(posterior={name: val for name, val in d_contrasts.items()}) #gpt suggested

    for name in d_contrasts:
        print(f"\nSummary of d' difference: {name.replace('_', ' ')}")
        print(az.summary(d_idata, var_names=[name], hdi_prob=0.94))

        az.plot_posterior(d_idata, var_names=[name], hdi_prob=0.94, ref_val=0)
        plt.tight_layout()
        plt.suptitle(f"Posterior of d' Contrast: {name.replace('_', ' ')}", y=1.02)
        plt.savefig(OUTPUT_DIR / f"dprime_contrast_{name}.png")
        plt.close()

    #for criterion
    mu_c_samples = idata.posterior['mean_criterion']

    c_contrasts = {
        'Easy_vs_Hard': mu_c_samples.sel(mean_criterion_dim_0=[0, 1]).mean(dim='mean_criterion_dim_0') -
                        mu_c_samples.sel(mean_criterion_dim_0=[2, 3]).mean(dim='mean_criterion_dim_0'),

        'Simple_vs_Complex': mu_c_samples.sel(mean_criterion_dim_0=[0, 2]).mean(dim='mean_criterion_dim_0') -
                         mu_c_samples.sel(mean_criterion_dim_0=[1, 3]).mean(dim='mean_criterion_dim_0'),

        'Easy_Simple_vs_Easy_Complex': mu_c_samples.sel(mean_criterion_dim_0=0) -
                                   mu_c_samples.sel(mean_criterion_dim_0=1),

        'Easy_Simple_vs_Hard_Simple': mu_c_samples.sel(mean_criterion_dim_0=0) -
                                  mu_c_samples.sel(mean_criterion_dim_0=2),

        'Hard_Simple_vs_Hard_Complex': mu_c_samples.sel(mean_criterion_dim_0=2) -
                                   mu_c_samples.sel(mean_criterion_dim_0=3),

        'Easy_Complex_vs_Hard_Complex': mu_c_samples.sel(mean_criterion_dim_0=1) -
                                    mu_c_samples.sel(mean_criterion_dim_0=3)
    }

    c_idata = az.from_dict(posterior={name: val for name, val in c_contrasts.items()})

    for name in c_contrasts:
        print(f"\nSummary of criterion difference: {name.replace('_', ' ')}")
        print(az.summary(c_idata, var_names=[name], hdi_prob=0.94))

        az.plot_posterior(c_idata, var_names=[name], hdi_prob=0.94, ref_val=0)
        plt.tight_layout()
        plt.suptitle(f"Posterior of Criterion Contrast: {name.replace('_', ' ')}", y=1.02)
        plt.savefig(OUTPUT_DIR / f"criterion_contrast_{name}.png")
        plt.close()

    #delta plots
    for pnum in data['pnum'].unique():
        draw_delta_plots(data, pnum)

# Main execution
def run_analysis():
    data = read_data(DATA_DIR / 'data.csv', prepare_for='sdt')
    show_summary_sdt(data)
    model = apply_hierarchical_sdt_model(data)
    idata = sample_posterior(model, draws=2000, tune=1000, chains=4, target_accept=0.9)
    delta_data = read_data(DATA_DIR / 'data.csv', prepare_for='delta plots')
    analyze_results(idata, delta_data)


if __name__ == "__main__":
    file_to_print = Path(__file__).parent.parent / 'README.md' #add the other parent since i have this folder structure
    with open(file_to_print, 'r') as file:
        print(file.read())
    run_analysis()
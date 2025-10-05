import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
import os
from pathlib import Path

# Configuration
pH_conditions = [
    {'PH': '40', 'PHdot': '4.0'},
    {'PH': '74', 'PHdot': '7.4'}, 
    {'PH': '85', 'PHdot': '8.5'}
]

def load_frame_data(fold, ph, ion_type):
    """Load frame analysis data for specific fold, pH, and ion type"""
    filename = f"FOLD{fold}_pH{ph}_Frame_Analysis_{ion_type}.txt"
    
    if not os.path.exists(filename):
        print(f" Warning: {filename} not found")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filename, sep='\t')
        df['fold'] = fold
        df['pH'] = float(ph.replace('40', '4.0').replace('74', '7.4').replace('85', '8.5'))
        df['ion_type'] = ion_type
        return df
    except Exception as e:
        print(f" Error loading {filename}: {e}")
        return pd.DataFrame()

def combine_all_data():
    """Combine all FOLD1 and FOLD2 data for both ion types"""
    all_data = []
    
    for condition in pH_conditions:
        PH = condition['PH']
        PHdot = condition['PHdot']
        
        print(f"Loading data for pH {PHdot}...")
        
        # Load data for both folds and both ion types
        for fold in ['1', '2']:
            for ion_type in ['Cl-', 'Na+']:
                df = load_frame_data(fold, PH, ion_type)
                if not df.empty:
                    all_data.append(df)
                    print(f" FOLD{fold} {ion_type}: {len(df)} frames")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n Combined dataset: {len(combined_df)} total data points")
        return combined_df
    else:
        print(" No data loaded!")
        return pd.DataFrame()

def perform_statistical_analysis(df, ion_type):
    """Perform statistical analysis between pH conditions"""
    print(f"\n Statistical Analysis for {ion_type}:")
    print("="*50)
    
    # Filter for specific ion type
    ion_data = df[df['ion_type'] == ion_type].copy()
    
    if ion_data.empty:
        print(f"No data for {ion_type}")
        return {}
    
    # Get pH groups
    ph_values = sorted(ion_data['pH'].unique())
    
    # Kruskal-Wallis test (overall difference)
    groups = [ion_data[ion_data['pH'] == ph]['ion_counts'].values for ph in ph_values]
    kruskal_stat, kruskal_p = kruskal(*groups)
    print(f"Kruskal-Wallis test: H = {kruskal_stat:.3f}, p = {kruskal_p:.4f}")
    
    # Pairwise Mann-Whitney U tests
    pairwise_results = {}
    for i, ph1 in enumerate(ph_values):
        for j, ph2 in enumerate(ph_values):
            if i < j:
                group1 = ion_data[ion_data['pH'] == ph1]['ion_counts']
                group2 = ion_data[ion_data['pH'] == ph2]['ion_counts']
                
                if len(group1) > 0 and len(group2) > 0:
                    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                    pairwise_results[(ph1, ph2)] = p_value
                    
                    # Calculate effect size (rank-biserial correlation)
                    n1, n2 = len(group1), len(group2)
                    r = 1 - (2 * u_stat) / (n1 * n2)
                    effect_size = abs(r)
                    
                    # Determine effect size magnitude
                    if effect_size < 0.1:
                        effect_mag = "negligible"
                    elif effect_size < 0.3:
                        effect_mag = "small"
                    elif effect_size < 0.5:
                        effect_mag = "medium"
                    else:
                        effect_mag = "large"
                    
                    print(f"pH {ph1} vs pH {ph2}: p = {p_value:.4f}, r = {effect_size:.3f} ({effect_mag})")
    
    return pairwise_results

def add_stat_annotation(ax, x1, x2, y, p_value, height_offset=0.1):
    """Add statistical annotation between two groups"""
    if p_value < 0.001:
        sig_symbol = '***'
    elif p_value < 0.01:
        sig_symbol = '**'
    elif p_value < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
    
    # Draw the bar
    ax.plot([x1, x1, x2, x2], [y, y + height_offset, y + height_offset, y], 
            lw=1, c='black')
    
    # Add the significance symbol
    ax.text((x1 + x2) * 0.5, y + height_offset, sig_symbol, 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

def create_violin_plot(df, ion_type, pairwise_results):
    """Create violin plot with individual frame dots"""
    # Filter for specific ion type
    plot_data = df[df['ion_type'] == ion_type].copy()
    
    if plot_data.empty:
        print(f"No data to plot for {ion_type}")
        return
    
    # Set up the plot
    plt.figure(figsize=(8, 10))
    sns.set_style("white")
    sns.set_style("ticks")
    
    # Define pH-scale inspired colors
    ph_colors = {
        4.0: '#bfff00',    # Lime green for acidic pH 4
        7.4: '#228B22',    # Forest green for neutral pH 7.4
        8.5: '#008B8B'     # Dark cyan for alkaline pH 8.5
    }
    
    # Create color palette
    ph_values = sorted(plot_data['pH'].unique())
    color_palette = [ph_colors[ph] for ph in ph_values]
    
    # Create violin plot
    ax = sns.violinplot(data=plot_data, x='pH', y='ion_counts', 
                       palette=color_palette, alpha=0.7, inner=None)
    
    # Add individual points with jitter - different colors for FOLD1 and FOLD2
    fold_colors = {'1': '#FF6347', '2': '#4169E1'}  # Tomato and Royal Blue
    
    for fold in ['1', '2']:
        fold_data = plot_data[plot_data['fold'] == fold]
        if not fold_data.empty:
            sns.stripplot(data=fold_data, x='pH', y='ion_counts', 
                         size=3, alpha=0.6, jitter=0.3, 
                         color=fold_colors[fold], ax=ax, 
                         label=f'FOLD{fold}')
    
    # Customize the plot
    ion_symbol = ion_type.replace('-', '⁻').replace('+', '⁺')
    ax.set_ylabel(f'{ion_symbol} Interactions per Frame', fontsize=16)
    ax.set_xlabel('')
    ax.set_title(f'Protein-{ion_symbol} Interactions per Frame by pH\n(Combined FOLD1 and FOLD2)', 
                fontsize=14, pad=20)
    
    # Set y-axis limits
    y_max = plot_data['ion_counts'].max()
    ax.set_ylim(-0.5, y_max * 1.3)
    
    # Add statistical annotations if we have pairwise results
    if pairwise_results:
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        base_height = y_max + y_range * 0.05
        
        # Get positions for pH values
        ph_positions = {ph: i for i, ph in enumerate(ph_values)}
        
        annotations_added = 0
        for (ph1, ph2), p_val in pairwise_results.items():
            if ph1 in ph_positions and ph2 in ph_positions:
                x1, x2 = ph_positions[ph1], ph_positions[ph2]
                y_pos = base_height + (annotations_added * y_range * 0.06)
                add_stat_annotation(ax, x1, x2, y_pos, p_val, height_offset=y_range*0.02)
                annotations_added += 1
    
    # Customize tick labels
    ax.set_xticklabels([f'pH {ph}' for ph in ph_values], fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=14)
    
    # Add legend for FOLD colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=fold_colors['1'], markersize=8, 
                                 alpha=0.6, label='FOLD1'),
                      plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=fold_colors['2'], markersize=8, 
                                 alpha=0.6, label='FOLD2')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    filename = f'Combined_FOLD1_FOLD2_{ion_type.replace("-", "Cl").replace("+", "Na")}_violin_plot.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f" Plot saved: {filename}")

def generate_summary_statistics(df):
    """Generate comprehensive summary statistics"""
    print("\n" + "="*60)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("="*60)
    
    for ion_type in ['Cl-', 'Na+']:
        print(f"\n{ion_type} Interactions:")
        print("-" * 40)
        
        ion_data = df[df['ion_type'] == ion_type]
        if ion_data.empty:
            print(f"No data for {ion_type}")
            continue
        
        # Overall statistics
        print("Overall Statistics:")
        print(f"  Total frames analyzed: {len(ion_data)}")
        print(f"  Mean interactions per frame: {ion_data['ion_counts'].mean():.2f}")
        print(f"  Median interactions per frame: {ion_data['ion_counts'].median():.2f}")
        print(f"  Standard deviation: {ion_data['ion_counts'].std():.2f}")
        print(f"  Min-Max range: {ion_data['ion_counts'].min()}-{ion_data['ion_counts'].max()}")
        
        # By pH
        print("\nBy pH condition:")
        ph_stats = ion_data.groupby('pH')['ion_counts'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)
        print(ph_stats)
        
        # By FOLD
        print("\nBy FOLD:")
        fold_stats = ion_data.groupby('fold')['ion_counts'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)
        print(fold_stats)
        
        # By pH and FOLD
        print("\nBy pH and FOLD:")
        combined_stats = ion_data.groupby(['pH', 'fold'])['ion_counts'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(3)
        print(combined_stats)

def main():
    """Main analysis pipeline"""
    print(" Combined FOLD1 and FOLD2 Ion Interaction Analysis")
    print("="*60)
    print(" Loading and combining frame data...")
    
    # Load and combine all data
    combined_df = combine_all_data()
    
    if combined_df.empty:
        print(" No data to analyze!")
        return
    
    print(f" Successfully loaded data from {combined_df['fold'].nunique()} folds")
    print(f" pH conditions: {sorted(combined_df['pH'].unique())}")
    print(f" Ion types: {sorted(combined_df['ion_type'].unique())}")
    
    # Generate summary statistics
    generate_summary_statistics(combined_df)
    
    # Perform analysis for each ion type
    for ion_type in ['Cl-', 'Na+']:
        print(f"\n{'='*60}")
        print(f"ANALYZING {ion_type} INTERACTIONS")
        print(f"{'='*60}")
        
        # Perform statistical tests
        pairwise_results = perform_statistical_analysis(combined_df, ion_type)
        
        # Create violin plot
        create_violin_plot(combined_df, ion_type, pairwise_results)
    
    # Create combined ratio analysis if both ion types are present
    if set(['Cl-', 'Na+']) <= set(combined_df['ion_type'].unique()):
        print(f"\n{'='*60}")
        print("CREATING Cl-/Na+ RATIO ANALYSIS")
        print(f"{'='*60}")
        
        # Calculate ratios per frame
        ratio_data = []
        for _, row in combined_df.iterrows():
            frame_id = f"{row['fold']}_{row['pH']}_{row['frame']}"
            ratio_data.append({
                'frame_id': frame_id,
                'fold': row['fold'],
                'pH': row['pH'],
                'frame': row['frame'],
                'ion_type': row['ion_type'],
                'ion_counts': row['ion_counts']
            })
        
        ratio_df = pd.DataFrame(ratio_data)
        
        # Pivot to get Cl- and Na+ in separate columns
        pivot_df = ratio_df.pivot_table(
            index=['fold', 'pH', 'frame'], 
            columns='ion_type', 
            values='ion_counts', 
            fill_value=0
        ).reset_index()
        
        # Calculate ratio (avoid division by zero)
        pivot_df['Cl-_Na+_ratio'] = np.where(
            pivot_df['Na+'] > 0,
            pivot_df['Cl-'] / pivot_df['Na+'],
            np.where(pivot_df['Cl-'] > 0, float('inf'), 0)
        )
        
        # Remove infinite values for statistical analysis
        finite_ratios = pivot_df[np.isfinite(pivot_df['Cl-_Na+_ratio'])].copy()
        
    print(f"\n Analysis complete! All plots and statistics generated.")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Ion Interactions Per Frame Analysis
Analyzes ion interactions for total frames and last 100 frames
Based on the existing monomer protein interaction analysis
"""

import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# === CONFIGURATION === 
cutoff = 5.0  # distance cutoff in Å
FOLD = '1'    # Single chain configuration

pH_conditions = [
    {'PH': '40', 'PHdot': '4.0'},
    #{'PH': '74', 'PHdot': '7.4'}, 
    #{'PH': '85', 'PHdot': '8.5'}
]

def load_ion_interactions(filename):
    """Load ion interaction data from file"""
    if not os.path.exists(filename):
        print(f" ERROR: File '{filename}' not found.")
        return pd.DataFrame()
    
    print(f" Reading: {filename}")
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    data = []
    for line in lines:
        # Expected format: "frame X RESNAME RESID chain SYSTEM interacts_with_ION_IONID"
        pattern = r"frame\s+(\d+)\s+(\w+)\s+(\d+)\s+.*?interacts_with_(Cl-|Na\+)_(\d+)"
        match = re.search(pattern, line)
        
        if match:
            frame, resname, resid, ion_type, ion_resid = match.groups()
            residue_id = f"{resname} {resid}"
            ion_id = f"{ion_type}_{ion_resid}"
            
            data.append({
                "frame": int(frame),
                "resname": resname,
                "resid": int(resid),
                "ion_type": ion_type,
                "ion_resid": int(ion_resid),
                "ion_id": ion_id,
                "residue_id": residue_id
            })
    
    print(f"  Loaded {len(data)} interactions")
    return pd.DataFrame(data)

def analyze_interactions_per_frame(df, ion_name):
    """
    Analyze ion interactions per frame
    Returns dictionary with frame-wise interaction counts
    """
    if df.empty:
        return {}
    
    frame_interactions = defaultdict(int)
    
    # Group by frame and count unique residue-ion interactions per frame
    for frame, frame_group in df.groupby("frame"):
        # Count unique residue_id values for this frame
        unique_residues = frame_group['residue_id'].nunique()
        frame_interactions[frame] = unique_residues
    
    return dict(frame_interactions)

def get_frame_statistics(frame_interactions, total_frames=None):
    """Calculate statistics for frame interactions"""
    if not frame_interactions:
        return {
            'total_frames_analyzed': 0,
            'frames_with_interactions': 0,
            'total_interactions': 0,
            'avg_interactions_per_frame': 0.0,
            'avg_interactions_per_active_frame': 0.0,
            'max_interactions_per_frame': 0,
            'min_interactions_per_frame': 0
        }
    
    frames_analyzed = list(frame_interactions.keys())
    interaction_counts = list(frame_interactions.values())
    
    # If total_frames is provided, include frames with 0 interactions
    if total_frames:
        all_frames = list(range(1, total_frames + 1))
        full_interactions = [frame_interactions.get(frame, 0) for frame in all_frames]
        interaction_counts = full_interactions
        frames_analyzed = all_frames
    
    stats = {
        'total_frames_analyzed': len(frames_analyzed),
        'frames_with_interactions': sum(1 for count in interaction_counts if count > 0),
        'total_interactions': sum(interaction_counts),
        'avg_interactions_per_frame': np.mean(interaction_counts) if interaction_counts else 0.0,
        'avg_interactions_per_active_frame': np.mean([c for c in interaction_counts if c > 0]) if any(c > 0 for c in interaction_counts) else 0.0,
        'max_interactions_per_frame': max(interaction_counts) if interaction_counts else 0,
        'min_interactions_per_frame': min(interaction_counts) if interaction_counts else 0
    }
    
    return stats

def analyze_last_n_frames(frame_interactions, n_frames=100):
    """Analyze interactions for the last N frames"""
    if not frame_interactions:
        return {}, {}
    
    all_frames = sorted(frame_interactions.keys())
    if len(all_frames) < n_frames:
        print(f"⚠ Warning: Only {len(all_frames)} frames available, using all frames instead of last {n_frames}")
        last_frames = all_frames
    else:
        last_frames = all_frames[-n_frames:]
    
    # Create subset for last N frames
    last_n_interactions = {frame: frame_interactions[frame] for frame in last_frames}
    
    return last_n_interactions, last_frames

def create_frame_analysis_plot(frame_interactions_cl, frame_interactions_na, pH, output_dir="plots"):
    """Create visualization of interactions per frame"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all frames
    all_frames_cl = set(frame_interactions_cl.keys()) if frame_interactions_cl else set()
    all_frames_na = set(frame_interactions_na.keys()) if frame_interactions_na else set()
    all_frames = sorted(all_frames_cl | all_frames_na)
    
    if not all_frames:
        print("No frames to plot")
        return
    
    # Prepare data for plotting
    cl_counts = [frame_interactions_cl.get(frame, 0) for frame in all_frames]
    na_counts = [frame_interactions_na.get(frame, 0) for frame in all_frames]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Cl- interactions
    ax1.plot(all_frames, cl_counts, 'b-', linewidth=1, alpha=0.7, label='Cl- interactions')
    ax1.set_ylabel('Cl- Interactions per Frame')
    ax1.set_title(f'Ion Interactions per Frame - pH {pH}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Na+ interactions
    ax2.plot(all_frames, na_counts, 'r-', linewidth=1, alpha=0.7, label='Na+ interactions')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Na+ Interactions per Frame')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(output_dir, f"FOLD{FOLD}_pH{pH.replace('.', '')}_interactions_per_frame.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Plot saved: {plot_filename}")

def get_total_frames_from_data(frame_interactions_cl, frame_interactions_na):
    """Determine total number of frames from the data - fixed to 750 frames"""
    # Always use 750 frames as the total
    return 750

def generate_simple_data_files(frame_interactions_cl, frame_interactions_na, PH):
    """Generate simple data files with frame and ion_counts format (including zero counts)"""
    
    files_written = []
    
    # Determine the total number of frames
    total_frames = get_total_frames_from_data(frame_interactions_cl, frame_interactions_na)
    
    if total_frames == 0:
        print(" No frame data found")
        return files_written
    
    print(f"  Total frames detected: {total_frames}")
    
    # Cl- data file (including zeros)
    cl_filename = f"FOLD{FOLD}_pH{PH}_Frame_Analysis_Cl-.txt"
    with open(cl_filename, 'w') as f:
        f.write("frame\tion_counts\n")
        for frame in range(1, total_frames + 1):
            count = frame_interactions_cl.get(frame, 0)
            f.write(f"{frame}\t{count}\n")
    files_written.append(cl_filename)
    print(f" Cl- data saved: {cl_filename}")
    
    # Na+ data file (including zeros)
    na_filename = f"FOLD{FOLD}_pH{PH}_Frame_Analysis_Na+.txt"
    with open(na_filename, 'w') as f:
        f.write("frame\tion_counts\n")
        for frame in range(1, total_frames + 1):
            count = frame_interactions_na.get(frame, 0)
            f.write(f"{frame}\t{count}\n")
    files_written.append(na_filename)
    print(f" Na+ data saved: {na_filename}")
    
    return files_written

def generate_frame_analysis_report(frame_stats_cl, frame_stats_na, last_100_stats_cl, last_100_stats_na, pH, PH):
    """Generate comprehensive frame analysis report"""
    
    filename = f"FOLD{FOLD}_pH{PH}_Frame_Analysis_Report.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Ion Interactions Per Frame Analysis - pH {pH}\n")
        f.write("="*60 + "\n")
        f.write("Analysis of unique residue-ion interactions per frame\n")
        f.write("Each residue counted only once per frame per ion type\n")
        f.write("-"*60 + "\n\n")
        
        # Total frames analysis
        f.write("TOTAL FRAMES ANALYSIS:\n")
        f.write("-"*30 + "\n")
        f.write("Cl- Interactions:\n")
        f.write(f"  Total frames analyzed: {frame_stats_cl['total_frames_analyzed']}\n")
        f.write(f"  Frames with interactions: {frame_stats_cl['frames_with_interactions']}\n")
        f.write(f"  Total interactions: {frame_stats_cl['total_interactions']}\n")
        f.write(f"  Average per frame: {frame_stats_cl['avg_interactions_per_frame']:.2f}\n")
        f.write(f"  Average per active frame: {frame_stats_cl['avg_interactions_per_active_frame']:.2f}\n")
        f.write(f"  Max interactions per frame: {frame_stats_cl['max_interactions_per_frame']}\n")
        f.write(f"  Min interactions per frame: {frame_stats_cl['min_interactions_per_frame']}\n\n")
        
        f.write("Na+ Interactions:\n")
        f.write(f"  Total frames analyzed: {frame_stats_na['total_frames_analyzed']}\n")
        f.write(f"  Frames with interactions: {frame_stats_na['frames_with_interactions']}\n")
        f.write(f"  Total interactions: {frame_stats_na['total_interactions']}\n")
        f.write(f"  Average per frame: {frame_stats_na['avg_interactions_per_frame']:.2f}\n")
        f.write(f"  Average per active frame: {frame_stats_na['avg_interactions_per_active_frame']:.2f}\n")
        f.write(f"  Max interactions per frame: {frame_stats_na['max_interactions_per_frame']}\n")
        f.write(f"  Min interactions per frame: {frame_stats_na['min_interactions_per_frame']}\n\n")
        
        # Last 100 frames analysis
        f.write("LAST 100 FRAMES ANALYSIS:\n")
        f.write("-"*30 + "\n")
        f.write("Cl- Interactions (Last 100 frames):\n")
        f.write(f"  Frames analyzed: {last_100_stats_cl['total_frames_analyzed']}\n")
        f.write(f"  Frames with interactions: {last_100_stats_cl['frames_with_interactions']}\n")
        f.write(f"  Total interactions: {last_100_stats_cl['total_interactions']}\n")
        f.write(f"  Average per frame: {last_100_stats_cl['avg_interactions_per_frame']:.2f}\n")
        f.write(f"  Average per active frame: {last_100_stats_cl['avg_interactions_per_active_frame']:.2f}\n")
        f.write(f"  Max interactions per frame: {last_100_stats_cl['max_interactions_per_frame']}\n")
        f.write(f"  Min interactions per frame: {last_100_stats_cl['min_interactions_per_frame']}\n\n")
        
        f.write("Na+ Interactions (Last 100 frames):\n")
        f.write(f"  Frames analyzed: {last_100_stats_na['total_frames_analyzed']}\n")
        f.write(f"  Frames with interactions: {last_100_stats_na['frames_with_interactions']}\n")
        f.write(f"  Total interactions: {last_100_stats_na['total_interactions']}\n")
        f.write(f"  Average per frame: {last_100_stats_na['avg_interactions_per_frame']:.2f}\n")
        f.write(f"  Average per active frame: {last_100_stats_na['avg_interactions_per_active_frame']:.2f}\n")
        f.write(f"  Max interactions per frame: {last_100_stats_na['max_interactions_per_frame']}\n")
        f.write(f"  Min interactions per frame: {last_100_stats_na['min_interactions_per_frame']}\n\n")
        
        # Comparison
        f.write("COMPARISON (Total vs Last 100 frames):\n")
        f.write("-"*40 + "\n")
        f.write("Cl- Interactions:\n")
        if frame_stats_cl['avg_interactions_per_frame'] > 0:
            change_cl = ((last_100_stats_cl['avg_interactions_per_frame'] - frame_stats_cl['avg_interactions_per_frame']) / 
                        frame_stats_cl['avg_interactions_per_frame'] * 100)
            f.write(f"  Change in average: {change_cl:+.1f}%\n")
        
        f.write("Na+ Interactions:\n")
        if frame_stats_na['avg_interactions_per_frame'] > 0:
            change_na = ((last_100_stats_na['avg_interactions_per_frame'] - frame_stats_na['avg_interactions_per_frame']) / 
                        frame_stats_na['avg_interactions_per_frame'] * 100)
            f.write(f"  Change in average: {change_na:+.1f}%\n")
    
    print(f" Report saved: {filename}")
    return filename

def main():
    """Main analysis pipeline for frame-wise ion interactions"""
    print(" Starting Ion Interactions Per Frame Analysis")
    print(f" Configuration: cutoff={cutoff} Å, fold={FOLD}")
    print(" Analyzing total frames and last 100 frames\n")
    
    for condition in pH_conditions:
        PH = condition['PH']
        PHdot = condition['PHdot']
        
        print(f" Processing pH {PHdot}...")
        
        # File paths
        cla_file = f"FOLD{FOLD}_pH{PH}_ion_residues_with_CLA_interactions_{cutoff}A.txt"
        sod_file = f"FOLD{FOLD}_pH{PH}_ion_residues_with_SOD_interactions_{cutoff}A.txt"
        
        # Load ion interactions
        cla_df = load_ion_interactions(cla_file)
        sod_df = load_ion_interactions(sod_file)
        
        # Analyze interactions per frame
        frame_interactions_cl = analyze_interactions_per_frame(cla_df, "Cl-")
        frame_interactions_na = analyze_interactions_per_frame(sod_df, "Na+")
        
        print(f"  Cl- interactions found in {len(frame_interactions_cl)} frames")
        print(f"  Na+ interactions found in {len(frame_interactions_na)} frames")
        
        # Get total frame statistics
        frame_stats_cl = get_frame_statistics(frame_interactions_cl)
        frame_stats_na = get_frame_statistics(frame_interactions_na)
        
        # Analyze last 100 frames
        last_100_cl, last_frames_cl = analyze_last_n_frames(frame_interactions_cl, 100)
        last_100_na, last_frames_na = analyze_last_n_frames(frame_interactions_na, 100)
        
        # Get statistics for last 100 frames
        last_100_stats_cl = get_frame_statistics(last_100_cl)
        last_100_stats_na = get_frame_statistics(last_100_na)
        
        # Generate simple data files (your requested format)
        data_files = generate_simple_data_files(frame_interactions_cl, frame_interactions_na, PH)
        
        # Generate comprehensive report
        report_file = generate_frame_analysis_report(
            frame_stats_cl, frame_stats_na, 
            last_100_stats_cl, last_100_stats_na,
            PHdot, PH
        )
        
        # Create visualization
        try:
            create_frame_analysis_plot(frame_interactions_cl, frame_interactions_na, PHdot)
        except Exception as e:
            print(f"Could not create plot: {e}")
        
        # Print summary to console
        print(f" Summary for pH {PHdot}:")
        print(f"    Total Cl- interactions: {frame_stats_cl['total_interactions']} (avg: {frame_stats_cl['avg_interactions_per_frame']:.2f}/frame)")
        print(f"    Total Na+ interactions: {frame_stats_na['total_interactions']} (avg: {frame_stats_na['avg_interactions_per_frame']:.2f}/frame)")
        print(f"    Last 100 frames Cl-: {last_100_stats_cl['total_interactions']} (avg: {last_100_stats_cl['avg_interactions_per_frame']:.2f}/frame)")
        print(f"    Last 100 frames Na+: {last_100_stats_na['total_interactions']} (avg: {last_100_stats_na['avg_interactions_per_frame']:.2f}/frame)")
        print(f"    Data files: {', '.join(data_files)}")
        print(f"    Report: {report_file}")
        print()
    
    print(" Frame-wise ion interaction analysis complete!")

if __name__ == "__main__":
    main()

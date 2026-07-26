import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up the root path
ROOT = Path(__file__).resolve().parents[1]

def load_telemetry(filename):
    path = ROOT / "outputs" / filename
    if not path.exists():
        print(f"Error: Could not find {path}. Did you run the baseline scripts first?")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)

def generate_comparison_plots():
    print("Loading baseline telemetry data...")
    lvl2 = load_telemetry("level02_telemetry.json")
    lvl3 = load_telemetry("level03_telemetry.json")

    # Set up a beautiful, wide figure with 3 subplots
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Baseline Evaluation: Level 02 (Drop) vs Level 03 (Pro Throw)', fontsize=16, fontweight='bold')

    # ---------------------------------------------------------
    # PLOT 1: BALL TRAJECTORY (X vs Z)
    # ---------------------------------------------------------
    axs[0].plot(lvl2['ball_x'], lvl2['ball_z'], label='Level 02: Chest Pass', color='orange', linewidth=2.5)
    axs[0].plot(lvl3['ball_x'], lvl3['ball_z'], label='Level 03: High-Arc Throw', color='blue', linewidth=2.5)
    
    # Plot Targets
    axs[0].scatter([0.55], [0.04], color='darkorange', marker='X', s=100, label='Target Lvl 02 [0.55, 0.0]')
    axs[0].scatter([1.80], [1.20], color='darkblue', marker='*', s=150, label='Hoop Target Lvl 03 [1.8, 1.2]')
    
    axs[0].set_title('Ball Flight Trajectory', fontsize=14)
    axs[0].set_xlabel('Horizontal Distance X (m)')
    axs[0].set_ylabel('Height Z (m)')
    axs[0].set_xlim(-0.2, 2.2)
    axs[0].set_ylim(0, 2.0)
    axs[0].legend()

    # ---------------------------------------------------------
    # PLOT 2: TORSO STABILITY (Pitch over Time)
    # ---------------------------------------------------------
    axs[1].plot(lvl2['time'], lvl2['pitch'], label='Level 02 (No Gyro)', color='orange', linewidth=2)
    axs[1].plot(lvl3['time'], lvl3['pitch'], label='Level 03 (PD Gyroscope)', color='blue', linewidth=2)
    
    axs[1].axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Task 1 Limit (+2°)')
    axs[1].axhline(y=-2.0, color='r', linestyle='--', alpha=0.5, label='Task 1 Limit (-2°)')
    
    axs[1].set_title('Torso Stability (Pitch Angle)', fontsize=14)
    axs[1].set_xlabel('Simulation Time (s)')
    axs[1].set_ylabel('Pitch (Degrees)')
    axs[1].legend()

    # ---------------------------------------------------------
    # PLOT 3: KEY METRICS BAR CHART
    # ---------------------------------------------------------
    metrics = ['Final Dist Error (m)', 'Max Impact Force (N)']
    l2_metrics = [lvl2['metrics']['final_distance'], lvl2['metrics']['max_impact_force']]
    l3_metrics = [lvl3['metrics']['final_distance'], lvl3['metrics']['max_impact_force']]
    
    x = np.arange(len(metrics))
    width = 0.35

    axs[2].bar(x - width/2, l2_metrics, width, label='Level 02', color='orange')
    axs[2].bar(x + width/2, l3_metrics, width, label='Level 03', color='blue')

    axs[2].set_title('Performance Metrics Comparison', fontsize=14)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(metrics)
    axs[2].legend()
    
    # Add data labels on top of bars
    for i in range(len(metrics)):
        axs[2].text(i - width/2, l2_metrics[i] + 0.1, f'{l2_metrics[i]:.2f}', ha='center')
        axs[2].text(i + width/2, l3_metrics[i] + 0.1, f'{l3_metrics[i]:.2f}', ha='center')

    plt.tight_layout()
    
    # Save the figure
    save_path = ROOT / "outputs" / "baseline_comparison_plots.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nPlots generated and saved successfully to:\n{save_path}")
    
    # Show the interactive plot
    plt.show()

if __name__ == "__main__":
    generate_comparison_plots()
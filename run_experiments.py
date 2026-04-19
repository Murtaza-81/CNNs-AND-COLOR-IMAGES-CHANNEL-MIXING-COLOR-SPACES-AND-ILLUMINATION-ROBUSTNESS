"""Main script to run all experiments"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import Config, set_seed, save_metrics
from src.train import train_model
from src.eval import analyze_channel_mixing, plot_confusion_matrix
from src.robustness import evaluate_robustness, plot_robustness_curves
from models.cnn_extension import CNNExtension

def main():
    print("\n" + "="*70)
    print("CS511 - Computer Vision Assignment 1: CNN Color Analysis")
    print("="*70)
    
    set_seed(42)
    Config.create_dirs()
    Config.print_config()
    
    # Task A & C: Train models with different color spaces
    print("\n📊 TASK A & C: Training models with different color spaces...")
    
    models = {}
    metrics_dict = {}
    
    for color_space in ['rgb', 'hsv', 'lab']:
        print(f"\n--- Training {color_space.upper()} model ---")
        model, test_loader, metrics = train_model(color_space=color_space)
        models[color_space] = model
        metrics_dict[color_space] = metrics
        
        # Plot confusion matrix
        plot_confusion_matrix(model, test_loader, color_space)
    
    # Task B: Channel mixing analysis on RGB model
    print("\n🔍 TASK B: Channel Mixing Analysis")
    analyze_channel_mixing(models['rgb'], "rgb")
    
    # Task C: Color space comparison plot
    print("\n📊 Creating color space comparison plot...")
    color_space_results = {
        "RGB": metrics_dict['rgb']['test_accuracy'],
        "HSV": metrics_dict['hsv']['test_accuracy'],
        "LAB": metrics_dict['lab']['test_accuracy']
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(color_space_results.keys(), color_space_results.values(), 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', linewidth=2)
    
    for bar, value in zip(bars, color_space_results.values()):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Color Space', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('CNN Performance Across Different Color Spaces', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, "color_space_comparison.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    save_metrics(color_space_results, "color_space_results.json")
    
    # Task D: Robustness benchmark on RGB model
    print("\n🛡️ TASK D: Robustness Benchmark")
    robustness_results = evaluate_robustness(models['rgb'], test_loader)
    plot_robustness_curves(robustness_results)
    
    # Extension: Train with learnable color transform
    print("\n🚀 EXTENSION: Learnable Color Transform")
    print("\n--- Training baseline RGB model (for comparison) ---")
    baseline_model, _, baseline_metrics = train_model(color_space="rgb", use_extension=False)
    
    print("\n--- Training model with extension ---")
    extension_model, _, extension_metrics = train_model(color_space="rgb", use_extension=True)
    
    # Visualize learned transform
    if hasattr(extension_model, 'color_transform'):
        transform_weights = extension_model.color_transform.weight.data.cpu().numpy()
        transform_weights = transform_weights.reshape(transform_weights.shape[0], -1)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(transform_weights, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        ax.set_xlabel('Input Channels')
        ax.set_ylabel('Output Channels')
        ax.set_title('Learned Color Transform Matrix')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['R', 'G', 'B'])
        ax.set_yticks(range(3))
        ax.set_yticklabels(['Out R', 'Out G', 'Out B'])
        plt.colorbar(im, label='Weight Value')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, "learned_color_transform.png"), dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nLearned color transform matrix:")
        print(transform_weights)
    
    # Extension comparison plot
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ['Baseline RGB', 'Extension']
    accuracies = [baseline_metrics['test_accuracy'], extension_metrics['test_accuracy']]
    bars = ax.bar(categories, accuracies, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=2)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Extension vs Baseline Performance', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, "extension_vs_baseline.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    extension_results = {
        "baseline_accuracy": baseline_metrics['test_accuracy'],
        "extension_accuracy": extension_metrics['test_accuracy'],
        "improvement": extension_metrics['test_accuracy'] - baseline_metrics['test_accuracy'],
        "learned_transform": transform_weights.tolist() if 'transform_weights' in locals() else None
    }
    
    save_metrics(extension_results, "extension_results.json")
    
    # Final summary
    summary = {
        "task_a_baseline": metrics_dict['rgb']['test_accuracy'],
        "task_c_color_spaces": color_space_results,
        "task_d_robustness": robustness_results,
        "extension": extension_results,
        "seed": 42,
        "hardware": str(Config.DEVICE),
        "total_training_time": sum([m['training_time_seconds'] for m in metrics_dict.values()])
    }
    
    save_metrics(summary, "all_experiments_summary.json")
    
    print("\n" + "="*70)
    print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nOutputs saved in:")
    print(f"  - Plots: {Config.PLOTS_DIR}")
    print(f"  - Logs: {Config.LOGS_DIR}")
    print(f"  - Checkpoints: {Config.CHECKPOINTS_DIR}")

if __name__ == "__main__":
    main()
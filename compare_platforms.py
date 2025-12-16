#!/usr/bin/env python3
"""
Platform Comparison Tool
Compares performance metrics for a specific model across different hardware platforms
(Pi4, Pi5, Computer)
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class PlatformComparator:
    def __init__(self):
        self.platforms = {
            'Pi4': './results_pi4',
            'Pi5': './results_pi5',
            'Computer': './results_computer'
        }
        self.platform_data = {}

    def load_platform_data(self, model_name):
        """Load data for a specific model from all available platforms"""
        print(f"\n{'='*60}")
        print(f"LOADING DATA FOR MODEL: {model_name}")
        print(f"{'='*60}\n")

        for platform, results_dir in self.platforms.items():
            results_path = Path(results_dir)

            if not results_path.exists():
                print(f"⚠️  {platform}: Directory not found ({results_dir})")
                continue

            # Find JSON files in this platform's results directory
            json_files = list(results_path.glob("*.json"))

            if not json_files:
                print(f"⚠️  {platform}: No result files found")
                continue

            # Search for the model in all JSON files
            model_found = False
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    results = data.get('results', [])

                    # Filter results for this specific model
                    model_results = [r for r in results if r.get('model') == model_name]

                    if model_results:
                        self.platform_data[platform] = {
                            'results': model_results,
                            'metrics': self._calculate_metrics(model_results)
                        }
                        model_found = True
                        print(f"✅ {platform}: Found {len(model_results)} test(s)")
                        break

                except Exception as e:
                    print(f"⚠️  {platform}: Error reading {json_file.name}: {e}")

            if not model_found:
                print(f"⚠️  {platform}: Model '{model_name}' not found")

        if not self.platform_data:
            print(f"\n❌ ERROR: Model '{model_name}' not found on any platform!")
            return False

        print(f"\n✅ Successfully loaded data from {len(self.platform_data)} platform(s)")
        return True

    def _calculate_metrics(self, results):
        """Calculate average metrics from results"""
        if not results:
            return {}

        metrics = {
            'tps': [],
            'ttft': [],
            'inference_time': [],
            'iops': [],
            'tpj': []
        }

        for result in results:
            if result.get('tokens_per_second'):
                metrics['tps'].append(result['tokens_per_second'])
            if result.get('time_to_first_token_s'):
                metrics['ttft'].append(result['time_to_first_token_s'])
            if result.get('inference_time_s'):
                metrics['inference_time'].append(result['inference_time_s'])
            if result.get('io_iops'):
                metrics['iops'].append(result['io_iops'])
            if result.get('tokens_per_joule'):
                metrics['tpj'].append(result['tokens_per_joule'])

        # Calculate averages
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[f'avg_{key}'] = sum(values) / len(values)
                avg_metrics[f'min_{key}'] = min(values)
                avg_metrics[f'max_{key}'] = max(values)
                avg_metrics[f'count_{key}'] = len(values)
            else:
                avg_metrics[f'avg_{key}'] = 0
                avg_metrics[f'min_{key}'] = 0
                avg_metrics[f'max_{key}'] = 0
                avg_metrics[f'count_{key}'] = 0

        return avg_metrics

    def print_comparison(self, model_name):
        """Print comparison table"""
        if not self.platform_data:
            print("No data to compare!")
            return

        print(f"\n{'='*80}")
        print(f"PERFORMANCE COMPARISON: {model_name}")
        print(f"{'='*80}\n")

        # Print header
        print(f"{'Metric':<30} | ", end="")
        for platform in self.platform_data.keys():
            print(f"{platform:>15} | ", end="")
        print()
        print("-" * 80)

        # Print metrics
        metrics_to_compare = [
            ('Tokens/Second (TPS)', 'avg_tps', True),
            ('Time to First Token (s)', 'avg_ttft', False),
            ('Inference Time (s)', 'avg_inference_time', False),
            ('IOPS', 'avg_iops', True),
            ('Tokens/Joule (TPJ)', 'avg_tpj', True)
        ]

        for metric_name, metric_key, higher_better in metrics_to_compare:
            print(f"{metric_name:<30} | ", end="")

            values = []
            for platform in self.platform_data.keys():
                value = self.platform_data[platform]['metrics'].get(metric_key, 0)
                values.append(value)
                print(f"{value:>15.3f} | ", end="")
            print()

            # Print winner
            if any(v > 0 for v in values):
                if higher_better:
                    winner_idx = values.index(max(values))
                else:
                    winner_idx = values.index(min([v for v in values if v > 0]))
                winner = list(self.platform_data.keys())[winner_idx]
                print(f"{'  → Winner: ' + winner:<30} | ", end="")
                print()

        print("=" * 80)

    def generate_comparison_graphs(self, model_name, output_dir="./graph_comparison"):
        """Generate comparison graphs"""
        if not self.platform_data:
            print("No data to visualize!")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print("GENERATING COMPARISON GRAPHS")
        print(f"{'='*60}\n")

        platforms = list(self.platform_data.keys())

        # Prepare data
        metrics = {
            'TPS\n(Tokens/Second)': ('avg_tps', True),
            'TTFT\n(seconds)': ('avg_ttft', False),
            'Inference Time\n(seconds)': ('avg_inference_time', False),
            'IOPS': ('avg_iops', True),
            'TPJ\n(Tokens/Joule)': ('avg_tpj', True)
        }

        # Create individual graphs for each metric
        for metric_name, (metric_key, higher_better) in metrics.items():
            values = []
            for platform in platforms:
                value = self.platform_data[platform]['metrics'].get(metric_key, 0)
                values.append(value)

            # Skip if all values are zero
            if all(v == 0 for v in values):
                print(f"⚠️  Skipping {metric_name.replace(chr(10), ' ')}: No data available")
                continue

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create bar chart
            x_pos = range(len(platforms))

            # Color based on performance (green = best, yellow = middle, red = worst)
            if higher_better:
                colors = ['green' if v == max(values) and v > 0 else 'orange' if v > 0 else 'gray' for v in values]
            else:
                min_nonzero = min([v for v in values if v > 0]) if any(v > 0 for v in values) else 0
                colors = ['green' if v == min_nonzero and v > 0 else 'orange' if v > 0 else 'gray' for v in values]

            bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

            # Customize
            ax.set_xlabel('Platform', fontsize=13, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
            ax.set_title(f'{metric_name} Comparison\n{model_name}',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(platforms, fontsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                           f'{value:.3f}',
                           ha='center', va='bottom', fontweight='bold', fontsize=11)

            plt.tight_layout()

            # Save
            clean_metric_name = metric_name.replace('\n', '_').replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            filename = f'{clean_metric_name}_comparison.png'
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Created: {filename}")

        # Create comprehensive comparison chart (all metrics)
        self._create_comprehensive_chart(model_name, platforms, metrics, output_path)

        print(f"\n✅ All graphs saved to: {output_path}")

    def _create_comprehensive_chart(self, model_name, platforms, metrics, output_path):
        """Create a comprehensive comparison chart with all metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (metric_name, (metric_key, higher_better)) in enumerate(metrics.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            values = []
            for platform in platforms:
                value = self.platform_data[platform]['metrics'].get(metric_key, 0)
                values.append(value)

            # Skip if all values are zero
            if all(v == 0 for v in values):
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       fontsize=14, transform=ax.transAxes)
                ax.set_title(metric_name, fontweight='bold')
                continue

            x_pos = range(len(platforms))

            # Color coding
            if higher_better:
                colors = ['green' if v == max(values) and v > 0 else 'orange' if v > 0 else 'gray' for v in values]
            else:
                min_nonzero = min([v for v in values if v > 0]) if any(v > 0 for v in values) else 0
                colors = ['green' if v == min_nonzero and v > 0 else 'orange' if v > 0 else 'gray' for v in values]

            bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

            ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(platforms, fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # Add value labels
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                           f'{value:.2f}',
                           ha='center', va='bottom', fontweight='bold', fontsize=9)

        # Hide the last subplot if we have fewer than 6 metrics
        if len(metrics) < 6:
            axes[-1].axis('off')

        plt.suptitle(f'Platform Performance Comparison\n{model_name}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        filename = 'comprehensive_comparison.png'
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created: {filename}")


def list_available_models():
    """List all available models across all platforms"""
    platforms = {
        'Pi4': './results_pi4',
        'Pi5': './results_pi5',
        'Computer': './results_computer'
    }

    all_models = set()

    for platform, results_dir in platforms.items():
        results_path = Path(results_dir)

        if not results_path.exists():
            continue

        json_files = list(results_path.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                results = data.get('results', [])
                for result in results:
                    model = result.get('model')
                    if model:
                        all_models.add(model)
            except:
                continue

    return sorted(list(all_models))


def main():
    print("\n" + "="*60)
    print("PLATFORM PERFORMANCE COMPARATOR")
    print("="*60)
    print("\nCompares model performance across Pi4, Pi5, and Computer\n")

    # List available models
    available_models = list_available_models()

    if not available_models:
        print("❌ No models found in any platform results!")
        print("\nExpected directories:")
        print("  - ./results_pi4/")
        print("  - ./results_pi5/")
        print("  - ./results_computer/")
        return

    print(f"Found {len(available_models)} unique model(s):\n")
    for idx, model in enumerate(available_models, 1):
        print(f"  {idx}. {model}")

    # Get user input
    print("\n" + "-"*60)
    model_choice = input("\nEnter model name (or number from list): ").strip()

    # Handle numeric input
    if model_choice.isdigit():
        choice_idx = int(model_choice) - 1
        if 0 <= choice_idx < len(available_models):
            model_name = available_models[choice_idx]
        else:
            print(f"❌ Invalid choice! Please enter 1-{len(available_models)}")
            return
    else:
        model_name = model_choice

    # Create comparator and run
    comparator = PlatformComparator()

    if comparator.load_platform_data(model_name):
        comparator.print_comparison(model_name)
        comparator.generate_comparison_graphs(model_name)

        print("\n" + "="*60)
        print("COMPARISON COMPLETE!")
        print("="*60)
        print("\nCheck ./graph_comparison/ for generated graphs")
    else:
        print("\n❌ Could not load data for comparison")


if __name__ == "__main__":
    main()

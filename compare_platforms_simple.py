#!/usr/bin/env python3
"""
Simple Platform Comparison Tool
Compares TPS, TPJ, and Inference Time across Pi4, Pi5, and Computer
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import statistics


class SimplePlatformComparator:
    def __init__(self):
        self.platforms = {
            'Pi4': './results_pi4',
            'Pi5': './results_pi5',
            'Computer': './results_computer'
        }
        self.platform_data = {}

    def load_platform_data(self, model_name):
        """Load TPS, TPJ, and inference time for a specific model from all platforms"""
        print(f"\n{'='*60}")
        print(f"LOADING DATA FOR MODEL: {model_name}")
        print(f"{'='*60}\n")

        for platform, results_dir in self.platforms.items():
            results_path = Path(results_dir)

            if not results_path.exists():
                print(f"⚠️  {platform}: Directory not found ({results_dir})")
                continue

            # Find JSON files
            json_files = list(results_path.glob("*.json"))

            if not json_files:
                print(f"⚠️  {platform}: No result files found")
                continue

            # Search for the model
            model_found = False
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    results = data.get('results', [])

                    # Filter for this model
                    model_results = [r for r in results if r.get('model') == model_name]

                    if model_results:
                        # Extract metrics
                        tps_values = [r.get('tokens_per_second', 0) for r in model_results]
                        tpj_values = [r.get('tokens_per_joule', 0) for r in model_results if r.get('tokens_per_joule')]
                        inference_time_values = [r.get('inference_time_s', 0) for r in model_results]

                        self.platform_data[platform] = {
                            'tps': statistics.mean(tps_values) if tps_values else 0,
                            'tpj': statistics.mean(tpj_values) if tpj_values else 0,
                            'inference_time': statistics.mean(inference_time_values) if inference_time_values else 0,
                            'test_count': len(model_results)
                        }

                        model_found = True
                        print(f"✅ {platform}: Found {len(model_results)} test(s)")
                        print(f"   TPS: {self.platform_data[platform]['tps']:.2f}")
                        print(f"   TPJ: {self.platform_data[platform]['tpj']:.4f}")
                        print(f"   Inference Time: {self.platform_data[platform]['inference_time']:.2f}s")
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

    def print_comparison_table(self, model_name):
        """Print a comparison table"""
        if not self.platform_data:
            print("No data to compare!")
            return

        print(f"\n{'='*80}")
        print(f"PERFORMANCE COMPARISON: {model_name}")
        print(f"{'='*80}\n")

        # TPS Comparison
        print("🚀 TOKENS PER SECOND (TPS) - Higher is Better")
        print("-" * 80)
        tps_values = {platform: data['tps'] for platform, data in self.platform_data.items()}
        max_tps = max(tps_values.values()) if tps_values else 0

        for platform, data in sorted(self.platform_data.items()):
            tps = data['tps']
            bar = '█' * int((tps / max_tps * 40)) if max_tps > 0 else ''
            winner = " 🏆" if tps == max_tps and tps > 0 else ""
            print(f"   {platform:10s} {tps:8.2f} tokens/s {bar}{winner}")

        # TPJ Comparison
        print("\n⚡ TOKENS PER JOULE (TPJ) - Energy Efficiency - Higher is Better")
        print("-" * 80)
        tpj_values = {platform: data['tpj'] for platform, data in self.platform_data.items() if data['tpj'] > 0}
        max_tpj = max(tpj_values.values()) if tpj_values else 0

        for platform, data in sorted(self.platform_data.items()):
            tpj = data['tpj']
            if tpj > 0:
                bar = '█' * int((tpj / max_tpj * 40)) if max_tpj > 0 else ''
                winner = " 🏆" if tpj == max_tpj else ""
                print(f"   {platform:10s} {tpj:8.4f} tok/J    {bar}{winner}")
            else:
                print(f"   {platform:10s}      N/A")

        # Inference Time Comparison
        print("\n⏱️  INFERENCE TIME - Lower is Better")
        print("-" * 80)
        time_values = {platform: data['inference_time'] for platform, data in self.platform_data.items() if data['inference_time'] > 0}
        min_time = min(time_values.values()) if time_values else 0
        max_time = max(time_values.values()) if time_values else 0

        for platform, data in sorted(self.platform_data.items()):
            inf_time = data['inference_time']
            if inf_time > 0:
                # Reverse bar - longer bar = slower (worse)
                bar_length = int(((inf_time - min_time) / (max_time - min_time) * 40)) if max_time > min_time else 0
                bar = '█' * bar_length
                winner = " 🏆" if inf_time == min_time else ""
                print(f"   {platform:10s} {inf_time:8.2f} seconds {bar}{winner}")
            else:
                print(f"   {platform:10s}      N/A")

        print(f"\n{'='*80}\n")

    def create_comparison_graphs(self, model_name, output_dir="./platform_comparison"):
        """Create comparison bar charts"""
        if not self.platform_data:
            print("No data to visualize!")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        platforms = list(self.platform_data.keys())

        # Graph 1: TPS Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        tps_values = [self.platform_data[p]['tps'] for p in platforms]
        colors = ['green' if v == max(tps_values) and v > 0 else 'steelblue' for v in tps_values]
        bars = ax.bar(platforms, tps_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

        ax.set_xlabel('Platform', fontsize=13, fontweight='bold')
        ax.set_ylabel('Tokens per Second', fontsize=13, fontweight='bold')
        ax.set_title(f'TPS Comparison - {model_name}\n(Higher is Better)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, tps_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tps_values)*0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_path / 'tps_comparison.png', dpi=300)
        plt.close()
        print(f"✅ Created: tps_comparison.png")

        # Graph 2: TPJ Comparison (only if data available)
        tpj_values = [self.platform_data[p]['tpj'] for p in platforms]
        if any(v > 0 for v in tpj_values):
            fig, ax = plt.subplots(figsize=(10, 6))
            max_tpj = max(tpj_values)
            colors = ['green' if v == max_tpj and v > 0 else 'orange' if v > 0 else 'gray' for v in tpj_values]
            bars = ax.bar(platforms, tpj_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

            ax.set_xlabel('Platform', fontsize=13, fontweight='bold')
            ax.set_ylabel('Tokens per Joule', fontsize=13, fontweight='bold')
            ax.set_title(f'Energy Efficiency (TPJ) - {model_name}\n(Higher is Better)', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, tpj_values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_tpj*0.02,
                           f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

            plt.tight_layout()
            plt.savefig(output_path / 'tpj_comparison.png', dpi=300)
            plt.close()
            print(f"✅ Created: tpj_comparison.png")
        else:
            print("⚠️  Skipping TPJ graph: No energy data available")

        # Graph 3: Inference Time Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        time_values = [self.platform_data[p]['inference_time'] for p in platforms]
        min_time = min([v for v in time_values if v > 0]) if any(v > 0 for v in time_values) else 0
        colors = ['green' if v == min_time and v > 0 else 'coral' for v in time_values]
        bars = ax.bar(platforms, time_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

        ax.set_xlabel('Platform', fontsize=13, fontweight='bold')
        ax.set_ylabel('Inference Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_title(f'Inference Time Comparison - {model_name}\n(Lower is Better)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, time_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(time_values)*0.02,
                       f'{val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_path / 'inference_time_comparison.png', dpi=300)
        plt.close()
        print(f"✅ Created: inference_time_comparison.png")

        # Combined Graph
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # TPS
        tps_values = [self.platform_data[p]['tps'] for p in platforms]
        colors = ['green' if v == max(tps_values) and v > 0 else 'steelblue' for v in tps_values]
        axes[0].bar(platforms, tps_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('Tokens/Second', fontweight='bold')
        axes[0].set_title('TPS (Higher Better)', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(tps_values):
            axes[0].text(i, v + max(tps_values)*0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

        # TPJ
        tpj_values = [self.platform_data[p]['tpj'] for p in platforms]
        if any(v > 0 for v in tpj_values):
            max_tpj = max(tpj_values)
            colors = ['green' if v == max_tpj and v > 0 else 'orange' if v > 0 else 'gray' for v in tpj_values]
            axes[1].bar(platforms, tpj_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            axes[1].set_ylabel('Tokens/Joule', fontweight='bold')
            axes[1].set_title('Energy Efficiency (Higher Better)', fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)
            for i, v in enumerate(tpj_values):
                if v > 0:
                    axes[1].text(i, v + max_tpj*0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'No Energy Data', ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title('Energy Efficiency (N/A)', fontweight='bold')

        # Inference Time
        time_values = [self.platform_data[p]['inference_time'] for p in platforms]
        min_time = min([v for v in time_values if v > 0]) if any(v > 0 for v in time_values) else 0
        colors = ['green' if v == min_time and v > 0 else 'coral' for v in time_values]
        axes[2].bar(platforms, time_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[2].set_ylabel('Seconds', fontweight='bold')
        axes[2].set_title('Inference Time (Lower Better)', fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        for i, v in enumerate(time_values):
            if v > 0:
                axes[2].text(i, v + max(time_values)*0.02, f'{v:.2f}s', ha='center', va='bottom', fontweight='bold')

        plt.suptitle(f'Platform Performance Comparison - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'combined_comparison.png', dpi=300)
        plt.close()
        print(f"✅ Created: combined_comparison.png")

        print(f"\n✅ All graphs saved to: {output_path}")


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
    print("SIMPLE PLATFORM COMPARATOR")
    print("Compares TPS, TPJ, and Inference Time")
    print("="*60)

    # List available models
    available_models = list_available_models()

    if not available_models:
        print("\n❌ No models found in any platform results!")
        print("\nExpected directories:")
        print("  - ./results_pi4/")
        print("  - ./results_pi5/")
        print("  - ./results_computer/")
        return

    print(f"\nFound {len(available_models)} unique model(s):\n")
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
    comparator = SimplePlatformComparator()

    if comparator.load_platform_data(model_name):
        comparator.print_comparison_table(model_name)
        comparator.create_comparison_graphs(model_name)

        print("\n" + "="*60)
        print("COMPARISON COMPLETE!")
        print("="*60)
        print("\nCheck ./platform_comparison/ for generated graphs")
    else:
        print("\n❌ Could not load data for comparison")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Model Ranking System
Simple rank-based system: ranks models 1st, 2nd, 3rd, etc. in each category.
Total ranking is the sum of all category ranks (lower is better).
"""

import json
import csv
from pathlib import Path

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib/numpy not installed. Run: pip install matplotlib numpy")
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas not installed. Run: pip install pandas")
    PANDAS_AVAILABLE = False


class ModelRankingSystem:
    """
    Simple rank-based model ranking system
    Ranks models 1st, 2nd, 3rd, etc. in each category
    Total ranking is the sum of all category ranks (lower is better)
    """

    def __init__(self,
                 teaching_ratings_file="teaching_effectiveness_ratings_computer.json",
                 performance_summary_file="analysis_summary_pi4.csv"):
        """
        Initialize the ranking system

        Args:
            teaching_ratings_file: Path to teaching effectiveness ratings JSON
            performance_summary_file: Path to performance summary CSV
        """
        self.teaching_file = Path(teaching_ratings_file)
        self.performance_file = Path(performance_summary_file)

        # Categories to rank (no weights needed)
        self.categories = [
            'teaching_effectiveness',
            'tokens_per_second',
            'tokens_per_joule',
            'energy_consumption',     # lower is better
            'time_to_first_token',    # lower is better
            'avg_iops'
        ]

        self.model_data = {}
        self.rankings = {}

        # Load data
        self._load_data()


    def _load_data(self):
        """Load teaching effectiveness and performance data"""
        print(f"\nLoading data from:")
        print(f"  Teaching ratings: {self.teaching_file}")
        print(f"  Performance data: {self.performance_file}")

        # Load teaching effectiveness ratings
        teaching_scores = {}
        if self.teaching_file.exists():
            with open(self.teaching_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for model, stats in data.get('model_summary', {}).items():
                    teaching_scores[model] = stats.get('average_score', 0)
        else:
            print(f"Warning: {self.teaching_file} not found")

        # Load performance metrics
        if self.performance_file.exists():
            with open(self.performance_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model = row['model']

                    self.model_data[model] = {
                        'teaching_effectiveness': teaching_scores.get(model, 0),
                        'tokens_per_second': float(row.get('avg_tokens_per_second', 0)),
                        'tokens_per_joule': float(row.get('avg_tokens_per_joule', 0)),
                        'energy_consumption': float(row.get('total_energy_joules', 0)),
                        'time_to_first_token': float(row.get('avg_ttft_s', 0)),
                        'avg_iops': float(row.get('avg_io_iops', 0)),
                        'model_parameters': row.get('model_parameters', 'unknown'),
                        'num_tests': int(row.get('num_tests', 0))
                    }
        else:
            print(f"Warning: {self.performance_file} not found")

        print(f"\nLoaded data for {len(self.model_data)} models")

    def _rank_models_by_category(self, category, higher_is_better=True):
        """
        Rank models in a specific category

        Args:
            category: The category to rank by
            higher_is_better: If True, higher values get better ranks (rank 1)

        Returns:
            Dictionary mapping model names to their rank in this category
        """
        # Get all models and their values for this category
        model_values = [(model, data[category]) for model, data in self.model_data.items()]

        # Sort by value (descending if higher is better, ascending if lower is better)
        model_values.sort(key=lambda x: x[1], reverse=higher_is_better)

        # Assign ranks (1 = best)
        ranks = {}
        for rank, (model, value) in enumerate(model_values, start=1):
            ranks[model] = rank

        return ranks

    def calculate_rankings(self):
        """Calculate rank-based scores for all models"""
        print("\n" + "="*60)
        print("CALCULATING MODEL RANKINGS")
        print("="*60)

        # Rank models in each category
        category_ranks = {}

        # Higher is better
        category_ranks['teaching_effectiveness'] = self._rank_models_by_category(
            'teaching_effectiveness', higher_is_better=True
        )
        category_ranks['tokens_per_second'] = self._rank_models_by_category(
            'tokens_per_second', higher_is_better=True
        )
        category_ranks['tokens_per_joule'] = self._rank_models_by_category(
            'tokens_per_joule', higher_is_better=True
        )
        category_ranks['avg_iops'] = self._rank_models_by_category(
            'avg_iops', higher_is_better=True
        )

        # Lower is better
        category_ranks['energy_consumption'] = self._rank_models_by_category(
            'energy_consumption', higher_is_better=False
        )
        category_ranks['time_to_first_token'] = self._rank_models_by_category(
            'time_to_first_token', higher_is_better=False
        )

        # Calculate total rank score (sum of all ranks - lower is better)
        for model, data in self.model_data.items():
            ranks = {}
            for category in self.categories:
                ranks[category] = category_ranks[category][model]

            # Total score is sum of all ranks (lower is better)
            total_rank = sum(ranks.values())

            self.rankings[model] = {
                'category_ranks': ranks,
                'raw_values': data,
                'total_rank': total_rank
            }

        # Sort by total rank (lower is better)
        sorted_rankings = sorted(
            self.rankings.items(),
            key=lambda x: x[1]['total_rank'],
            reverse=False  # Lower rank is better
        )

        print("\nTop 10 Models (by total rank - lower is better):")
        for i, (model, data) in enumerate(sorted_rankings[:10], 1):
            print(f"  {i}. {model}: Total Rank Score = {data['total_rank']}")

        return sorted_rankings

    def print_detailed_rankings(self):
        """Print detailed rankings for all categories"""
        print("\n" + "="*80)
        print("DETAILED CATEGORY RANKINGS")
        print("="*80)

        for category in self.categories:
            print(f"\n{category.upper().replace('_', ' ')}")
            print("-" * 60)

            # Sort by this category rank
            sorted_by_category = sorted(
                self.rankings.items(),
                key=lambda x: x[1]['category_ranks'][category],
                reverse=False  # Lower rank is better
            )

            for model, data in sorted_by_category[:10]:
                rank = data['category_ranks'][category]
                raw_value = data['raw_values'][category]
                print(f"  Rank {rank:3d}: {model:30s} (Value: {raw_value:.2f})")

    def export_rankings_csv(self, output_file="model_rankings.csv"):
        """Export rankings to CSV file"""
        if not PANDAS_AVAILABLE:
            print("Cannot export to CSV: pandas not installed")
            return

        # Prepare data for export
        export_data = []
        for model, data in self.rankings.items():
            row = {
                'model': model,
                'total_rank': data['total_rank'],
                'model_parameters': data['raw_values']['model_parameters']
            }

            # Add category ranks
            for category, rank in data['category_ranks'].items():
                row[f'{category}_rank'] = rank

            # Add raw values
            for category, value in data['raw_values'].items():
                if category not in ['model_parameters', 'num_tests']:
                    row[f'{category}_raw'] = value

            export_data.append(row)

        # Create DataFrame and sort by total rank (lower is better)
        df = pd.DataFrame(export_data)
        df = df.sort_values('total_rank', ascending=True)
        df.to_csv(output_file, index=False)

        print(f"\nRankings exported to: {output_file}")

    def generate_visualizations(self, output_dir="./ranking_graphs"):
        """Generate comprehensive ranking visualizations"""
        if not MATPLOTLIB_AVAILABLE:
            print("Cannot generate graphs: matplotlib not installed")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nGenerating ranking visualizations...")

        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

        self._plot_overall_rankings(output_path)
        self._plot_category_scores_heatmap(output_path)
        self._plot_radar_charts_top_models(output_path)
        self._plot_category_rankings(output_path)

        print(f"Visualizations saved to: {output_path}")

    def _plot_overall_rankings(self, output_path):
        """Plot overall model rankings"""
        sorted_models = sorted(
            self.rankings.items(),
            key=lambda x: x[1]['total_rank'],
            reverse=False  # Lower rank is better
        )

        models = [m[0] for m in sorted_models[:15]]  # Top 15
        total_ranks = [m[1]['total_rank'] for m in sorted_models[:15]]

        fig, ax = plt.subplots(figsize=(14, 8))

        # Color gradient based on rank (inverted - lower is better)
        max_rank = max(total_ranks)
        min_rank = min(total_ranks)
        colors = plt.cm.RdYlGn_r([(r - min_rank) / (max_rank - min_rank) for r in total_ranks])
        bars = ax.barh(range(len(models)), total_ranks, color=colors, alpha=0.8, edgecolor='black')

        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.replace(':', '\n') for m in models])
        ax.set_xlabel('Total Rank Score (Lower is Better)', fontsize=12, fontweight='bold')
        ax.set_title('Overall Model Rankings (Sum of Category Ranks)', fontsize=16, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_xaxis()  # Invert so best (lowest) is on the right

        # Add rank labels
        for i, (bar, rank) in enumerate(zip(bars, total_ranks)):
            ax.text(rank - 1, i, f'{rank}', va='center', fontweight='bold')

        # Add position numbers
        for i in range(len(models)):
            ax.text(max(total_ranks) + max(total_ranks)*0.02, i, f'#{i+1}', va='center', ha='left', fontweight='bold', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_path / 'overall_rankings.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Created: overall_rankings.png")

    def _plot_category_scores_heatmap(self, output_path):
        """Plot heatmap of category ranks for all models"""
        sorted_models = sorted(
            self.rankings.items(),
            key=lambda x: x[1]['total_rank'],
            reverse=False  # Lower rank is better
        )

        models = [m[0] for m in sorted_models[:20]]  # Top 20
        categories = list(self.categories)

        # Create rank matrix
        rank_matrix = []
        for model, _ in sorted_models[:20]:
            ranks = [self.rankings[model]['category_ranks'][cat] for cat in categories]
            rank_matrix.append(ranks)

        rank_matrix = np.array(rank_matrix)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Use inverted colormap since lower rank is better
        max_rank = np.max(rank_matrix)
        im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=max_rank)

        # Set ticks
        ax.set_xticks(np.arange(len(categories)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45, ha='right')
        ax.set_yticklabels(models)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Rank (Lower is Better)', rotation=270, labelpad=20)

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(categories)):
                text = ax.text(j, i, f'{rank_matrix[i, j]:.0f}',
                             ha='center', va='center', color='black', fontsize=8)

        ax.set_title('Model Ranking Heatmap Across Categories', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_path / 'category_ranks_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Created: category_ranks_heatmap.png")

    def _plot_radar_charts_top_models(self, output_path):
        """Plot radar charts for top 6 models"""
        sorted_models = sorted(
            self.rankings.items(),
            key=lambda x: x[1]['total_rank'],
            reverse=False  # Lower rank is better
        )[:6]

        categories = list(self.categories)
        num_vars = len(categories)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set2(range(len(sorted_models)))

        # Find max rank across all models for normalization
        all_ranks = []
        for _, data in self.rankings.items():
            all_ranks.extend(data['category_ranks'].values())
        max_rank = max(all_ranks)

        for idx, (model, data) in enumerate(sorted_models):
            # Invert ranks for visualization (so rank 1 = max value on radar)
            values = [max_rank - data['category_ranks'][cat] + 1 for cat in categories]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], fontsize=10)
        ax.set_ylim(0, max_rank)
        ax.set_title('Top 6 Models - Multi-dimensional Performance\n(Higher = Better Rank)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_path / 'top_models_radar_chart.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Created: top_models_radar_chart.png")

    def _plot_category_rankings(self, output_path):
        """Plot individual category rankings"""
        categories = list(self.categories)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, category in enumerate(categories):
            ax = axes[idx]

            # Sort by this category (lower rank is better)
            sorted_by_cat = sorted(
                self.rankings.items(),
                key=lambda x: x[1]['category_ranks'][category],
                reverse=False
            )[:10]

            models = [m[0] for m in sorted_by_cat]
            ranks = [m[1]['category_ranks'][category] for m in sorted_by_cat]

            # Color gradient (inverted - lower rank is better)
            max_rank = max(ranks)
            min_rank = min(ranks)
            colors = plt.cm.RdYlGn_r([(r - min_rank) / (max_rank - min_rank) if max_rank > min_rank else 0.5 for r in ranks])
            bars = ax.barh(range(len(models)), ranks, color=colors, alpha=0.8, edgecolor='black')

            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([m.replace(':', '\n') for m in models], fontsize=8)
            ax.set_xlabel('Rank (Lower is Better)', fontsize=10)
            ax.set_title(f'{category.replace("_", " ").title()}',
                        fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.invert_xaxis()  # Invert so best (lowest) is on the right

            # Add rank labels
            for i, (bar, rank) in enumerate(zip(bars, ranks)):
                ax.text(rank - 0.5, i, f'{rank}', va='center', fontsize=8)

        plt.suptitle('Individual Category Rankings (Top 10 per Category)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'category_rankings.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Created: category_rankings.png")


def main():
    print("="*60)
    print("MODEL RANKING SYSTEM")
    print("="*60)

    # Initialize ranking system
    ranker = ModelRankingSystem()

    # Calculate rankings
    ranker.calculate_rankings()

    # Print detailed rankings
    ranker.print_detailed_rankings()

    # Export to CSV
    ranker.export_rankings_csv()

    # Generate visualizations
    ranker.generate_visualizations()

    print("\n" + "="*60)
    print("RANKING COMPLETE!")
    print("="*60)
    print("\nRanking method: Simple rank-based system")
    print("Each model is ranked 1st, 2nd, 3rd, etc. in each category.")
    print("Total rank = sum of all category ranks (lower is better).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick script to visualize Ollama model sizes
"""

import ollama
import matplotlib.pyplot as plt


def get_model_sizes():
    """Get all Ollama models with their sizes"""
    try:
        models_list = ollama.list()

        model_data = []
        for model in models_list['models']:
            name = model['model']
            # Size is in bytes, convert to GB
            size_bytes = model.get('size', 0)
            size_gb = size_bytes / (1024 ** 3)  # Convert bytes to GB

            model_data.append({
                'name': name,
                'size_gb': size_gb
            })

        return model_data

    except Exception as e:
        print(f"Error getting model list: {e}")
        return []


def plot_model_sizes(model_data):
    """Create a bar chart of model sizes"""

    if not model_data:
        print("No model data to plot!")
        return

    # Sort by size (ascending)
    model_data = sorted(model_data, key=lambda x: x['size_gb'])

    # Extract data for plotting
    names = [m['name'] for m in model_data]
    sizes = [m['size_gb'] for m in model_data]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.4)))

    # Create horizontal bar chart for better readability with long model names
    colors = plt.cm.viridis([s/max(sizes) if max(sizes) > 0 else 0 for s in sizes])
    bars = ax.barh(range(len(names)), sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Customize
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Size (GB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Ollama Model Sizes', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, size in zip(bars, sizes):
        width = bar.get_width()
        ax.text(width + max(sizes)*0.01, bar.get_y() + bar.get_height()/2,
                f'{size:.2f} GB',
                ha='left', va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()

    # Save
    output_path = './model_sizes.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] Graph saved to: {output_path}")

    plt.show()

    # Print summary
    print("\n" + "="*60)
    print("MODEL SIZE SUMMARY")
    print("="*60)
    print(f"\nTotal models: {len(model_data)}")
    print(f"Total size: {sum(sizes):.2f} GB")
    print(f"\nLargest: {names[-1]} - {sizes[-1]:.2f} GB")
    print(f"Smallest: {names[0]} - {sizes[0]:.2f} GB")
    print(f"Average: {sum(sizes)/len(sizes):.2f} GB")
    print("="*60)


def main():
    print("\n" + "="*60)
    print("OLLAMA MODEL SIZE VISUALIZATION")
    print("="*60 + "\n")

    print("Fetching model information from Ollama...")
    model_data = get_model_sizes()

    if model_data:
        print(f"Found {len(model_data)} model(s)")
        plot_model_sizes(model_data)
    else:
        print("No models found!")


if __name__ == "__main__":
    main()

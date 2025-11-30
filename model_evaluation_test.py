#!/usr/bin/env python3
"""
Ollama Model Evaluation - BLEU Score and Perplexity Testing
Tests models on text generation quality and language modeling capability
"""

import ollama
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import time

# Try to import BLEU score library, install if needed
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
except ImportError:
    print("Installing NLTK for BLEU score calculation...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'nltk'])
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt', quiet=True)


class ModelEvaluator:
    def __init__(self, models=None):
        """Initialize evaluator with list of models to test"""
        if models is None:
            # Default to your available models
            self.models = ['gemma3:270m', 'cogito:8b']
        else:
            self.models = models

        self.results = {}
        self.output_dir = Path("./evaluation_results")
        self.output_dir.mkdir(exist_ok=True)

        # Test dataset: questions with reference answers
        self.test_data = [
            {
                "prompt": "What is the capital of France?",
                "reference": "The capital of France is Paris."
            },
            {
                "prompt": "Explain photosynthesis in one sentence.",
                "reference": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar."
            },
            {
                "prompt": "What is 15 multiplied by 8?",
                "reference": "15 multiplied by 8 equals 120."
            },
            {
                "prompt": "Name the three primary colors.",
                "reference": "The three primary colors are red, blue, and yellow."
            },
            {
                "prompt": "What is the chemical formula for water?",
                "reference": "The chemical formula for water is H2O."
            },
            {
                "prompt": "Who wrote Romeo and Juliet?",
                "reference": "William Shakespeare wrote Romeo and Juliet."
            },
            {
                "prompt": "What is the largest planet in our solar system?",
                "reference": "Jupiter is the largest planet in our solar system."
            },
            {
                "prompt": "Define gravity in simple terms.",
                "reference": "Gravity is a force that pulls objects toward each other, especially toward the center of the Earth."
            },
            {
                "prompt": "How many continents are there?",
                "reference": "There are seven continents on Earth."
            },
            {
                "prompt": "What is the boiling point of water in Celsius?",
                "reference": "The boiling point of water is 100 degrees Celsius at sea level."
            }
        ]

        # Perplexity test sentences
        self.perplexity_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "The Earth orbits around the Sun once every 365 days.",
            "Mathematics is the study of numbers, shapes, and patterns."
        ]

    def generate_response(self, model, prompt, max_tokens=100):
        """Generate response from model"""
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'num_predict': max_tokens
                }
            )
            return response['response'].strip()
        except Exception as e:
            print(f"Error generating response for {model}: {e}")
            return ""

    def calculate_bleu_score(self, reference, candidate):
        """Calculate BLEU score between reference and candidate text"""
        try:
            # Tokenize sentences
            reference_tokens = word_tokenize(reference.lower())
            candidate_tokens = word_tokenize(candidate.lower())

            # Use smoothing to avoid zero scores
            smoothing = SmoothingFunction().method1

            # Calculate BLEU score (using single reference)
            score = sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                smoothing_function=smoothing
            )
            return score
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return 0.0

    def calculate_perplexity(self, model, sentence):
        """
        Calculate perplexity for a sentence
        Note: This is an approximation since Ollama doesn't expose log probabilities directly
        We'll use the model's ability to predict the next word
        """
        try:
            words = sentence.split()
            if len(words) < 2:
                return float('inf')

            # Calculate average prediction confidence
            total_log_prob = 0
            num_predictions = 0

            for i in range(1, len(words)):
                context = ' '.join(words[:i])
                target = words[i]

                # Generate next word prediction
                prompt = f"Complete this sentence with the next word only: {context}"
                response = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options={'num_predict': 5, 'temperature': 0.1}
                )

                predicted = response['response'].strip().split()[0] if response['response'] else ""

                # Simple matching score (1 if correct, 0.1 if wrong)
                if predicted.lower() == target.lower():
                    prob = 0.9
                else:
                    prob = 0.1

                total_log_prob += np.log(prob)
                num_predictions += 1

            # Calculate perplexity
            avg_log_prob = total_log_prob / num_predictions
            perplexity = np.exp(-avg_log_prob)

            return perplexity

        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')

    def evaluate_model(self, model):
        """Run full evaluation on a model"""
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model}")
        print(f"{'='*60}")

        model_results = {
            'model': model,
            'bleu_scores': [],
            'perplexity_scores': [],
            'responses': [],
            'avg_bleu': 0,
            'avg_perplexity': 0,
            'total_time': 0
        }

        start_time = time.time()

        # BLEU Score Evaluation
        print("\n📊 Running BLEU Score Evaluation...")
        for i, test_case in enumerate(self.test_data, 1):
            print(f"  Test {i}/{len(self.test_data)}: {test_case['prompt'][:50]}...")

            response = self.generate_response(model, test_case['prompt'])
            bleu = self.calculate_bleu_score(test_case['reference'], response)

            model_results['bleu_scores'].append(bleu)
            model_results['responses'].append({
                'prompt': test_case['prompt'],
                'reference': test_case['reference'],
                'response': response,
                'bleu': bleu
            })

            print(f"    BLEU Score: {bleu:.4f}")

        # Perplexity Evaluation
        print("\n📈 Running Perplexity Evaluation...")
        for i, sentence in enumerate(self.perplexity_sentences, 1):
            print(f"  Sentence {i}/{len(self.perplexity_sentences)}: {sentence[:50]}...")

            perplexity = self.calculate_perplexity(model, sentence)
            model_results['perplexity_scores'].append(perplexity)

            print(f"    Perplexity: {perplexity:.2f}")

        # Calculate averages
        model_results['avg_bleu'] = np.mean(model_results['bleu_scores'])
        model_results['avg_perplexity'] = np.mean(model_results['perplexity_scores'])
        model_results['total_time'] = time.time() - start_time

        print(f"\n✅ Evaluation complete for {model}")
        print(f"   Average BLEU Score: {model_results['avg_bleu']:.4f}")
        print(f"   Average Perplexity: {model_results['avg_perplexity']:.2f}")
        print(f"   Total Time: {model_results['total_time']:.2f}s")

        return model_results

    def run_evaluation(self):
        """Run evaluation on all models"""
        print("🚀 Starting Model Evaluation")
        print(f"Models to test: {', '.join(self.models)}")

        for model in self.models:
            try:
                results = self.evaluate_model(model)
                self.results[model] = results
            except Exception as e:
                print(f"❌ Failed to evaluate {model}: {e}")

        # Save results
        self.save_results()

        # Generate visualizations
        self.create_visualizations()

        print(f"\n✅ All evaluations complete!")
        print(f"📁 Results saved to: {self.output_dir}")

    def save_results(self):
        """Save results to JSON file"""
        output_file = self.output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for model, data in self.results.items():
            serializable_results[model] = {
                'model': data['model'],
                'avg_bleu': float(data['avg_bleu']),
                'avg_perplexity': float(data['avg_perplexity']),
                'total_time': float(data['total_time']),
                'bleu_scores': [float(x) for x in data['bleu_scores']],
                'perplexity_scores': [float(x) for x in data['perplexity_scores']],
                'responses': data['responses']
            }

        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

    def create_visualizations(self):
        """Create visualization graphs"""
        if not self.results:
            print("No results to visualize")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))

        # 1. Average BLEU Score Comparison
        ax1 = plt.subplot(2, 3, 1)
        models = list(self.results.keys())
        avg_bleu = [self.results[m]['avg_bleu'] for m in models]
        colors = ['#3498db', '#e74c3c'][:len(models)]

        bars1 = ax1.bar(range(len(models)), avg_bleu, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Model', fontweight='bold')
        ax1.set_ylabel('Average BLEU Score', fontweight='bold')
        ax1.set_title('Average BLEU Score Comparison', fontweight='bold', fontsize=12)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.replace(':', '\n') for m in models], rotation=0)
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold')

        # 2. Average Perplexity Comparison
        ax2 = plt.subplot(2, 3, 2)
        avg_perplexity = [self.results[m]['avg_perplexity'] for m in models]

        bars2 = ax2.bar(range(len(models)), avg_perplexity, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Model', fontweight='bold')
        ax2.set_ylabel('Average Perplexity (lower is better)', fontweight='bold')
        ax2.set_title('Average Perplexity Comparison', fontweight='bold', fontsize=12)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace(':', '\n') for m in models], rotation=0)
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontweight='bold')

        # 3. BLEU Score Distribution
        ax3 = plt.subplot(2, 3, 3)
        for i, model in enumerate(models):
            bleu_scores = self.results[model]['bleu_scores']
            ax3.plot(range(1, len(bleu_scores) + 1), bleu_scores,
                    marker='o', label=model, color=colors[i], linewidth=2, markersize=6)

        ax3.set_xlabel('Test Case', fontweight='bold')
        ax3.set_ylabel('BLEU Score', fontweight='bold')
        ax3.set_title('BLEU Score per Test Case', fontweight='bold', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Perplexity Distribution
        ax4 = plt.subplot(2, 3, 4)
        for i, model in enumerate(models):
            perplexity_scores = self.results[model]['perplexity_scores']
            ax4.plot(range(1, len(perplexity_scores) + 1), perplexity_scores,
                    marker='s', label=model, color=colors[i], linewidth=2, markersize=6)

        ax4.set_xlabel('Sentence', fontweight='bold')
        ax4.set_ylabel('Perplexity', fontweight='bold')
        ax4.set_title('Perplexity per Sentence', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Combined Score (BLEU high + Perplexity low = better)
        ax5 = plt.subplot(2, 3, 5)
        # Normalize scores for comparison (0-1 scale)
        max_perplexity = max(avg_perplexity)
        normalized_bleu = avg_bleu  # Already 0-1
        normalized_perplexity = [1 - (p / max_perplexity) for p in avg_perplexity]  # Invert so higher is better
        combined_score = [(b + p) / 2 for b, p in zip(normalized_bleu, normalized_perplexity)]

        bars5 = ax5.bar(range(len(models)), combined_score, color=colors, alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Model', fontweight='bold')
        ax5.set_ylabel('Combined Score (0-1)', fontweight='bold')
        ax5.set_title('Overall Performance Score', fontweight='bold', fontsize=12)
        ax5.set_xticks(range(len(models)))
        ax5.set_xticklabels([m.replace(':', '\n') for m in models], rotation=0)
        ax5.grid(axis='y', alpha=0.3)

        for bar in bars5:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold')

        # 6. Evaluation Time
        ax6 = plt.subplot(2, 3, 6)
        eval_times = [self.results[m]['total_time'] for m in models]

        bars6 = ax6.bar(range(len(models)), eval_times, color=colors, alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Model', fontweight='bold')
        ax6.set_ylabel('Time (seconds)', fontweight='bold')
        ax6.set_title('Total Evaluation Time', fontweight='bold', fontsize=12)
        ax6.set_xticks(range(len(models)))
        ax6.set_xticklabels([m.replace(':', '\n') for m in models], rotation=0)
        ax6.grid(axis='y', alpha=0.3)

        for bar in bars6:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s',
                    ha='center', va='bottom', fontweight='bold')

        plt.suptitle('Model Evaluation: BLEU Score & Perplexity Analysis',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save figure
        output_file = self.output_dir / f"evaluation_graphs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Graphs saved to: {output_file}")

        # Show plot
        plt.show()


def main():
    """Main execution function"""
    print("=" * 60)
    print("MODEL EVALUATION - BLEU SCORE & PERPLEXITY")
    print("=" * 60)

    # Initialize evaluator with your models
    evaluator = ModelEvaluator(models=['gemma3:270m', 'cogito:8b'])

    # Run evaluation
    evaluator.run_evaluation()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Model Evaluation - ROUGE Score and Perplexity Testing
Tests models on text generation quality and language modeling capability
Enhanced version with real dataset support
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import ollama
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import time

# Try to import ROUGE score library, install if needed
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Installing rouge-score for ROUGE score calculation...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rouge-score'])
        from rouge_score import rouge_scorer
        ROUGE_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Could not install rouge-score library: {e}")
        ROUGE_AVAILABLE = False

# Try to import Hugging Face datasets library
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Installing datasets library for real dataset support...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'datasets'])
        from datasets import load_dataset
        DATASETS_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Could not install datasets library: {e}")
        print("Continuing with built-in test data only...")
        DATASETS_AVAILABLE = False


class ModelEvaluator:
    def __init__(self, models=None, dataset_name='squad', num_samples=50, use_hf_dataset=True):
        """
        Initialize evaluator with list of models to test

        Args:
            models: List of model names to evaluate
            dataset_name: Name of dataset to use ('squad', 'triviaqa', 'builtin')
            num_samples: Number of samples to evaluate (default: 50)
            use_hf_dataset: Whether to use Hugging Face datasets (default: True)
        """
        if models is None:
            self.models = ['gemma3:270m', 'cogito:8b']
        else:
            self.models = models

        self.results = {}
        self.output_dir = Path("./evaluation_results")
        self.output_dir.mkdir(exist_ok=True)

        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.use_hf_dataset = use_hf_dataset and DATASETS_AVAILABLE

        # Load test data
        self.test_data = self._load_test_data()
        self.perplexity_sentences = self._load_perplexity_sentences()

    def _load_test_data(self):
        """Load test dataset from Hugging Face or use built-in data"""
        if not self.use_hf_dataset or not DATASETS_AVAILABLE:
            print(f"[INFO] Using built-in test data (10 samples)")
            return self._get_builtin_data()

        print(f"[INFO] Loading dataset: {self.dataset_name} with {self.num_samples} samples")

        try:
            if self.dataset_name == 'squad':
                return self._load_squad_dataset()
            elif self.dataset_name == 'triviaqa':
                return self._load_triviaqa_dataset()
            elif self.dataset_name == 'natural_questions':
                return self._load_natural_questions_dataset()
            elif self.dataset_name == 'builtin':
                return self._get_builtin_data()
            else:
                print(f"[WARNING] Unknown dataset '{self.dataset_name}', using built-in data")
                return self._get_builtin_data()
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            print("[INFO] Falling back to built-in test data")
            return self._get_builtin_data()

    def _load_squad_dataset(self):
        """Load SQuAD dataset for question answering"""
        try:
            print("[*] Loading SQuAD dataset...")
            dataset = load_dataset('squad', split='validation')

            test_data = []
            for i, item in enumerate(dataset):
                if i >= self.num_samples:
                    break

                # SQuAD has context, question, and answers
                test_data.append({
                    'prompt': item['question'],
                    'reference': item['answers']['text'][0] if item['answers']['text'] else "No answer",
                    'context': item['context'][:200] + "..."  # Truncate context for display
                })

            print(f"[OK] Loaded {len(test_data)} samples from SQuAD")
            return test_data
        except Exception as e:
            print(f"[ERROR] Failed to load SQuAD: {e}")
            raise

    def _load_triviaqa_dataset(self):
        """Load TriviaQA dataset for trivia questions"""
        try:
            print("[*] Loading TriviaQA dataset...")
            dataset = load_dataset('trivia_qa', 'unfiltered.nocontext', split='validation')

            test_data = []
            for i, item in enumerate(dataset):
                if i >= self.num_samples:
                    break

                # TriviaQA has question and answer
                test_data.append({
                    'prompt': item['question'],
                    'reference': item['answer']['value']
                })

            print(f"[OK] Loaded {len(test_data)} samples from TriviaQA")
            return test_data
        except Exception as e:
            print(f"[ERROR] Failed to load TriviaQA: {e}")
            raise

    def _load_natural_questions_dataset(self):
        """Load Google Natural Questions dataset"""
        try:
            print("[*] Loading Natural Questions dataset...")
            dataset = load_dataset('natural_questions', split='validation')

            test_data = []
            for i, item in enumerate(dataset):
                if i >= self.num_samples:
                    break

                # Extract question and short answer
                question = item['question']['text']
                annotations = item['annotations']

                if annotations and annotations['short_answers']:
                    short_answer = annotations['short_answers'][0]
                    answer_text = item['document']['tokens'][
                        short_answer['start_token']:short_answer['end_token']
                    ]
                    reference = ' '.join(answer_text)
                else:
                    reference = "No short answer available"

                test_data.append({
                    'prompt': question,
                    'reference': reference
                })

            print(f"[OK] Loaded {len(test_data)} samples from Natural Questions")
            return test_data
        except Exception as e:
            print(f"[ERROR] Failed to load Natural Questions: {e}")
            raise

    def _get_builtin_data(self):
        """Get built-in test dataset (original 10 questions + 40 more)"""
        return [
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
            },
            # Additional 40 questions for expanded testing
            {
                "prompt": "What is the speed of light?",
                "reference": "The speed of light is approximately 299,792,458 meters per second in a vacuum."
            },
            {
                "prompt": "Who painted the Mona Lisa?",
                "reference": "Leonardo da Vinci painted the Mona Lisa."
            },
            {
                "prompt": "What is the smallest unit of life?",
                "reference": "The cell is the smallest unit of life."
            },
            {
                "prompt": "What year did World War II end?",
                "reference": "World War II ended in 1945."
            },
            {
                "prompt": "What is the main gas in Earth's atmosphere?",
                "reference": "Nitrogen is the main gas in Earth's atmosphere."
            },
            {
                "prompt": "Who invented the telephone?",
                "reference": "Alexander Graham Bell invented the telephone."
            },
            {
                "prompt": "What is the square root of 144?",
                "reference": "The square root of 144 is 12."
            },
            {
                "prompt": "What is the capital of Japan?",
                "reference": "The capital of Japan is Tokyo."
            },
            {
                "prompt": "How many bones are in the human body?",
                "reference": "There are 206 bones in the adult human body."
            },
            {
                "prompt": "What is the freezing point of water in Fahrenheit?",
                "reference": "The freezing point of water is 32 degrees Fahrenheit."
            },
            {
                "prompt": "Who wrote 'To Kill a Mockingbird'?",
                "reference": "Harper Lee wrote 'To Kill a Mockingbird'."
            },
            {
                "prompt": "What is the chemical symbol for gold?",
                "reference": "The chemical symbol for gold is Au."
            },
            {
                "prompt": "What is the longest river in the world?",
                "reference": "The Nile River is the longest river in the world."
            },
            {
                "prompt": "What organ pumps blood through the body?",
                "reference": "The heart pumps blood through the body."
            },
            {
                "prompt": "What is the value of pi to two decimal places?",
                "reference": "The value of pi to two decimal places is 3.14."
            },
            {
                "prompt": "Who discovered penicillin?",
                "reference": "Alexander Fleming discovered penicillin."
            },
            {
                "prompt": "What is the smallest prime number?",
                "reference": "The smallest prime number is 2."
            },
            {
                "prompt": "What is the capital of Australia?",
                "reference": "The capital of Australia is Canberra."
            },
            {
                "prompt": "How many days are in a leap year?",
                "reference": "There are 366 days in a leap year."
            },
            {
                "prompt": "What is the hardest natural substance on Earth?",
                "reference": "Diamond is the hardest natural substance on Earth."
            },
            {
                "prompt": "Who was the first person to walk on the moon?",
                "reference": "Neil Armstrong was the first person to walk on the moon."
            },
            {
                "prompt": "What is the powerhouse of the cell?",
                "reference": "The mitochondria is the powerhouse of the cell."
            },
            {
                "prompt": "What is 25% of 200?",
                "reference": "25% of 200 is 50."
            },
            {
                "prompt": "Who developed the theory of relativity?",
                "reference": "Albert Einstein developed the theory of relativity."
            },
            {
                "prompt": "What is the largest ocean on Earth?",
                "reference": "The Pacific Ocean is the largest ocean on Earth."
            },
            {
                "prompt": "How many sides does a hexagon have?",
                "reference": "A hexagon has six sides."
            },
            {
                "prompt": "What is the capital of Canada?",
                "reference": "The capital of Canada is Ottawa."
            },
            {
                "prompt": "Who painted 'The Starry Night'?",
                "reference": "Vincent van Gogh painted 'The Starry Night'."
            },
            {
                "prompt": "What is the atomic number of carbon?",
                "reference": "The atomic number of carbon is 6."
            },
            {
                "prompt": "What is the largest mammal in the world?",
                "reference": "The blue whale is the largest mammal in the world."
            },
            {
                "prompt": "Who wrote '1984'?",
                "reference": "George Orwell wrote '1984'."
            },
            {
                "prompt": "What is the sum of angles in a triangle?",
                "reference": "The sum of angles in a triangle is 180 degrees."
            },
            {
                "prompt": "What is the currency of the United Kingdom?",
                "reference": "The currency of the United Kingdom is the pound sterling."
            },
            {
                "prompt": "Who was the first President of the United States?",
                "reference": "George Washington was the first President of the United States."
            },
            {
                "prompt": "What is the main component of the Sun?",
                "reference": "Hydrogen is the main component of the Sun."
            },
            {
                "prompt": "How many planets are in our solar system?",
                "reference": "There are eight planets in our solar system."
            },
            {
                "prompt": "What is the Pythagorean theorem?",
                "reference": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides."
            },
            {
                "prompt": "Who composed 'The Four Seasons'?",
                "reference": "Antonio Vivaldi composed 'The Four Seasons'."
            },
            {
                "prompt": "What is the boiling point of water at sea level?",
                "reference": "The boiling point of water at sea level is 100 degrees Celsius or 212 degrees Fahrenheit."
            },
            {
                "prompt": "What is the capital of Brazil?",
                "reference": "The capital of Brazil is Brasília."
            }
        ][:self.num_samples] if hasattr(self, 'num_samples') else [
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

    def _load_perplexity_sentences(self):
        """Load perplexity test sentences (expanded)"""
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "The Earth orbits around the Sun once every 365 days.",
            "Mathematics is the study of numbers, shapes, and patterns.",
            # Additional sentences for better perplexity testing
            "Quantum mechanics describes the behavior of particles at atomic scales.",
            "The Industrial Revolution transformed manufacturing and society in the 18th century.",
            "DNA contains the genetic instructions for the development of living organisms.",
            "Climate change is affecting weather patterns across the globe.",
            "The invention of the internet revolutionized global communication.",
            "Photosynthesis converts light energy into chemical energy in plants.",
            "The Renaissance was a period of cultural rebirth in Europe.",
            "Artificial neural networks are inspired by biological neural systems.",
            "The theory of evolution explains the diversity of life on Earth.",
            "Renewable energy sources include solar, wind, and hydroelectric power.",
            "The human brain contains approximately 86 billion neurons.",
            "Shakespeare's plays have been translated into every major language.",
            "The speed of sound varies depending on the medium it travels through.",
            "Democracy is a form of government where power rests with the people.",
            "The periodic table organizes chemical elements by their properties."
        ]

        # Return subset based on num_samples or all if fewer
        max_sentences = min(len(sentences), max(20, self.num_samples // 3))
        return sentences[:max_sentences]

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

    def calculate_rouge_score(self, reference, candidate):
        """Calculate ROUGE score between reference and candidate text"""
        try:
            if not ROUGE_AVAILABLE:
                print("Warning: ROUGE scorer not available, returning 0.0")
                return 0.0

            # Initialize ROUGE scorer (ROUGE-1, ROUGE-2, ROUGE-L)
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

            # Calculate ROUGE scores
            scores = scorer.score(reference, candidate)

            # Return average F1 score across all ROUGE metrics
            avg_f1 = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3

            return avg_f1
        except Exception as e:
            print(f"Error calculating ROUGE score: {e}")
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
            'rouge_scores': [],
            'perplexity_scores': [],
            'responses': [],
            'avg_rouge': 0,
            'avg_perplexity': 0,
            'total_time': 0
        }

        start_time = time.time()

        # ROUGE Score Evaluation
        print("\n[*] Running ROUGE Score Evaluation...")
        for i, test_case in enumerate(self.test_data, 1):
            print(f"  Test {i}/{len(self.test_data)}: {test_case['prompt'][:50]}...")

            response = self.generate_response(model, test_case['prompt'])
            rouge = self.calculate_rouge_score(test_case['reference'], response)

            model_results['rouge_scores'].append(rouge)
            model_results['responses'].append({
                'prompt': test_case['prompt'],
                'reference': test_case['reference'],
                'response': response,
                'rouge': rouge
            })

            print(f"    ROUGE Score: {rouge:.4f}")

        # Perplexity Evaluation
        print("\n[*] Running Perplexity Evaluation...")
        for i, sentence in enumerate(self.perplexity_sentences, 1):
            print(f"  Sentence {i}/{len(self.perplexity_sentences)}: {sentence[:50]}...")

            perplexity = self.calculate_perplexity(model, sentence)
            model_results['perplexity_scores'].append(perplexity)

            print(f"    Perplexity: {perplexity:.2f}")

        # Calculate averages
        model_results['avg_rouge'] = np.mean(model_results['rouge_scores'])
        model_results['avg_perplexity'] = np.mean(model_results['perplexity_scores'])
        model_results['total_time'] = time.time() - start_time

        print(f"\n[OK] Evaluation complete for {model}")
        print(f"   Average ROUGE Score: {model_results['avg_rouge']:.4f}")
        print(f"   Average Perplexity: {model_results['avg_perplexity']:.2f}")
        print(f"   Total Time: {model_results['total_time']:.2f}s")

        return model_results

    def run_evaluation(self):
        """Run evaluation on all models"""
        print("[START] Starting Model Evaluation")
        print(f"Models to test: {', '.join(self.models)}")

        for model in self.models:
            try:
                results = self.evaluate_model(model)
                self.results[model] = results
            except Exception as e:
                print(f"[ERROR] Failed to evaluate {model}: {e}")

        # Save results
        self.save_results()

        # Generate visualizations
        self.create_visualizations()

        print(f"\n[OK] All evaluations complete!")
        print(f"[SAVE] Results saved to: {self.output_dir}")

    def save_results(self):
        """Save results to JSON file"""
        output_file = self.output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for model, data in self.results.items():
            serializable_results[model] = {
                'model': data['model'],
                'avg_rouge': float(data['avg_rouge']),
                'avg_perplexity': float(data['avg_perplexity']),
                'total_time': float(data['total_time']),
                'rouge_scores': [float(x) for x in data['rouge_scores']],
                'perplexity_scores': [float(x) for x in data['perplexity_scores']],
                'responses': data['responses']
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n[SAVE] Results saved to: {output_file}")

    def create_visualizations(self):
        """Create visualization graphs"""
        if not self.results:
            print("No results to visualize")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))

        # 1. Average ROUGE Score Comparison
        ax1 = plt.subplot(2, 3, 1)
        models = list(self.results.keys())
        avg_rouge = [self.results[m]['avg_rouge'] for m in models]
        colors = ['#3498db', '#e74c3c'][:len(models)]

        bars1 = ax1.bar(range(len(models)), avg_rouge, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Model', fontweight='bold')
        ax1.set_ylabel('Average ROUGE Score', fontweight='bold')
        ax1.set_title('Average ROUGE Score Comparison', fontweight='bold', fontsize=12)
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

        # 3. ROUGE Score Distribution
        ax3 = plt.subplot(2, 3, 3)
        for i, model in enumerate(models):
            rouge_scores = self.results[model]['rouge_scores']
            ax3.plot(range(1, len(rouge_scores) + 1), rouge_scores,
                    marker='o', label=model, color=colors[i], linewidth=2, markersize=6)

        ax3.set_xlabel('Test Case', fontweight='bold')
        ax3.set_ylabel('ROUGE Score', fontweight='bold')
        ax3.set_title('ROUGE Score per Test Case', fontweight='bold', fontsize=12)
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

        # 5. Combined Score (ROUGE high + Perplexity low = better)
        ax5 = plt.subplot(2, 3, 5)
        # Normalize scores for comparison (0-1 scale)
        max_perplexity = max(avg_perplexity) if max(avg_perplexity) > 0 else 1
        normalized_rouge = avg_rouge  # Already 0-1
        normalized_perplexity = [1 - (p / max_perplexity) for p in avg_perplexity]  # Invert so higher is better
        combined_score = [(r + p) / 2 for r, p in zip(normalized_rouge, normalized_perplexity)]

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

        plt.suptitle('Model Evaluation: ROUGE Score & Perplexity Analysis',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save figure
        output_file = self.output_dir / f"evaluation_graphs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[GRAPH] Graphs saved to: {output_file}")

        # Show plot
        plt.show()


def main():
    """Main execution function"""
    print("=" * 60)
    print("MODEL EVALUATION - ROUGE SCORE & PERPLEXITY")
    print("Enhanced Version with Real Dataset Support")
    print("=" * 60)

    # Configuration options:
    # dataset_name options: 'squad', 'triviaqa', 'natural_questions', 'builtin'
    # num_samples: number of samples to evaluate (recommended: 50-200)
    # use_hf_dataset: True to use Hugging Face datasets, False for built-in only

    print("\nDataset Options:")
    print("  1. 'squad' - Stanford Question Answering Dataset (100k+ Q&A pairs)")
    print("  2. 'triviaqa' - Trivia Questions Dataset (650k+ Q&A pairs)")
    print("  3. 'natural_questions' - Google Natural Questions Dataset")
    print("  4. 'builtin' - Built-in 50 general knowledge questions")
    print()

    # Example 1: Use SQuAD dataset with 50 samples
    evaluator = ModelEvaluator(
        models=['gemma3:270m', 'cogito:8b'],
        dataset_name='squad',  # Change to 'triviaqa', 'natural_questions', or 'builtin'
        num_samples=50,        # Change to desired number (10-500 recommended)
        use_hf_dataset=True    # Set to False to use only built-in data
    )

    # Example 2: Use built-in expanded dataset (50 questions)
    # evaluator = ModelEvaluator(
    #     models=['gemma3:270m', 'cogito:8b'],
    #     dataset_name='builtin',
    #     num_samples=50,
    #     use_hf_dataset=False
    # )

    # Run evaluation
    evaluator.run_evaluation()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

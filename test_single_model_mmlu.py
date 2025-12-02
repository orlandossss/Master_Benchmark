import ollama
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
import pandas as pd
import re
import json
import os
from datetime import datetime

class OllamaModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        

        response = ollama.generate(model=self.model_name, prompt=prompt)
        answer = response["response"].strip()

        # Extract only the letter choice (A, B, C, or D)
        # Look for first occurrence of A, B, C, or D
        match = re.search(r'\b[A-D]\b', answer, re.IGNORECASE)
        if match:
            return match.group(0).upper()

        # Fallback: check if answer starts with a letter
        if answer and answer[0].upper() in ['A', 'B', 'C', 'D']:
            return answer[0].upper()

        # Last resort: return the full answer (will likely be wrong)
        return answer


def run_mmlu_single_model(model_name):
    """Run MMLU benchmark for a single model"""

    # Create output directory
    output_dir = "./MMLU"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("MMLU BENCHMARK - SINGLE MODEL")
    print("="*60)
    print(f"\nTesting model: {model_name}")
    print("-"*60)

    try:
        # Initialize model
        model = OllamaModel(model_name)

        # Run benchmark
        benchmark = MMLU(
            tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.FORMAL_LOGIC, MMLUTask.GLOBAL_FACTS, MMLUTask.COLLEGE_COMPUTER_SCIENCE, MMLUTask.COLLEGE_MATHEMATICS, MMLUTask.HIGH_SCHOOL_MACROECONOMICS, MMLUTask.MARKETING],
            n_shots=3
        )

        benchmark.evaluate(model=model)

        # Print first three predictions
        if benchmark.predictions:
            print("\n📝 First 3 predictions:")
            for i, pred in enumerate(benchmark.predictions[:3], 1):
                print(f"   {i}. Predicted: {pred.get('prediction', 'N/A'):5s} | Actual: {pred.get('actual_output', 'N/A'):5s} | Correct: {pred.get('success', False)}")

        print(f"\n✅ Score: {benchmark.overall_score:.4f}")

        # Store results
        result = {
            'model_name': model_name,
            'overall_score': benchmark.overall_score,
            'task': 'HIGH_SCHOOL_COMPUTER_SCIENCE',
            'n_shots': 3,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'success',
            'predictions': benchmark.predictions
        }

        # Save results
        save_results(result, output_dir, model_name)

        return result

    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        result = {
            'model_name': model_name,
            'overall_score': 0.0,
            'task': 'HIGH_SCHOOL_COMPUTER_SCIENCE',
            'n_shots': 3,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'failed',
            'error': str(e),
            'predictions': []
        }
        save_results(result, output_dir, model_name)
        return result


def save_results(result, output_dir, model_name):
    """Save results to JSON and CSV files"""

    # Clean model name for filename (replace special characters)
    clean_model_name = model_name.replace(':', '_').replace('/', '_')

    # Save to JSON (with detailed predictions)
    json_path = os.path.join(output_dir, f'{clean_model_name}_MMLU.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {json_path}")

    # Prepare data for CSV (without detailed predictions)
    csv_row = {
        'model_name': result['model_name'],
        'overall_score': result['overall_score'],
        'task': result['task'],
        'n_shots': result['n_shots'],
        'timestamp': result['timestamp'],
        'status': result['status']
    }
    if 'error' in result:
        csv_row['error'] = result['error']

    # Save to CSV
    df = pd.DataFrame([csv_row])
    csv_path = os.path.join(output_dir, f'{clean_model_name}_MMLU.csv')
    df.to_csv(csv_path, index=False)
    print(f"💾 Results saved to: {csv_path}")


if __name__ == "__main__":
    # Configure the model to test here
    model_name = "gemma3:270m"  # Change this to test different models
    run_mmlu_single_model(model_name)

from deepeval.benchmarks import IFEval
import ollama


class OllamaModel:
    """Wrapper class for Ollama models to work with DeepEval benchmarks"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        """Generate response using Ollama"""
        self.call_count += 1

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': 500,
                    'temperature': 0.1
                }
            )
            return response["response"].strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return ""


def run_ifeval_benchmark(model_name: str = "granite4:3b-h", n_problems: int = 5):
    """Run IFEval benchmark for instruction-following evaluation"""

    print("\n" + "="*60)
    print("IFEVAL BENCHMARK - Instruction Following Evaluation")
    print("="*60)
    print(f"\nModel: {model_name}")
    print(f"Number of problems: {n_problems}")
    print("-"*60)

    # Initialize model
    model = OllamaModel(model_name)

    # Create benchmark
    benchmark = IFEval(n_problems=n_problems)

    # Evaluate
    print("\nStarting evaluation...")
    benchmark.evaluate(model=model)

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Overall Score: {benchmark.overall_score:.4f} ({benchmark.overall_score*100:.2f}%)")
    print(f"Total prompts evaluated: {model.call_count}")
    print("="*60 + "\n")

    return benchmark.overall_score


if __name__ == "__main__":
    # Run benchmark
    run_ifeval_benchmark(model_name="granite4:1b", n_problems=20)
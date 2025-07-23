#!/usr/bin/env python3
"""
Comprehensive API validation script using the full Iris dataset.

This script downloads the iris dataset and tests every sample against the running API server,
providing detailed accuracy metrics and performance validation.

Usage:
    python scripts/validation/api_comprehensive_test.py [--host HOST] [--port PORT]
"""

import argparse
import asyncio
import random
import sys
from typing import Dict, List, Tuple

import httpx

# Import after path manipulation (ruff: E402 allowed here)
from research.data import load_iris_data  # noqa: E402


async def test_api_health(client: httpx.AsyncClient, base_url: str) -> bool:
    try:
        response = await client.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API Health Check: {health_data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ API Health Check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health Check failed: {e}")
        return False


async def test_single_prediction(
    client: httpx.AsyncClient, base_url: str, sample: Dict[str, float]
) -> Tuple[bool, str]:
    try:
        response = await client.post(f"{base_url}/predict", json=sample, timeout=10.0)
        if response.status_code == 200:
            result = response.json()
            return True, result.get("prediction", "unknown")
        else:
            print(f"⚠️  API request failed: HTTP {response.status_code}")
            return False, "error"
    except Exception as e:
        print(f"⚠️  API request failed: {e}")
        return False, "error"


def prepare_api_samples() -> Tuple[List[Dict[str, float]], List[str]]:
    print("📊 Loading Iris dataset...")
    X, y, _ = load_iris_data()

    # Convert to list of dictionaries for API requests
    samples = []
    for i in range(len(X)):
        sample = {
            "sepal_length": float(X[i, 0]),
            "sepal_width": float(X[i, 1]),
            "petal_length": float(X[i, 2]),
            "petal_width": float(X[i, 3]),
        }
        samples.append(sample)

    # Convert numeric targets to species names
    species_names = ["setosa", "versicolor", "virginica"]
    true_species = [species_names[int(target)] for target in y]

    print(f"📈 Loaded {len(samples)} samples across {len(set(y))} species")
    return samples, true_species


async def run_comprehensive_test(host: str, port: int) -> int:
    """Full dataset validation with expected 96% accuracy threshold and perfect Setosa separation."""
    base_url = f"http://{host}:{port}"

    print("🧪 Starting Comprehensive API Validation")
    print("=" * 60)
    print(f"🌐 Testing API at: {base_url}")
    print(f"📋 Swagger UI: {base_url}/docs")
    print()

    # Load dataset
    samples, true_species = prepare_api_samples()

    async with httpx.AsyncClient() as client:
        # Test API health
        if not await test_api_health(client, base_url):
            print("❌ API server is not responding. Please start the server first.")
            print("💡 Try: make api-layer-isolate")
            return 1

        print()
        print("🔄 Testing all samples against API...")
        print()

        # Test all samples and collect results
        predictions = []
        failed_requests = 0
        inference_results = []  # Store full results for display

        for i, sample in enumerate(samples):
            success, prediction = await test_single_prediction(client, base_url, sample)

            if success:
                predictions.append(prediction)
                # Store inference result for potential display
                inference_results.append(
                    {
                        "index": i,
                        "sample": sample,
                        "true_label": true_species[i],
                        "prediction": prediction,
                        "correct": prediction == true_species[i],
                    }
                )
            else:
                predictions.append("error")
                failed_requests += 1

        # Display 50 random inference results
        if inference_results:
            num_to_show = min(50, len(inference_results))
            random_results = random.sample(inference_results, num_to_show)

            print(f"📊 Showing {num_to_show} random inference results:")
            print("=" * 80)

            for result in sorted(random_results, key=lambda x: x["index"]):
                correctness = "✅" if result["correct"] else "❌"
                print(
                    f"Sample #{result['index'] + 1:3d}: "
                    f"SL={result['sample']['sepal_length']:.1f}, "
                    f"SW={result['sample']['sepal_width']:.1f}, "
                    f"PL={result['sample']['petal_length']:.1f}, "
                    f"PW={result['sample']['petal_width']:.1f} | "
                    f"True: {result['true_label']:>10s} | "
                    f"Pred: {result['prediction']:>10s} {correctness}"
                )

            print("=" * 80)

        print(f"\n✅ Completed testing {len(samples)} samples")
        print()

        # Calculate metrics
        if failed_requests > 0:
            print(f"⚠️  {failed_requests} API requests failed")
            print()

        # Filter out failed requests for accuracy calculation
        valid_predictions = []
        valid_true_labels = []
        for pred, true in zip(predictions, true_species):
            if pred != "error":
                valid_predictions.append(pred)
                valid_true_labels.append(true)

        if not valid_predictions:
            print("❌ No successful predictions to analyze")
            return 1

        # Calculate overall accuracy
        correct = sum(1 for pred, true in zip(valid_predictions, valid_true_labels) if pred == true)
        total_valid = len(valid_predictions)
        overall_accuracy = correct / total_valid

        # Calculate per-species metrics
        species_stats = {
            "setosa": {"total": 0, "correct": 0},
            "versicolor": {"total": 0, "correct": 0},
            "virginica": {"total": 0, "correct": 0},
        }

        for pred, true in zip(valid_predictions, valid_true_labels):
            species_stats[true]["total"] += 1
            if pred == true:
                species_stats[true]["correct"] += 1

        # Display results
        print("📊 COMPREHENSIVE API VALIDATION RESULTS")
        print("=" * 60)
        print(f"🎯 Overall Accuracy: {overall_accuracy:.1%} ({correct}/{total_valid})")
        print()

        print("📈 Per-Species Performance:")
        for species, stats in species_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                print(f"   {species.capitalize():>12}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")

        print()

        # Validation against expected performance
        expected_accuracy = 0.96  # From research findings
        if overall_accuracy >= expected_accuracy:
            print(f"✅ API Performance: EXCELLENT (≥{expected_accuracy:.1%})")
        elif overall_accuracy >= 0.90:
            print("🟡 API Performance: GOOD (≥90%)")
        else:
            print("❌ API Performance: NEEDS IMPROVEMENT (<90%)")

        # Check setosa perfect separation
        setosa_accuracy = species_stats["setosa"]["correct"] / species_stats["setosa"]["total"]
        if setosa_accuracy == 1.0:
            print("✅ Setosa Classification: PERFECT (100% separation)")
        else:
            print(f"⚠️  Setosa Classification: {setosa_accuracy:.1%} (expected 100%)")

        print()
        print("🎉 Comprehensive validation completed!")

        if failed_requests == 0 and overall_accuracy >= expected_accuracy:
            print("🏆 All tests passed - API is production ready!")
            return 0
        else:
            print("⚠️  Some issues detected - please review results above")
            return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive API validation using the full Iris dataset")
    parser.add_argument("--host", type=str, default="localhost", help="API server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="API server port (default: 8000)")

    args = parser.parse_args()

    try:
        exit_code = asyncio.run(run_comprehensive_test(args.host, args.port))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

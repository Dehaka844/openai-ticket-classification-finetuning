
import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import SYSTEM_PROMPT


def classify_ticket(
    client: OpenAI,
    model: str,
    ticket: str,
    expected_category: str,
) -> dict[str, Any]:
    """Clasifica un ticket y registra el uso de tokens y la latencia."""

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ticket},
        ],
        temperature=0,
    )

    latency = time.perf_counter() - start_time
    prediction = (response.choices[0].message.content or "").strip().lower()
    usage = response.usage

    return {
        "ticket": ticket,
        "expected_category": expected_category,
        "predicted_category": prediction,
        "correct": prediction == expected_category,
        "latency_seconds": round(latency, 4),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcula métricas agregadas de una evaluación."""

    total = len(results)
    correct = sum(result["correct"] for result in results)

    latencies = [
        result["latency_seconds"]
        for result in results
        if result["latency_seconds"] is not None
    ]
    token_counts = [
        result["total_tokens"]
        for result in results
        if result["total_tokens"] is not None
    ]

    return {
        "total_tickets": total,
        "correct_predictions": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "mean_latency_seconds": (
            round(sum(latencies) / len(latencies), 4)
            if latencies else None
        ),
        "mean_total_tokens": (
            round(sum(token_counts) / len(token_counts), 2)
            if token_counts else None
        ),
    }


def evaluate_model(
    client: OpenAI,
    model: str,
    tickets: list[dict[str, str]],
) -> dict[str, Any]:
    """Evalúa un modelo sobre el mismo conjunto de tickets."""

    results = [
        classify_ticket(
            client=client,
            model=model,
            ticket=item["ticket"],
            expected_category=item["expected_category"],
        )
        for item in tickets
    ]

    return {
        "model": model,
        "summary": summarize_results(results),
        "predictions": results,
    }


def save_evaluation(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Guarda los resultados de la comparación en formato JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

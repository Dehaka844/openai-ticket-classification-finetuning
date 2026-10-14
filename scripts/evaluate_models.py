
import json

from src.client import get_client
from src.config import BASE_MODEL, RESULTS_DIR, CATEGORIES
from src.evaluation import evaluate_model, save_evaluation

JOB_INFO_FILE = RESULTS_DIR / "training_job.json"
EVALUATION_FILE = RESULTS_DIR / "evaluation_results.json"

# Cinco tickets de prueba distintos de los ejemplos de entrenamiento.
TEST_TICKETS = [
    {
        "ticket": "Me han cobrado dos veces el mismo recibo este mes.",
        "expected_category": "facturacion",
    },
    {
        "ticket": "La aplicación se cierra cada vez que intento abrir un informe.",
        "expected_category": "tecnico",
    },
    {
        "ticket": "He olvidado mi contraseña y no consigo acceder a mi perfil.",
        "expected_category": "cuenta",
    },
    {
        "ticket": "¿Puedo utilizar la aplicación desde varios dispositivos?",
        "expected_category": "consulta",
    },
    {
        "ticket": "Quiero dar de baja mi suscripción antes de la próxima renovación.",
        "expected_category": "cancelacion",
    },
]


def main() -> None:
    if not JOB_INFO_FILE.is_file():
        raise FileNotFoundError(
            "No existe results/training_job.json. "
            "Se necesita el registro del trabajo para obtener el modelo ajustado."
        )

    with JOB_INFO_FILE.open("r", encoding="utf-8") as file:
        job_info = json.load(file)

    fine_tuned_model = job_info.get("fine_tuned_model")

    if not fine_tuned_model:
        raise ValueError(
            "No hay un identificador de modelo ajustado en el registro. "
            "No se puede realizar una comparación real."
        )

    for item in TEST_TICKETS:
        if item["expected_category"] not in CATEGORIES:
            raise ValueError(
                f"Categoría no válida: {item['expected_category']}"
            )

    client = get_client()

    print(f"Evaluando modelo base: {BASE_MODEL}")
    base_results = evaluate_model(client, BASE_MODEL, TEST_TICKETS)

    print(f"Evaluando modelo ajustado: {fine_tuned_model}")
    fine_tuned_results = evaluate_model(
        client, fine_tuned_model, TEST_TICKETS
    )

    comparison = {
        "base_model": base_results,
        "fine_tuned_model": fine_tuned_results,
        "note": (
            "Resultados medidos mediante llamadas reales a la API. "
            "La latencia puede variar entre ejecuciones."
        ),
    }

    save_evaluation(comparison, EVALUATION_FILE)

    print(f"Resultados guardados en: {EVALUATION_FILE}")
    print("\nResumen del modelo base:")
    print(json.dumps(base_results["summary"], indent=4, ensure_ascii=False))
    print("\nResumen del modelo ajustado:")
    print(json.dumps(
        fine_tuned_results["summary"], indent=4, ensure_ascii=False
    ))


if __name__ == "__main__":
    main()

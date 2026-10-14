
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import BASE_MODEL, MODEL_SUFFIX, N_EPOCHS, TRAINING_FILE


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def upload_training_file(
    client: OpenAI,
    file_path: Path = TRAINING_FILE,
) -> str:
    """
    Sube el dataset a OpenAI y devuelve el identificador del archivo.
    """
    if not file_path.is_file():
        raise FileNotFoundError(
            f"No se encuentra el archivo de entrenamiento: {file_path}"
        )

    print(f"Subiendo dataset: {file_path.name}")

    with file_path.open("rb") as file:
        uploaded_file = client.files.create(
            file=file,
            purpose="fine-tune",
        )

    print(f"Archivo subido correctamente. ID: {uploaded_file.id}")
    return uploaded_file.id


def create_fine_tuning_job(
    client: OpenAI,
    file_id: str,
) -> str:
    """
    Crea un trabajo de fine-tuning y devuelve su identificador.
    """
    if not file_id:
        raise ValueError("El identificador del archivo no puede estar vacío.")

    print(f"Creando trabajo con el modelo base {BASE_MODEL}...")

    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=BASE_MODEL,
        suffix=MODEL_SUFFIX,
        hyperparameters={
            "n_epochs": N_EPOCHS,
        },
    )

    print(f"Trabajo creado correctamente. ID: {job.id}")
    print(f"Estado inicial: {job.status}")

    return job.id


def get_job_status(client: OpenAI, job_id: str) -> Any:
    """
    Recupera el estado actual de un trabajo existente.
    """
    if not job_id:
        raise ValueError("El identificador del trabajo no puede estar vacío.")

    return client.fine_tuning.jobs.retrieve(job_id)


def monitor_fine_tuning_job(
    client: OpenAI,
    job_id: str,
    poll_interval: int = 30,
    timeout_seconds: int = 3600,
) -> Any:
    """
    Consulta periódicamente el trabajo hasta que termine o venza el timeout.

    El timeout solo detiene la espera local; no cancela el entrenamiento.
    """
    if poll_interval <= 0:
        raise ValueError("poll_interval debe ser mayor que cero.")

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds debe ser mayor que cero.")

    start_time = time.monotonic()

    while True:
        job = get_job_status(client, job_id)
        status = job.status

        print(f"Estado del trabajo: {status}")

        if status in TERMINAL_STATUSES:
            if status == "succeeded":
                print("Entrenamiento completado correctamente.")
                print(f"Modelo entrenado: {job.fine_tuned_model}")
            else:
                print(f"El trabajo terminó con estado: {status}")

            return job

        elapsed = time.monotonic() - start_time
        remaining = timeout_seconds - elapsed

        if remaining <= 0:
            raise TimeoutError(
                f"Se agotó el tiempo de espera del trabajo {job_id}. "
                "El entrenamiento puede seguir ejecutándose en OpenAI."
            )

        time.sleep(min(poll_interval, remaining))

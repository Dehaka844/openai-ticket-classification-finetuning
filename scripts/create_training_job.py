
import argparse
import json

from src.client import get_client
from src.config import BASE_MODEL, N_EPOCHS, RESULTS_DIR
from src.training import create_fine_tuning_job


JOB_INFO_FILE = RESULTS_DIR / "training_job.json"


def main():
    parser = argparse.ArgumentParser(
        description="Crear un trabajo de fine-tuning en OpenAI."
    )
    parser.add_argument(
        "--file-id",
        required=True,
        help="Identificador del archivo de entrenamiento subido a OpenAI.",
    )
    args = parser.parse_args()

    # Evita crear otro trabajo si ya tenemos uno registrado.
    if JOB_INFO_FILE.exists():
        raise RuntimeError(
            f"Ya existe el registro {JOB_INFO_FILE}. "
            "Comprueba el trabajo existente antes de crear otro."
        )

    client = get_client()

    # Esta llamada inicia el entrenamiento en OpenAI.
    job_id = create_fine_tuning_job(client, args.file_id)

    # Guardamos los identificadores para poder retomar el proceso.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    job_info = {
        "training_file_id": args.file_id,
        "job_id": job_id,
        "base_model": BASE_MODEL,
        "n_epochs": N_EPOCHS,
    }

    with JOB_INFO_FILE.open("w", encoding="utf-8") as file:
        json.dump(job_info, file, indent=4, ensure_ascii=False)

    print(f"\nRegistro guardado en: {JOB_INFO_FILE}")
    print(f"ID del trabajo: {job_id}")


if __name__ == "__main__":
    main()

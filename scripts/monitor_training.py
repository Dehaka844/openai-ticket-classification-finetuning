
import json

from src.client import get_client
from src.config import RESULTS_DIR
from src.training import monitor_fine_tuning_job

JOB_INFO_FILE = RESULTS_DIR / "training_job.json"


def main() -> None:
    if not JOB_INFO_FILE.is_file():
        raise FileNotFoundError(
            f"No existe el registro del entrenamiento: {JOB_INFO_FILE}"
        )

    with JOB_INFO_FILE.open("r", encoding="utf-8") as file:
        job_info = json.load(file)

    job_id = job_info.get("job_id")

    if not job_id:
        raise ValueError(
            "El registro no contiene un identificador de trabajo."
        )

    client = get_client()

    job = monitor_fine_tuning_job(
        client=client,
        job_id=job_id,
    )

    job_info["status"] = job.status
    job_info["fine_tuned_model"] = job.fine_tuned_model

    with JOB_INFO_FILE.open("w", encoding="utf-8") as file:
        json.dump(job_info, file, indent=4, ensure_ascii=False)

    print(f"Estado final: {job.status}")

    if job.fine_tuned_model:
        print(f"Modelo ajustado: {job.fine_tuned_model}")


if __name__ == "__main__":
    main()

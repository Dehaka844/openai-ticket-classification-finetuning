
from src.client import get_client
from src.training import upload_training_file
from src.validation import validate_jsonl


def main():
    total = validate_jsonl()
    print(f"Dataset validado: {total} ejemplos.")

    client = get_client()
    file_id = upload_training_file(client)

    print("\nGuarda este identificador para el siguiente paso:")
    print(f"TRAINING_FILE_ID={file_id}")


if __name__ == "__main__":
    main()

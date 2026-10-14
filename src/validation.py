
import json

from src.config import (
    CATEGORIES,
    MIN_TRAINING_EXAMPLES,
    SYSTEM_PROMPT,
    TRAINING_FILE,
)


def validate_jsonl() -> int:
    """
    Valida el archivo JSONL completo.

    Returns:
        Número de ejemplos válidos.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el contenido o la estructura son incorrectos.
    """
    if not TRAINING_FILE.is_file():
        raise FileNotFoundError(
            f"No se encuentra el dataset: {TRAINING_FILE}. "
            "Genera primero el archivo con src.dataset."
        )

    total = 0

    with TRAINING_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(
                    f"La línea {line_number} está vacía."
                )

            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON inválido en la línea {line_number}: {error}"
                ) from error

            if not isinstance(example, dict):
                raise ValueError(
                    f"La línea {line_number} debe contener un objeto JSON."
                )

            if set(example.keys()) != {"messages"}:
                raise ValueError(
                    f"Claves incorrectas en la línea {line_number}."
                )

            messages = example["messages"]

            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError(
                    f"Se esperaban tres mensajes en la línea {line_number}."
                )

            expected_roles = ["system", "user", "assistant"]

            for message, expected_role in zip(messages, expected_roles):
                if not isinstance(message, dict):
                    raise ValueError(
                        f"Mensaje inválido en la línea {line_number}."
                    )

                if set(message.keys()) != {"role", "content"}:
                    raise ValueError(
                        f"Estructura incorrecta en la línea {line_number}."
                    )

                if message["role"] != expected_role:
                    raise ValueError(
                        f"Rol incorrecto en la línea {line_number}."
                    )

                if (
                    not isinstance(message["content"], str)
                    or not message["content"].strip()
                ):
                    raise ValueError(
                        f"Contenido vacío en la línea {line_number}."
                    )

            if messages[0]["content"] != SYSTEM_PROMPT:
                raise ValueError(
                    f"System prompt incorrecto en la línea {line_number}."
                )

            if messages[2]["content"] not in CATEGORIES:
                raise ValueError(
                    f"Categoría inválida en la línea {line_number}."
                )

            total += 1

    if total < MIN_TRAINING_EXAMPLES:
        raise ValueError(
            f"El archivo contiene {total} ejemplos; "
            f"se requieren al menos {MIN_TRAINING_EXAMPLES}."
        )

    return total


if __name__ == "__main__":
    total = validate_jsonl()
    print(f"Validación completada: {total} ejemplos válidos.")

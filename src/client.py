
from openai import OpenAI

from src.config import OPENAI_API_KEY


def get_client() -> OpenAI:
    """
    Crea un cliente autenticado de OpenAI.

    Raises:
        RuntimeError: si no se ha configurado la API key.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "No se ha encontrado OPENAI_API_KEY. "
            "Comprueba que existe el archivo .env en la raíz del proyecto."
        )

    return OpenAI(api_key=OPENAI_API_KEY)

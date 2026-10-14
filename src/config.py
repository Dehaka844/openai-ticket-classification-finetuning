
import os
from pathlib import Path

from dotenv import load_dotenv


# Raíz del proyecto: carpeta que contiene src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cargar las variables del archivo .env situado en la raíz.
load_dotenv(PROJECT_ROOT / ".env")

# API de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Modelo y configuración de fine-tuning
BASE_MODEL = "gpt-4o-mini-2024-07-18"
MODEL_SUFFIX = "clasificador-tickets"
N_EPOCHS = 3

# Rutas del proyecto
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAINING_FILE = DATA_DIR / "tickets_clasificacion.jsonl"
EVALUATION_FILE = RESULTS_DIR / "evaluation_results.json"

# Categorías permitidas
CATEGORIES = (
    "facturacion",
    "tecnico",
    "cuenta",
    "consulta",
    "cancelacion",
)

# Número mínimo de ejemplos exigido por el ejercicio.
MIN_TRAINING_EXAMPLES = 50

# Prompt compartido por los ejemplos y la inferencia.
SYSTEM_PROMPT = (
    "Clasifica cada ticket en una de estas categorías: "
    "facturacion, tecnico, cuenta, consulta o cancelacion. "
    "Responde únicamente con el nombre exacto de la categoría."
)

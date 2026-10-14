# Fine-Tuning de OpenAI para la Clasificación de Tickets

Proyecto desarrollado como ejercicio del máster de AI Engineer para preparar un modelo de clasificación de tickets de soporte mediante fine-tuning de la API de OpenAI.

El objetivo es clasificar automáticamente las solicitudes de los usuarios en cinco categorías, preparar el conjunto de datos de entrenamiento y comparar el comportamiento del modelo base con el modelo ajustado.

> **Estado del proyecto:** dataset generado y validado, archivo subido a OpenAI y módulos de entrenamiento y evaluación preparados. La creación del trabajo de fine-tuning no pudo completarse debido a una restricción de disponibilidad de OpenAI.

## Objetivos

- Crear un dataset JSONL con al menos 50 ejemplos etiquetados.
- Validar la estructura y las categorías de los datos.
- Preparar la subida del dataset y la creación de un trabajo de fine-tuning.
- Implementar el seguimiento del entrenamiento.
- Preparar una evaluación con cinco tickets no utilizados en el entrenamiento.
- Comparar precisión, latencia y consumo de tokens entre el modelo base y el ajustado.

## Categorías

| Categoría | Descripción |
|---|---|
| `facturacion` | Pagos, recibos, facturas y suscripciones relacionadas con facturación. |
| `tecnico` | Errores, fallos y problemas de funcionamiento. |
| `cuenta` | Acceso, contraseñas y gestión del perfil. |
| `consulta` | Preguntas generales sobre el uso de la aplicación. |
| `cancelacion` | Solicitudes de baja o cancelación del servicio. |

## Tecnologías utilizadas

- Python
- OpenAI Python SDK
- API de OpenAI
- `python-dotenv`
- JSON y JSONL

Modelo base configurado: `gpt-4o-mini-2024-07-18`.

## Estructura del proyecto

```text
fine-tuning-openai/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── validation.py
│   ├── client.py
│   ├── training.py
│   └── evaluation.py
├── scripts/
│   ├── __init__.py
│   ├── upload_dataset.py
│   ├── create_training_job.py
│   ├── monitor_training.py
│   └── evaluate_models.py
├── data/
│   └── tickets_clasificacion.jsonl
├── results/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

### Responsabilidad de los módulos

- `config.py`: configuración del modelo, categorías, rutas y prompt de clasificación.
- `dataset.py`: generación del dataset de entrenamiento.
- `validation.py`: validación del archivo JSONL y sus ejemplos.
- `client.py`: inicialización del cliente de OpenAI.
- `training.py`: subida del dataset, creación del trabajo y seguimiento de su estado.
- `upload_dataset.py`: validación y subida del archivo de entrenamiento.
- `create_training_job.py`: creación y registro del trabajo de fine-tuning.
- `monitor_training.py`: consulta del estado y registro del modelo ajustado.
- `evaluation.py`: clasificación de tickets y cálculo de métricas.
- `evaluate_models.py`: preparación de la comparación entre el modelo base y el ajustado.

## Instalación

Se recomienda Python 3.11 o una versión compatible con las dependencias instaladas.

Crear y activar un entorno virtual:

```bash
python -m venv .venv
```

En Windows:

```powershell
.venv\Scripts\activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Configuración

Crear un archivo `.env` en la raíz del proyecto:

```env
OPENAI_API_KEY=tu_clave_de_api
```

Es necesario disponer de una clave de API válida y de los permisos necesarios para utilizar los servicios de OpenAI.

**Nunca se debe subir el archivo `.env` a GitHub.**

## Generación y validación del dataset

Generar los ejemplos de entrenamiento:

```bash
python -m src.dataset
```

Validar el archivo generado:

```bash
python -m src.validation
```

El dataset contiene 50 ejemplos, distribuidos entre las cinco categorías. Cada ejemplo sigue el formato de mensajes utilizado por la API de OpenAI para tareas de fine-tuning.

La validación comprueba la estructura de los ejemplos y que las categorías sean válidas.

## Subida del dataset

Para validar y subir el archivo a OpenAI:

```bash
python -m scripts.upload_dataset
```

Este paso se completó correctamente. La API devolvió un identificador para el archivo subido.

## Entrenamiento

El proyecto incluye un módulo para crear un trabajo de fine-tuning con el modelo base configurado.

El comando previsto es:

```bash
python -m scripts.create_training_job --file-id "ID_DEL_ARCHIVO"
```

El identificador debe corresponder al archivo de entrenamiento subido previamente.

Si el trabajo se crea correctamente, se guarda información en `results/training_job.json`. El módulo de seguimiento permite consultar su estado y obtener el identificador del modelo ajustado cuando el entrenamiento termina.

### Incidencia encontrada

Al ejecutar la creación del trabajo, OpenAI devolvió el error `403 PermissionDeniedError`, con el código `training_not_available`.

El mensaje indicaba que la organización utilizada ya no tenía permiso para crear nuevos trabajos de entrenamiento debido a la retirada progresiva de la plataforma de fine-tuning.

Por este motivo, el entrenamiento no llegó a iniciarse y no se obtuvo un identificador de modelo ajustado.

## Evaluación de modelos

El proyecto incluye una evaluación preparada para cinco tickets independientes del dataset de entrenamiento.

Las métricas previstas son:

- **Exactitud (accuracy):** proporción de tickets clasificados correctamente.
- **Latencia:** tiempo empleado en obtener cada respuesta.
- **Tokens de entrada:** tokens utilizados en el prompt y el ticket.
- **Tokens de salida:** tokens generados por el modelo.
- **Tokens totales:** suma de los tokens de entrada y salida.

El script `scripts/evaluate_models.py` utiliza el mismo conjunto de prueba para ambos modelos y guarda los resultados en `results/evaluation_results.json`.

Esta evaluación requiere que exista un modelo ajustado accesible. Como el entrenamiento no se completó, la comparación real está pendiente y no se dispone de métricas experimentales para el modelo ajustado.

## Resultados y estado final

| Tarea | Estado |
|---|---|
| Generación de 50 ejemplos | Completada |
| Validación del dataset | Completada |
| Inicialización del cliente | Completada |
| Subida del dataset a OpenAI | Completada |
| Creación del trabajo de fine-tuning | Bloqueada por permisos de OpenAI |
| Obtención del modelo ajustado | Pendiente |
| Evaluación real de cinco tickets | Pendiente |
| Comparación experimental de modelos | Pendiente |

## Posibles mejoras futuras

- Repetir el entrenamiento cuando se disponga de acceso a una organización habilitada.
- Ampliar el dataset con ejemplos más variados y casos ambiguos.
- Evaluar el modelo con un conjunto de prueba más amplio.
- Analizar los errores de clasificación y las categorías que se confunden con mayor frecuencia.
- Comparar el fine-tuning con una estrategia basada en prompts sin entrenamiento adicional.

## Conclusión

El ejercicio permitió preparar un flujo modular para la clasificación de tickets, generar y validar los datos de entrenamiento y desarrollar el código necesario para gestionar un trabajo de fine-tuning y evaluar sus resultados.

La limitación de disponibilidad de OpenAI impidió completar la fase experimental. Por tanto, el proyecto no permite concluir que el modelo ajustado mejore al modelo base, ya que esa comparación no llegó a realizarse.

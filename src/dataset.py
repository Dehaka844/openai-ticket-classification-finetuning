
import json

from src.config import (
    CATEGORIES,
    MIN_TRAINING_EXAMPLES,
    SYSTEM_PROMPT,
    TRAINING_FILE,
)


# Dataset equilibrado: 10 ejemplos por categoría.
TICKETS = [
    # Facturación
    ("Me han cobrado dos veces por la misma suscripción", "facturacion"),
    ("Necesito descargar la factura del mes pasado", "facturacion"),
    ("Mi tarjeta ha sido rechazada al realizar el pago", "facturacion"),
    ("El importe de mi factura es incorrecto", "facturacion"),
    ("Quiero actualizar el método de pago", "facturacion"),
    ("Aparece un cargo pendiente desde hace varios días", "facturacion"),
    ("Necesito una factura con los datos fiscales de mi empresa", "facturacion"),
    ("Se ha renovado mi suscripción y no reconozco el cobro", "facturacion"),
    ("¿Cuándo recibiré el reembolso de mi último pago?", "facturacion"),
    ("No puedo completar el pago con mi tarjeta bancaria", "facturacion"),

    # Técnico
    ("La aplicación se cierra al exportar un archivo PDF", "tecnico"),
    ("El botón de guardar no funciona en la versión móvil", "tecnico"),
    ("Recibo un error 500 al subir documentos", "tecnico"),
    ("La página se queda cargando indefinidamente", "tecnico"),
    ("No se genera el informe mensual correctamente", "tecnico"),
    ("La aplicación muestra una pantalla en blanco al abrirla", "tecnico"),
    ("Desde la última actualización no funciona la búsqueda", "tecnico"),
    ("No puedo descargar archivos porque aparece un error", "tecnico"),
    ("El sistema responde muy lentamente al abrir los informes", "tecnico"),
    ("Al guardar los cambios aparece un mensaje de error", "tecnico"),

    # Cuenta
    ("No puedo iniciar sesión con mi contraseña habitual", "cuenta"),
    ("He olvidado mi contraseña y necesito recuperarla", "cuenta"),
    ("No recibo el correo de verificación de mi cuenta", "cuenta"),
    ("Quiero cambiar el correo electrónico asociado a mi perfil", "cuenta"),
    ("Mi cuenta aparece bloqueada después de varios intentos", "cuenta"),
    ("No recuerdo qué nombre de usuario utilicé al registrarme", "cuenta"),
    ("¿Cómo puedo activar la autenticación en dos pasos?", "cuenta"),
    ("Quiero modificar mi nombre y mis datos personales", "cuenta"),
    ("No consigo verificar mi identidad para entrar", "cuenta"),
    ("Necesito restablecer el acceso a mi perfil", "cuenta"),

    # Consulta
    ("¿Cómo puedo cambiar mi foto de perfil?", "consulta"),
    ("¿Qué diferencias hay entre el plan básico y el premium?", "consulta"),
    ("¿Se puede integrar la aplicación con Google Calendar?", "consulta"),
    ("¿Cómo exporto mis datos a un archivo Excel?", "consulta"),
    ("¿Qué formatos de archivo admite la plataforma?", "consulta"),
    ("¿Puedo compartir un informe con otros usuarios?", "consulta"),
    ("¿Dónde puedo consultar la documentación del producto?", "consulta"),
    ("¿Cómo se configura una notificación automática?", "consulta"),
    ("¿Existe una versión de la aplicación para móviles?", "consulta"),
    ("¿Cuántos usuarios puedo añadir a mi espacio de trabajo?", "consulta"),

    # Cancelación
    ("Quiero cancelar mi suscripción antes de la renovación", "cancelacion"),
    ("Necesito dar de baja mi cuenta definitivamente", "cancelacion"),
    ("Ya no necesito el servicio y quiero dejar de pagarlo", "cancelacion"),
    ("¿Cómo puedo eliminar permanentemente mi cuenta?", "cancelacion"),
    ("Quiero cancelar el plan premium y evitar futuros cobros", "cancelacion"),
    ("Solicito la baja de mi suscripción empresarial", "cancelacion"),
    ("Deseo dejar de utilizar la plataforma y cerrar mi cuenta", "cancelacion"),
    ("¿Qué pasos debo seguir para darme de baja?", "cancelacion"),
    ("No quiero renovar mi suscripción el próximo mes", "cancelacion"),
    ("Quiero eliminar mi cuenta y todos mis datos asociados", "cancelacion"),
]


def create_training_example(ticket: str, category: str) -> dict:
    """Convierte un ticket etiquetado al formato conversacional de OpenAI."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ticket},
            {"role": "assistant", "content": category},
        ]
    }


def validate_dataset() -> None:
    """Valida los ejemplos antes de escribir el archivo."""
    if len(TICKETS) < MIN_TRAINING_EXAMPLES:
        raise ValueError(
            f"Se requieren al menos {MIN_TRAINING_EXAMPLES} ejemplos. "
            f"Se han encontrado {len(TICKETS)}."
        )

    for index, (ticket, category) in enumerate(TICKETS, start=1):
        if not isinstance(ticket, str) or not ticket.strip():
            raise ValueError(f"El ticket {index} está vacío o no es válido.")

        if category not in CATEGORIES:
            raise ValueError(
                f"Categoría inválida en el ticket {index}: {category}"
            )


def generate_jsonl() -> None:
    """Genera el archivo JSONL de entrenamiento."""
    validate_dataset()

    TRAINING_FILE.parent.mkdir(parents=True, exist_ok=True)

    with TRAINING_FILE.open("w", encoding="utf-8") as file:
        for ticket, category in TICKETS:
            example = create_training_example(ticket, category)
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Dataset generado: {TRAINING_FILE}")
    print(f"Total de ejemplos: {len(TICKETS)}")


if __name__ == "__main__":
    generate_jsonl()

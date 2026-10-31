
# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

import re
from pathlib import Path

origen = ROOT / "versiones/v1/entrenar.py"
destino = ROOT / "versiones/v2/entrenar_v2.py"

if destino.exists():
    raise RuntimeError(
        "entrenar_v2.py ya existe. No se sobrescribió."
    )

codigo = origen.read_text(encoding="utf-8-sig")


def reemplazar(patron, nuevo):
    global codigo

    codigo, cantidad = re.subn(patron, nuevo, codigo)

    if cantidad != 1:
        raise RuntimeError(
            f"Se esperaba una coincidencia para: {patron}. "
            f"Se encontraron: {cantidad}. No se creó el archivo."
        )


reemplazar(
    r'outputs/roberta_squad2',
    'outputs/roberta_squad2_v2',
)
reemplazar(
    r'logs/roberta_squad2',
    'logs/roberta_squad2_v2',
)
reemplazar(
    r'num_train_epochs\s*=\s*3\s*,',
    'num_train_epochs=2,',
)
reemplazar(
    r'gradient_accumulation_steps\s*=\s*4\s*,',
    'gradient_accumulation_steps=8,',
)
reemplazar(
    r'learning_rate\s*=\s*2e-5\s*,',
    'learning_rate=1e-5,',
)
reemplazar(
    r'warmup_steps\s*=\s*4944\s*,',
    'warmup_steps=824,',
)
reemplazar(
    r'processing_class\s*=\s*tokenizer\s*,',
    (
        'processing_class=tokenizer,\n'
        '        callbacks=[EarlyStoppingCallback(\n'
        '            early_stopping_patience=6\n'
        '        )],'
    ),
)

codigo = (
    "from transformers import EarlyStoppingCallback\n"
    + codigo
)

#Comprobamos la sintaxis antes de escribir el archivo.
compile(codigo, str(destino), "exec")
destino.write_text(codigo, encoding="utf-8")

print("Creado:", destino)
print("La primera versión permanece intacta.")
print("Salida nueva: outputs/roberta_squad2_v2")
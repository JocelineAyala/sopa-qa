"""Comprobación sin entrenamientos, descargas ni servidor HTTP."""
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
report = {}
files = list((ROOT / 'versiones').rglob('*.py')) + list(ROOT.glob('*.py'))
dependencies = set()
local = set()
for p in files:
    code = p.read_text(encoding='utf-8-sig')
    compile(code, str(p), 'exec')
    for node in ast.walk(ast.parse(code)):
        names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
        for name in names:
            base = name.split('.')[0]
            if (ROOT / (base + '.py')).exists() or (ROOT / base / '__init__.py').exists():
                assert importlib.util.find_spec(name), name
                local.add(name)
            elif base not in sys.stdlib_module_names:
                assert importlib.util.find_spec(base), base
                dependencies.add(base)
report['syntax_files'] = len(files)
report['local_imports_resolved'] = sorted(local)
report['external_imports_available'] = sorted(dependencies)
packages = ['torch', 'transformers', 'datasets', 'accelerate', 'numpy', 'tqdm', 'matplotlib', 'safetensors', 'peft', 'fastapi', 'pydantic', 'uvicorn']
(ROOT / 'verificaciones/dependencias.json').write_text(json.dumps({p: importlib.metadata.version(p) for p in packages}, indent=2), encoding='utf-8')

# Solo módulos sin ejecución de pruebas, descargas o entrenamiento al importarse.
os.chdir(ROOT / 'versiones' / 'v6')
for name in ['qa_v3', 'versiones.v1.evaluar', 'versiones.v1.app_historica', 'versiones.v2.evaluar_v2',
             'versiones.v3.entrenar_v3', 'versiones.v3.evaluar_v3',
             'versiones.v4_lora.qa_v4', 'versiones.v4_lora.entrenar_v4',
             'versiones.v4_lora.evaluar_v4', 'versiones.v4_lora.finalizar_v4',
             'versiones.v5.entrenar_v5_rapido', 'versiones.v6.entrenar_v6', 'app']:
    importlib.import_module(name)
report['imports_from_other_cwd'] = True
historical = importlib.import_module('versiones.v1.app_historica')
assert historical.MODELO == ROOT / 'outputs/roberta_squad2/modelo'
assert historical.CONFIGURACION == ROOT / 'logs/roberta_squad2/umbral.json'
import app
assert app.MODELO == ROOT / 'outputs/roberta_squad2_v2/modelo'
assert app.CONFIGURACION == ROOT / 'logs/roberta_squad2_v2/umbral.json'
resources = app.cargar()
report['api'] = {'model': str(app.MODELO.relative_to(ROOT)), 'configuration': str(app.CONFIGURACION.relative_to(ROOT)),
                 'threshold': resources['threshold'], 'params': resources['params'], 'warmup': 'ok', 'device': str(resources['device'])}
print('API cargada en CPU y calentamiento correcto.', flush=True)
before = json.loads((ROOT / 'verificaciones/archivos_protegidos.json').read_text(encoding='utf-8'))
changed = []
for name, (size, timestamp, checksum) in before.items():
    p = ROOT / name
    if not p.exists() or (p.stat().st_size, p.stat().st_mtime_ns) != (size, timestamp):
        changed.append(name)
    elif checksum and hashlib.sha256(p.read_bytes()).hexdigest() != checksum:
        changed.append(name)
report['protected_files_checked'] = len(before)
report['protected_files_changed'] = changed
assert not changed, changed
(ROOT / 'verificaciones/resultado.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)

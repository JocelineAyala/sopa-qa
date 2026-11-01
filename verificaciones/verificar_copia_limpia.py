"""Reproduce a CPU-only serving check without reading existing model weights.

Run with system Python: --phase prepare, install, download, test (in order).
Only the temporary server PID created by this script is stopped.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'verificaciones/copia_limpia.json'
PACKAGES = ['torch', 'transformers', 'safetensors', 'numpy', 'datasets', 'tqdm', 'fastapi', 'pydantic>=2', 'uvicorn']
ASSET = 'https://github.com/JocelineAyala/sopa-qa/releases/download/modelo-v2/sopa-qa-v2-modelo.zip'

def save(record):
    STATE.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['prepare', 'install', 'download', 'test'])
    phase = parser.parse_args().phase
    if phase == 'prepare':
        location = Path(tempfile.mkdtemp(prefix='sopa-qa-clean-')).resolve()
        assert not location.is_relative_to(ROOT)
        listed = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
        copied = []
        for name in sorted(set(filter(None, listed)) | {'REPORT.md'}):
            source = ROOT / name
            if not source.is_file() or Path(name).parts[0] in {'.venv', 'data', 'models', 'outputs', 'tools'}:
                continue
            if subprocess.run(['git', 'check-ignore', '--no-index', '-q', '--', name], cwd=ROOT).returncode == 0:
                continue
            destination = location / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append({'path': name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        record = dict(temporary_copy=str(location), source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), copied_files=copied,
            python=sys.version, phases={'prepare': 'ok'}, note='Tracked working-tree files plus updated REPORT.md; no original weights or environment copied.')
        save(record)
        subprocess.run([sys.executable, '-m', 'venv', str(location / '.venv')], check=True)
        record['phases']['venv'] = 'ok'
        save(record)
        print('Prepared isolated copy:', location, flush=True)
        return
    record = json.loads(STATE.read_text(encoding='utf-8'))
    location = Path(record['temporary_copy'])
    python = location / '.venv/Scripts/python.exe'
    try:
        if phase == 'install':
            command = [str(python), '-m', 'pip', 'install', '--disable-pip-version-check', '--index-url', 'https://pypi.org/simple', *PACKAGES]
            record['install_command'] = command
            with (location / 'install.log').open('w', encoding='utf-8') as output:
                result = subprocess.run(command, cwd=location, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                print((location / 'install.log').read_text(encoding='utf-8', errors='replace')[-5000:])
                raise RuntimeError('Dependency installation failed; see temporary install.log')
            record['installed_packages'] = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], cwd=location, text=True).splitlines()
        elif phase == 'download':
            archive = location / 'sopa-qa-v2-modelo.zip'
            urllib.request.urlretrieve(ASSET, archive)
            record['asset'] = dict(url=ASSET, bytes=archive.stat().st_size, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            extraction = location / 'outputs/roberta_squad2_v2'
            extraction.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
                for name in names:
                    assert (extraction / name).resolve().is_relative_to(extraction), 'Unsafe ZIP path'
                record['zip_members'] = names
                z.extractall(extraction)
            model = location / 'outputs/roberta_squad2_v2/modelo/config.json'
            record['documented_extraction_path_exists'] = model.is_file()
            if not model.is_file():
                candidates = list(location.rglob('config.json'))
                candidates = [p for p in candidates if '.venv' not in p.parts]
                if len(candidates) != 1:
                    raise RuntimeError('ZIP layout differs and model directory is ambiguous')
                destination = model.parent
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(candidates[0].parent), str(destination))
                record['zip_layout_correction'] = 'Move extracted model directory to outputs/roberta_squad2_v2/modelo'
            assert model.is_file()
        elif phase == 'test':
            record['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
            record['pip_check'] = subprocess.check_output([str(python), '-m', 'pip', 'check'], cwd=location, text=True).strip()
            config = (location / '.venv/pyvenv.cfg').read_text(encoding='utf-8')
            assert 'include-system-site-packages = false' in config
            record['isolated_site_packages'] = True
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 8001))
            token = secrets.token_urlsafe(32)
            environment = os.environ.copy()
            for name in ['PYTHONPATH', 'PYTHONHOME']:
                environment.pop(name, None)
            environment.update(QA_API_TOKEN=token, QA_TEAM='sopa', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            results = {}
            def request(path, payload=None, credential=None):
                headers = {}
                if credential is not None:
                    headers['Authorization'] = 'Bearer ' + credential
                data = None
                if payload is not None:
                    data = json.dumps(payload).encode('utf-8')
                    headers['Content-Type'] = 'application/json'
                req = urllib.request.Request('http://127.0.0.1:8001' + path, data=data, headers=headers)
                start = time.perf_counter()
                try:
                    with urllib.request.urlopen(req, timeout=60) as response:
                        return response.status, json.load(response), time.perf_counter() - start
                except urllib.error.HTTPError as error:
                    return error.code, json.load(error), time.perf_counter() - start
            server = None
            with (location / 'server.log').open('w', encoding='utf-8') as output:
                try:
                    server = subprocess.Popen([str(python), '-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', '8001', '--workers', '1'], cwd=location, env=environment, stdout=output, stderr=subprocess.STDOUT)
                    record['temporary_server_pid'] = server.pid
                    save(record)
                    deadline = time.monotonic() + 240
                    while time.monotonic() < deadline:
                        if server.poll() is not None:
                            raise RuntimeError('Temporary server exited; inspect temporary server.log')
                        try:
                            status, body, _ = request('/health')
                            if status == 200 and body == {'status': 'ok'}:
                                break
                        except (OSError, urllib.error.URLError):
                            pass
                        time.sleep(1)
                    else:
                        raise RuntimeError('Timed out waiting for temporary server readiness')
                    results['health'] = {'status': status, 'body': body}
                    results['authentication'] = []
                    for path, payload in [('/info', None), ('/predict', {'id':'auth','context':'A test.','question':'What?'})]:
                        for credential in [None, 'deliberately-invalid-test-token']:
                            status, body, _ = request(path, payload, credential)
                            assert status == 401 and body.get('error') == 'unauthorized'
                            results['authentication'].append({'path':path,'case':'missing' if credential is None else 'invalid','status':status})
                    status, body, _ = request('/info', credential=token)
                    assert status == 200 and body['team'] == 'sopa' and body['model'] == 'FacebookAI/roberta-base'
                    assert type(body['params']) is int and body['params'] > 0
                    assert body['max_seq_length'] == 384 and body['doc_stride'] == 128
                    results['info'] = body
                    results['samples'] = []
                    samples = json.loads((location / 'public_sample.json').read_text(encoding='utf-8'))['items']
                    for example in samples:
                        payload = {k: example[k] for k in ['id', 'context', 'question']}
                        status, body, seconds = request('/predict', payload, token)
                        assert status == 200 and set(body) == {'id','answer','start','score'}
                        assert body['id'] == example['id']
                        assert type(body['score']) in (int,float) and math.isfinite(body['score']) and 0 <= body['score'] <= 1
                        if body['answer'] is None:
                            assert body['start'] is None
                        else:
                            assert isinstance(body['answer'],str) and type(body['start']) is int and body['start'] >= 0
                            assert example['context'][body['start']:body['start']+len(body['answer'])] == body['answer']
                        match = body['answer'] is None if example['is_impossible'] else body['answer'] in [a['text'] for a in example['answers']]
                        results['samples'].append(dict(id=example['id'], response=body, contract_valid=True, reference_match=match, seconds=seconds))
                    status, body, _ = request('/predict', {'id':'bad','context':'Example.'}, token)
                    assert status == 422 and body['error'] == 'bad_request'
                    results['incomplete_request'] = {'status':status,'body':body}
                    results['contract_valid'] = sum(x['contract_valid'] for x in results['samples'])
                    results['reference_matches'] = sum(x['reference_match'] for x in results['samples'])
                    results['mean_latency_ms'] = 1000 * sum(x['seconds'] for x in results['samples']) / len(samples)
                finally:
                    if server is not None:
                        if server.poll() is None:
                            server.terminate()
                            try:
                                server.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                server.kill()
                                server.wait(timeout=10)
                        record['temporary_server_stopped'] = server.poll() is not None
                    record['tests'] = results
                    save(record)
                    token = None
                    environment.pop('QA_API_TOKEN', None)
        record['phases'][phase] = 'ok'
        record.pop('last_error', None)
        save(record)
        print(phase + ': ok', flush=True)
        if phase == 'test':
            print(json.dumps(record['tests'], ensure_ascii=False, indent=2))
    except Exception as error:
        record['phases'][phase] = 'failed'
        record['last_error'] = str(error)
        save(record)
        raise

if __name__ == '__main__':
    main()

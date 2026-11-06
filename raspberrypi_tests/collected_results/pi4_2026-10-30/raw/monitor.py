#!/usr/bin/env python3
"""Executa UM benchmark e coleta recursos + ACS712/Arduino (processo completo).

CSV de energia: instante de RECEPÇÃO no Raspberry e timestamp enviado pelo Arduino.
A energia é estimada por integral trapezoidal usando tempo do Arduino entre amostras
recebidas dentro da janela do processo; NÃO equivale à energia apenas das inferências.
A potência fornecida pelo Arduino assume tensão fixa de 5V e exige calibração.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

try:
    import psutil
    import serial
except ImportError as exc:
    sys.exit(f'Dependência ausente: {exc}. Instale psutil e pyserial no .venv.')

RESOURCE_FIELDS = ['experimento_id', 'timestamp_utc', 'tempo_s', 'modelo', 'repeticao', 'fase',
                   'cpu_percent', 'ram_usada_mb', 'ram_percent', 'processo_cpu_percent',
                   'processo_rss_mb', 'gpu_uso_percent', 'gpu_clock_mhz', 'temperatura_c']
ENERGY_FIELDS = ['experimento_id', 'modelo', 'repeticao', 'timestamp_utc', 'tempo_raspberry_ns',
                 'tempo_arduino_ms', 'corrente_A', 'potencia_W', 'energia_acumulada_J', 'fase']


def clock():
    if not shutil.which('vcgencmd'):
        return None
    try:
        r = subprocess.run(['vcgencmd', 'measure_clock', 'core'], capture_output=True,
                           text=True, timeout=1)
        return round(float(r.stdout.strip().split('=')[-1]) / 1e6, 1) if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def gpu_usage():
    for p in ('/sys/class/drm/card0/device/gpu_busy_percent',
              '/sys/class/drm/card1/device/gpu_busy_percent'):
        try:
            v = float(Path(p).read_text().strip())
            if 0 <= v <= 100:
                return v
        except (OSError, ValueError):
            pass
    return None


def temperature():
    try:
        return round(float(Path('/sys/class/thermal/thermal_zone0/temp').read_text().strip()) / 1000, 1)
    except (OSError, ValueError):
        return None


def proc_metrics(root):
    try:
        children = [root] + root.children(recursive=True)
    except (psutil.Error, AttributeError):
        return None, None
    cpu, rss, ok = 0., 0, False
    for p in children:
        try:
            cpu += p.cpu_percent(interval=None)
            rss += p.memory_info().rss
            ok = True
        except psutil.Error:
            pass
    return (round(cpu, 2), round(rss/1e6, 1)) if ok else (None, None)


def stop_process(p):
    if p is None or p.poll() is not None:
        return
    for s, seconds in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        try:
            os.killpg(p.pid, s)
        except ProcessLookupError:
            return
        try:
            p.wait(timeout=seconds)
            return
        except subprocess.TimeoutExpired:
            pass


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--modelo', required=True)
    ap.add_argument('--repeticao', type=int, required=True)
    ap.add_argument('--intervalo', type=float, default=0.5)
    ap.add_argument('--csv', required=True, help='CSV de recursos')
    ap.add_argument('--energia-csv', required=True, help='CSV de energia do Arduino')
    ap.add_argument('--resumo', required=True)
    ap.add_argument('--porta', default='/dev/ttyACM0')
    ap.add_argument('--baudrate', type=int, default=115200)
    ap.add_argument('--sensor-timeout', type=float, default=15., help='Tempo máximo para primeira amostra')
    ap.add_argument('--comando', nargs=argparse.REMAINDER, required=True)
    a = ap.parse_args()
    cmd = a.comando[1:] if a.comando and a.comando[0] == '--' else a.comando
    if a.intervalo <= 0 or a.repeticao <= 0 or a.sensor_timeout <= 0 or not cmd:
        ap.error('Intervalo, repeticao, sensor-timeout e comando devem ser válidos.')

    resource, energy, summary = map(Path, (a.csv, a.energia_csv, a.resumo))
    if len({str(p.resolve()) for p in (resource, energy, summary)}) != 3:
        ap.error('Arquivos CSV e JSON devem ter caminhos diferentes.')
    for p in (resource, energy, summary):
        if p.exists():
            ap.error(f'Arquivo já existe: {p}; use outra repetição para não sobrescrever.')
    for p in (resource, energy, summary):
        p.parent.mkdir(parents=True, exist_ok=True)

    expid = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    samples = []
    lock = threading.Lock()
    stop_evt = threading.Event()
    first_sample = threading.Event()
    serial_error = []
    proc = None
    code = 1
    started_at = ended_at = None
    start_ns = end_ns = None
    count = 0
    interrupted = False
    try:
        # Falha de conexão serial impede benchmark sem medição elétrica.
        with serial.Serial(a.porta, a.baudrate, timeout=0.25) as arduino:
            print(f'[ENERGIA] Arduino conectado: {a.porta} @ {a.baudrate}', flush=True)
            with energy.open('x', newline='', encoding='utf-8') as ef, resource.open('x', newline='', encoding='utf-8') as rf:
                ew = csv.DictWriter(ef, fieldnames=ENERGY_FIELDS)
                rw = csv.DictWriter(rf, fieldnames=RESOURCE_FIELDS)
                ew.writeheader(); rw.writeheader()
                ef.flush(); rf.flush()

                def read_energy():
                    previous = None
                    accumulated = 0.
                    while not stop_evt.is_set():
                        try:
                            line = arduino.readline().decode('utf-8', errors='replace').strip()
                        except (serial.SerialException, OSError) as exc:
                            serial_error.append(str(exc)); stop_evt.set(); break
                        if not line or line.startswith('#') or line.startswith('tempo_ms'):
                            continue
                        parts = line.split(',')
                        if len(parts) != 3:
                            continue
                        try:
                            ms, amps, watts = int(parts[0]), float(parts[1]), float(parts[2])
                            if not all(map(__import__('math').isfinite, (amps, watts))):
                                continue
                        except ValueError:
                            continue
                        received_ns = time.monotonic_ns()
                        # Use monotonic timestamp for actual process interval.
                        phase = 'processo_completo' if start_ns is not None and (end_ns is None or received_ns <= end_ns) and received_ns >= start_ns else 'pre_inicio'
                        if previous is not None and phase == 'processo_completo' and previous[3] == 'processo_completo':
                            dt = (ms - previous[0])/1000.0
                            if 0 < dt < 5:
                                accumulated += (watts + previous[1]) * 0.5 * dt
                        previous = (ms, watts, received_ns, phase)
                        record = dict(experimento_id=expid, modelo=a.modelo, repeticao=a.repeticao,
                                      timestamp_utc=now_iso(), tempo_raspberry_ns=received_ns,
                                      tempo_arduino_ms=ms, corrente_A=amps, potencia_W=watts,
                                      energia_acumulada_J=round(accumulated, 6), fase=phase)
                        with lock:
                            ew.writerow(record); ef.flush(); samples.append(record)
                        first_sample.set()

                thread = threading.Thread(target=read_energy, name='arduino-acquisition', daemon=True)
                thread.start()
                if not first_sample.wait(a.sensor_timeout) or serial_error:
                    raise RuntimeError('Arduino não forneceu amostras válidas antes do prazo: ' + str(serial_error))
                print('[ENERGIA] Primeira amostra recebida; iniciando benchmark.', flush=True)

                psutil.cpu_percent(interval=None)
                started_at = now_iso()
                start_ns = time.monotonic_ns()
                proc = subprocess.Popen(cmd, start_new_session=True)
                root = psutil.Process(proc.pid)
                root.cpu_percent(interval=None)
                print(f'[MONITOR] {a.modelo} run{a.repeticao} PID={proc.pid} ID={expid}', flush=True)
                next_sample = time.monotonic()
                while True:
                    t = time.monotonic()
                    if t >= next_sample:
                        mem = psutil.virtual_memory()
                        pcpu, prss = proc_metrics(root)
                        rw.writerow(dict(experimento_id=expid, timestamp_utc=now_iso(),
                                         tempo_s=round((time.monotonic_ns()-start_ns)/1e9, 3),
                                         modelo=a.modelo, repeticao=a.repeticao,
                                         fase='processo_completo', cpu_percent=psutil.cpu_percent(interval=None),
                                         ram_usada_mb=round(mem.used/1e6,1), ram_percent=mem.percent,
                                         processo_cpu_percent=pcpu, processo_rss_mb=prss,
                                         gpu_uso_percent=gpu_usage(), gpu_clock_mhz=clock(),
                                         temperatura_c=temperature()))
                        rf.flush(); count += 1
                        next_sample += a.intervalo
                        if next_sample < time.monotonic():
                            next_sample = time.monotonic()+a.intervalo
                    if serial_error:
                        raise RuntimeError('Leitura serial interrompida: '+str(serial_error))
                    if proc.poll() is not None:
                        break
                    time.sleep(min(.05, max(0., next_sample-time.monotonic())))
                code = proc.wait()
                end_ns = time.monotonic_ns()
                ended_at = now_iso()
                stop_evt.set()
                thread.join(timeout=2)
    except KeyboardInterrupt:
        interrupted = True
        code = 130
        print('\n[MONITOR] Interrompido.', file=sys.stderr)
        stop_process(proc)
    except (OSError, psutil.Error, serial.SerialException, RuntimeError) as exc:
        print(f'[MONITOR] ERRO: {exc}', file=sys.stderr)
        stop_process(proc)
        code = 1
    finally:
        if proc is not None and end_ns is None:
            end_ns = time.monotonic_ns()
            ended_at = now_iso()
        stop_evt.set()
        measured = [r for r in samples if r['fase'] == 'processo_completo']
        joules = measured[-1]['energia_acumulada_J'] if len(measured) > 1 else None
        mean_power = (sum(r['potencia_W'] for r in measured)/len(measured)) if measured else None
        metadata = dict(experimento_id=expid, modelo=a.modelo, repeticao=a.repeticao,
                        inicio_utc=started_at, fim_utc=ended_at,
                        duracao_processo_s=round((end_ns-start_ns)/1e9,3) if start_ns and end_ns else None,
                        n_amostras_recursos=count, n_amostras_energia=len(measured),
                        intervalo_recursos_s=a.intervalo, porta=a.porta, baudrate=a.baudrate,
                        comando=cmd, exit_code=code, interrompido=interrupted,
                        energia_j=round(joules,6) if joules is not None else None,
                        potencia_media_amostral_w=round(mean_power,6) if mean_power is not None else None,
                        potencia_informada_por_arduino='Estimativa I*5V; requer calibração e tensão real.',
                        escopo='processo_completo (inicialização, warmup, inferências, finalização)',
                        observacao='Energia aproximada entre primeiras e últimas amostras na janela do processo, sem extrapolar bordas; não é energia das 100 inferências.')
        try:
            with summary.open('x', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                f.write('\n')
        except OSError as exc:
            print(f'[MONITOR] Falha ao gravar metadados: {exc}', file=sys.stderr)
            code = 1
        print(f'[MONITOR] status={code} recursos={count} energia={len(measured)} E≈{joules} J', flush=True)
    return code if isinstance(code,int) and code >= 0 else 1

if __name__ == '__main__':
    sys.exit(main())

"""
Run the OpenEMMA benchmark sweep with a CARLA restart supervisor.

benchmark.py owns job resume through the output CSV. This wrapper rotates CARLA
between towns, keeps CARLA alive across crashes, and reruns benchmark.py until
it exits cleanly.
"""

import argparse
import os
import subprocess
import sys
import time

from ui_common import carla_setup


CARLA_ROOT = carla_setup.setup_carla_paths()

import carla

from aggregate_results import aggregate_results


HOST = 'localhost'
SERVER_READY_TIMEOUT_S = 120.0
SERVER_READY_SETTLE_S = 5.0
SERVER_READY_ATTEMPT_TIMEOUT_S = 3.0
SERVER_READY_POLL_INTERVAL_S = 2.0
SERVER_RESTART_DELAY_S = 3.0
NATIVE_CRASH_RESTART_DELAY_S = 10.0
SERVER_LAUNCH_SAFETY_MARGIN = 5
BENCHMARK_CONNECTION_LOST = 42
BENCHMARK_TOWN_DONE = 43


def carla_executable():
    if os.name == 'nt':
        executable = os.path.join(CARLA_ROOT, 'CarlaUE4.exe')
    else:
        executable = os.path.join(CARLA_ROOT, 'CarlaUE4.sh')

    if not os.path.exists(executable):
        raise FileNotFoundError(
            f'CARLA server executable not found at {executable}'
        )
    return executable


def try_server_version(port, timeout_s=SERVER_READY_ATTEMPT_TIMEOUT_S):
    client = carla.Client(HOST, port)
    client.set_timeout(timeout_s)
    return client.get_server_version()


def wait_for_server_ready(port, proc=None, timeout_s=SERVER_READY_TIMEOUT_S):
    deadline = time.monotonic() + timeout_s
    last_error = None
    while True:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f'CARLA server exited before ready with code {proc.returncode}'
            )
        try:
            return try_server_version(port)
        except Exception as exc:
            last_error = exc
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        time.sleep(min(SERVER_READY_POLL_INTERVAL_S, remaining_s))
    raise TimeoutError(
        f'CARLA server was not ready after {timeout_s:.0f}s: {last_error}'
    )


def is_server_ready(port):
    try:
        version = try_server_version(port)
    except Exception:
        return False
    print(f'[Sweep] CARLA server ready: {version}')
    return True


def launch_server(port, quality_level):
    executable = carla_executable()
    cmd = [
        executable,
        f'-carla-rpc-port={port}',
        f'-quality-level={quality_level}',
        '-RenderOffScreen',
        '-nosound',
    ]
    print('[Sweep] Launching CARLA server:')
    print('[Sweep] ' + ' '.join(cmd))

    kwargs = {
        'cwd': CARLA_ROOT,
        'stdout': subprocess.DEVNULL,
        'stderr': subprocess.DEVNULL,
    }
    if os.name == 'nt':
        kwargs['creationflags'] = getattr(
            subprocess,
            'CREATE_NEW_PROCESS_GROUP',
            0,
        )
    else:
        kwargs['start_new_session'] = True

    return subprocess.Popen(cmd, **kwargs)


def check_server_ready(args, server_proc, launched_by_us):
    if is_server_ready(args.port):
        return server_proc, launched_by_us, True

    if server_proc is not None and server_proc.poll() is not None:
        print(f'[Sweep] Previous CARLA process exited: {server_proc.returncode}')
        server_proc = None
        launched_by_us = False

    return server_proc, launched_by_us, False


def terminate_process(proc):
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10.0)
        return
    except Exception:
        pass
    try:
        proc.kill()
        proc.wait(timeout=10.0)
    except Exception:
        pass


def taskkill_carla_windows():
    if os.name != 'nt':
        return
    for image_name in ('CarlaUE4.exe', 'CarlaUE4-Win64-Shipping.exe'):
        subprocess.run(
            ['taskkill', '/F', '/IM', image_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def teardown_for_restart(server_proc, settle_delay_s=SERVER_RESTART_DELAY_S):
    print('[Sweep] Tearing down CARLA before restart...')
    terminate_process(server_proc)
    taskkill_carla_windows()
    if settle_delay_s > 0:
        print(f'[Sweep] Waiting {settle_delay_s:.0f}s before relaunch...')
        time.sleep(settle_delay_s)
    return None, False


def stop_launched_server(server_proc, launched_by_us):
    if not launched_by_us:
        return
    print('[Sweep] Stopping CARLA server launched by run_sweep.py...')
    terminate_process(server_proc)
    taskkill_carla_windows()


def benchmark_command(args):
    benchmark_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'benchmark.py')
    cmd = [
        sys.executable,
        benchmark_path,
        '--out',
        args.out,
        '--duration',
        str(args.duration),
        '--reps',
        str(args.reps),
        '--port',
        str(args.port),
    ]
    if args.config:
        cmd.extend(['--config', args.config])
    return cmd


def run_benchmark_once(args):
    cmd = benchmark_command(args)
    print('[Sweep] Running benchmark.py:')
    print('[Sweep] ' + ' '.join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                try:
                    print(line, end='')
                except UnicodeEncodeError:
                    print(line.encode('utf-8', 'replace').decode('utf-8', 'replace'),
                          end='')
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
    return proc.wait()


def restart_or_fail(
    args,
    restart_count,
    server_proc,
    reason,
    settle_delay_s=SERVER_RESTART_DELAY_S,
):
    restart_count += 1
    print(
        f'[Sweep] Restart {restart_count}/{args.max_restarts} required: {reason}'
    )
    if restart_count > args.max_restarts:
        print('[Sweep] Maximum restart count exceeded; stopping sweep.')
        server_proc, _ = teardown_for_restart(server_proc, settle_delay_s)
        return restart_count, server_proc, False
    server_proc, _ = teardown_for_restart(server_proc, settle_delay_s)
    return restart_count, server_proc, True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run benchmark.py to completion with CARLA auto-restarts'
    )
    parser.add_argument('--out', required=True,
                        help='Benchmark results CSV path')
    parser.add_argument('--duration', type=float, default=150.0,
                        help='Measured simulation seconds per run')
    parser.add_argument('--reps', type=int, default=3,
                        help='Seeds per condition')
    parser.add_argument('--config', default=None,
                        help='Optional JSON condition list passed to benchmark.py')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA RPC port')
    parser.add_argument('--max-restarts', type=int, default=20,
                        help='Maximum CARLA restarts before failing')
    parser.add_argument('--quality-level', default='Low',
                        help='CARLA quality level')
    args = parser.parse_args()

    if args.duration <= 0:
        raise ValueError('--duration must be positive')
    if args.reps <= 0:
        raise ValueError('--reps must be positive')
    if args.max_restarts < 0:
        raise ValueError('--max-restarts must be non-negative')
    args.out = os.path.abspath(args.out)
    if args.config:
        args.config = os.path.abspath(args.config)
    return args


def main():
    # Best-effort: keep the supervisor's own output from crashing on consoles
    # whose codepage (e.g. Windows cp949) cannot encode chars in the streamed
    # benchmark output. Falls back silently on streams without reconfigure().
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    args = parse_args()
    server_proc = None
    launched_by_us = False
    restart_count = 0
    planned_town_rotations = 0
    server_launch_count = 0
    done = False

    try:
        while True:
            max_server_launches = (
                args.max_restarts
                + planned_town_rotations
                + SERVER_LAUNCH_SAFETY_MARGIN
                + 1
            )
            server_proc, launched_by_us, ready = check_server_ready(
                args,
                server_proc,
                launched_by_us,
            )
            if not ready:
                if server_launch_count >= max_server_launches:
                    print(
                        '[Sweep] Maximum CARLA launch attempts exceeded; '
                        'stopping sweep.'
                    )
                    server_proc, _ = teardown_for_restart(server_proc)
                    return 1

                server_launch_count += 1
                print(
                    f'[Sweep] CARLA launch attempt '
                    f'{server_launch_count}/{max_server_launches}'
                )
                try:
                    server_proc = launch_server(args.port, args.quality_level)
                    launched_by_us = True
                    print(
                        f'[Sweep] Waiting {SERVER_READY_SETTLE_S:.0f}s '
                        'before readiness checks...'
                    )
                    time.sleep(SERVER_READY_SETTLE_S)
                    version = wait_for_server_ready(
                        args.port,
                        proc=server_proc,
                    )
                    print(f'[Sweep] CARLA server ready: {version}')
                except Exception as exc:
                    print(f'[Sweep] CARLA startup failed ({exc}); retrying.')
                    server_proc, _ = teardown_for_restart(server_proc)
                    launched_by_us = False
                    continue

            exit_code = run_benchmark_once(args)
            if exit_code == 0:
                print('[Sweep] benchmark.py completed all remaining jobs.')
                done = True
                break

            if exit_code == BENCHMARK_TOWN_DONE:
                planned_town_rotations += 1
                print(
                    '[Sweep] benchmark.py completed one town '
                    f'(exit {BENCHMARK_TOWN_DONE}); rotating CARLA server.'
                )
                server_proc, _ = teardown_for_restart(
                    server_proc,
                    SERVER_RESTART_DELAY_S,
                )
                launched_by_us = False
                continue

            if exit_code == BENCHMARK_CONNECTION_LOST:
                reason = 'benchmark.py reported CARLA connection loss (exit 42)'
                settle_delay_s = SERVER_RESTART_DELAY_S
            else:
                reason = f'benchmark.py exited with code {exit_code}'
                settle_delay_s = NATIVE_CRASH_RESTART_DELAY_S

            restart_count, server_proc, should_continue = restart_or_fail(
                args,
                restart_count,
                server_proc,
                reason,
                settle_delay_s,
            )
            launched_by_us = False
            if not should_continue:
                return 1

        print('[Sweep] Aggregating results...')
        aggregate_results(args.out)
        return 0
    finally:
        if done:
            stop_launched_server(server_proc, launched_by_us)


if __name__ == '__main__':
    raise SystemExit(main())

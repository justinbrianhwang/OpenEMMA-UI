"""
Headless closed-loop benchmark harness for the OpenEMMA LLaMA 4-bit backend.

The model is loaded once per invocation, then reused for one active town's
remaining weather and seed conditions. Results are appended as one CSV row per
run and can be resumed by rerunning with the same output file.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback

from ui_common.carla_setup import setup_carla_paths
setup_carla_paths()

import carla

from openemmaUI import LOCAL_MODELS, OpenEMMACarlaAgent
from ui_common.agent_runner import AgentRunner


SERVER_DEATH_EXIT_CODE = 42
EXIT_TOWN_DONE = 43
TARGET_FPS = 20
FIXED_DELTA_SECONDS = 1.0 / TARGET_FPS
OFFROAD_TERMINAL_SECONDS = 10.0

FIELDNAMES = [
    'town',
    'weather_label',
    'seed',
    'duration_s',
    'route_len',
    'route_reached',
    'completion_pct',
    'distance_m',
    'offroad_pct',
    'collisions',
    'stuck_events',
    'recovery_events',
    'route_regens',
    'vlm_frame_pct',
    'fallback_frame_pct',
    'speed_floor_pct',
    'speed_clamp_events',
    'curv_clamp_events',
    'degenerate_rejections',
    'hallucination_pct',
    'avg_speed_mps',
    'steer_std',
    'outcome',
]


def make_weather(cloudiness=0.0, precipitation=0.0,
                 precipitation_deposits=0.0, wind_intensity=0.0,
                 sun_altitude_angle=70.0, sun_azimuth_angle=0.0,
                 fog_density=0.0, wetness=0.0):
    weather = carla.WeatherParameters()
    weather.cloudiness = cloudiness
    weather.precipitation = precipitation
    weather.precipitation_deposits = precipitation_deposits
    weather.wind_intensity = wind_intensity
    weather.sun_altitude_angle = sun_altitude_angle
    weather.sun_azimuth_angle = sun_azimuth_angle
    weather.fog_density = fog_density
    if hasattr(weather, 'wetness'):
        weather.wetness = wetness
    return weather


WEATHER_PRESETS = {
    'ClearNoon': make_weather(cloudiness=5.0, sun_altitude_angle=70.0),
    'HardRainNoon': make_weather(
        cloudiness=95.0,
        precipitation=90.0,
        precipitation_deposits=85.0,
        wind_intensity=60.0,
        sun_altitude_angle=70.0,
        wetness=100.0,
    ),
    'WetCloudyNoon': make_weather(
        cloudiness=85.0,
        precipitation=0.0,
        precipitation_deposits=70.0,
        sun_altitude_angle=70.0,
        wetness=80.0,
    ),
    'ClearSunset': make_weather(cloudiness=10.0, sun_altitude_angle=-5.0),
    'ClearNight': make_weather(cloudiness=10.0, sun_altitude_angle=-20.0),
    'HardRainNight': make_weather(
        cloudiness=95.0,
        precipitation=90.0,
        precipitation_deposits=85.0,
        wind_intensity=60.0,
        sun_altitude_angle=-20.0,
        wetness=100.0,
    ),
}


def dedupe_conditions(conditions):
    seen = set()
    deduped = []
    for condition in conditions:
        key = (
            condition['town'],
            condition['weather_preset'],
            condition['label'],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(condition)
    return deduped


def built_in_conditions():
    conditions = [
        {'town': 'Town01', 'weather_preset': 'ClearNoon', 'label': 'ClearNoon'},
    ]
    for town in ('Town01', 'Town02', 'Town03', 'Town05'):
        conditions.append({
            'town': town,
            'weather_preset': 'ClearNoon',
            'label': 'ClearNoon',
        })
    for preset_name in WEATHER_PRESETS:
        conditions.append({
            'town': 'Town01',
            'weather_preset': preset_name,
            'label': preset_name,
        })
    return dedupe_conditions(conditions)


CONDITIONS = built_in_conditions()


def load_conditions(config_path):
    if not config_path:
        return CONDITIONS

    with open(config_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        raw_conditions = payload.get('conditions', [])
    elif isinstance(payload, list):
        raw_conditions = payload
    else:
        raise ValueError('--config must contain a list or an object with conditions')

    conditions = []
    for item in raw_conditions:
        town = item['town']
        preset = (
            item.get('weather_preset')
            or item.get('weather')
            or item.get('preset')
        )
        label = item.get('label') or item.get('weather_label') or preset
        if preset not in WEATHER_PRESETS:
            raise ValueError(f'Unknown weather preset in --config: {preset}')
        conditions.append({
            'town': town,
            'weather_preset': preset,
            'label': label,
        })
    return dedupe_conditions(conditions)


def build_jobs(conditions, reps):
    jobs = []
    for condition in conditions:
        for seed in range(reps):
            jobs.append((condition, seed))
    return sorted(jobs, key=lambda job: job[0]['town'])


def job_key(condition, seed):
    return (condition['town'], condition['label'], str(seed))


def read_completed_keys(csv_path):
    completed = set()
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return completed

    with open(csv_path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            town = row.get('town')
            weather_label = row.get('weather_label')
            seed = row.get('seed')
            if town and weather_label and seed is not None:
                completed.add((town, weather_label, str(seed)))
    return completed


def open_csv_writer(csv_path):
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    handle = open(csv_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
    return handle, writer


class CollisionCounter:
    def __init__(self, world, vehicle):
        self.count = 0
        self.sensor = None
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            blueprint,
            carla.Transform(),
            attach_to=vehicle,
        )
        self.sensor.listen(self._on_collision)

    def _on_collision(self, _event):
        self.count += 1

    def destroy(self):
        if self.sensor is None:
            return
        try:
            self.sensor.stop()
            self.sensor.destroy()
        except Exception:
            pass
        self.sensor = None


def is_carla_server_error(exc):
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    patterns = (
        'timeout',
        'time-out',
        'connection',
        'connect',
        'rpc',
        'server',
        'simulator',
        'episode',
        'stream',
        'disconnect',
        'disconnected',
        'lost',
        'closed',
        'broken pipe',
        'unavailable',
        'transport',
    )
    return any(pattern in message for pattern in patterns)


def wait_for_cot_idle(agent, timeout_s=300.0):
    start = time.monotonic()
    while getattr(agent, '_cot_running', False):
        if time.monotonic() - start > timeout_s:
            print('[Benchmark] Timed out waiting for pending CoT inference; continuing.')
            return
        time.sleep(0.1)


def create_headless_runner(args):
    runner = AgentRunner(
        host=args.host,
        port=args.port,
        town='',
        model_name='OpenEMMA LLaMA-4bit Benchmark',
        target_fps=TARGET_FPS,
    )

    print(f"[Benchmark] Connecting to CARLA at {args.host}:{args.port}...")
    runner.client = carla.Client(args.host, args.port)
    runner.client.set_timeout(30.0)
    return runner


def apply_sync_settings(runner):
    settings = runner.world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    runner.world.apply_settings(settings)

    traffic_manager = runner.client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    runner.traffic_manager = traffic_manager


def load_benchmark_world(runner, town):
    print(f"[Benchmark] Loading {town}...")
    runner.town = town
    runner.world = runner.client.load_world(town)
    apply_sync_settings(runner)
    return town


def prepare_run_world(runner, condition):
    weather = WEATHER_PRESETS[condition['weather_preset']]
    runner.world.set_weather(weather)

    runner._spawn_ego_vehicle()
    runner.world.tick()


def destroy_sensor_data(sensor_data):
    if sensor_data is None:
        return

    sensor_actors = list(getattr(sensor_data, '_sensor_actors', []) or [])
    if sensor_actors:
        for sensor in sensor_actors:
            if sensor is None:
                continue
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                sensor.destroy()
            except Exception:
                pass
    else:
        try:
            sensor_data.cleanup()
        except Exception:
            pass
        return

    for attr_name in ('_sensor_actors', '_pseudo_sensors'):
        items = getattr(sensor_data, attr_name, None)
        if items is not None:
            try:
                items.clear()
            except Exception:
                pass

    sensors = getattr(sensor_data, '_sensors', None)
    if sensors is not None:
        try:
            sensors.clear()
        except Exception:
            pass


def teardown_run_actors(runner, collision_counter=None, tick=True):
    if collision_counter is not None:
        try:
            collision_counter.destroy()
        except Exception:
            pass

    if runner is None:
        return

    if getattr(runner, 'sensor_data', None) is not None:
        destroy_sensor_data(runner.sensor_data)
        runner.sensor_data = None

    if getattr(runner, 'ego_vehicle', None) is not None:
        try:
            runner.ego_vehicle.destroy()
        except Exception:
            pass
        runner.ego_vehicle = None

    if getattr(runner, 'safety_limiter', None) is not None:
        try:
            runner.safety_limiter.set_route(None)
        except Exception:
            pass

    if tick and getattr(runner, 'world', None) is not None:
        try:
            runner.world.tick()
        except Exception:
            pass


def cleanup_headless(runner, collision_counter=None):
    if runner is None:
        return

    teardown_run_actors(runner, collision_counter, tick=False)

    if getattr(runner, 'traffic_manager', None) is not None:
        try:
            runner.traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass

    if getattr(runner, 'world', None) is not None:
        try:
            settings = runner.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            runner.world.apply_settings(settings)
        except Exception:
            pass


def generate_seeded_route(runner, seed):
    start_location = runner.ego_vehicle.get_transform().location
    spawn_points = runner.world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError('No spawn points found in the map')

    distances = []
    for index, spawn_point in enumerate(spawn_points):
        distance = start_location.distance(spawn_point.location)
        distances.append((distance, index, spawn_point))
    distances.sort(key=lambda item: (-item[0], item[1]))

    destination = distances[seed % len(distances)][2]
    return runner.generate_route(
        start_location=start_location,
        end_location=destination.location,
    )


def run_warmup(world, vehicle, warmup_seconds):
    ticks = int(max(0.0, warmup_seconds) / FIXED_DELTA_SECONDS)
    if ticks <= 0:
        return

    brake = carla.VehicleControl()
    brake.throttle = 0.0
    brake.brake = 1.0
    for _ in range(ticks):
        vehicle.apply_control(brake)
        world.tick()


def is_vehicle_offroad(carla_map, vehicle):
    veh_loc = vehicle.get_location()
    veh_transform = vehicle.get_transform()
    veh_fwd = veh_transform.get_forward_vector()
    veh_right_x = veh_fwd.y
    veh_right_y = -veh_fwd.x

    waypoint = carla_map.get_waypoint(veh_loc, project_to_road=False)
    if waypoint is None or waypoint.lane_type != carla.LaneType.Driving:
        return True

    for side in (1.0, -1.0):
        edge_loc = carla.Location(
            x=veh_loc.x + veh_right_x * side,
            y=veh_loc.y + veh_right_y * side,
            z=veh_loc.z,
        )
        edge_wp = carla_map.get_waypoint(edge_loc, project_to_road=False)
        if edge_wp is None or edge_wp.lane_type != carla.LaneType.Driving:
            return True
    return False


def pct(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def round_value(value):
    return round(float(value), 4)


def build_result_row(condition, seed, metrics, sim_duration, distance_m,
                     offroad_frames, loop_frames, collisions, outcome):
    route_idx = max(0, int(metrics.get('route_idx', 0)))
    route_len = int(metrics.get('route_len', 0))
    route_reached = route_idx
    completion_pct = pct(route_reached, max(route_len, 1))

    frames = int(metrics.get('frames', 0))
    vlm_frame_pct = pct(metrics.get('vlm_frames', 0), frames)
    fallback_frame_pct = pct(metrics.get('fallback_frames', 0), frames)
    speed_floor_pct = pct(metrics.get('speed_floor_frames', 0), frames)
    hallucination_pct = pct(metrics.get('cot_stop_or_red', 0), metrics.get('cot_cycles', 0))
    offroad_pct = pct(offroad_frames, loop_frames)

    if frames > 0:
        steer_mean = float(metrics.get('steer_sum', 0.0)) / frames
        steer_sq_mean = float(metrics.get('steer_sq_sum', 0.0)) / frames
        steer_std = math.sqrt(max(0.0, steer_sq_mean - steer_mean * steer_mean))
        avg_speed_mps = float(metrics.get('speed_sum', 0.0)) / frames
    else:
        steer_std = 0.0
        avg_speed_mps = 0.0

    return {
        'town': condition['town'],
        'weather_label': condition['label'],
        'seed': seed,
        'duration_s': round_value(sim_duration),
        'route_len': route_len,
        'route_reached': route_reached,
        'completion_pct': round_value(completion_pct),
        'distance_m': round_value(distance_m),
        'offroad_pct': round_value(offroad_pct),
        'collisions': collisions,
        'stuck_events': int(metrics.get('stuck_events', 0)),
        'recovery_events': int(metrics.get('recovery_events', 0)),
        'route_regens': int(metrics.get('route_regens', 0)),
        'vlm_frame_pct': round_value(vlm_frame_pct),
        'fallback_frame_pct': round_value(fallback_frame_pct),
        'speed_floor_pct': round_value(speed_floor_pct),
        'speed_clamp_events': int(metrics.get('speed_clamp_events', 0)),
        'curv_clamp_events': int(metrics.get('curv_clamp_events', 0)),
        'degenerate_rejections': int(metrics.get('degenerate_rejections', 0)),
        'hallucination_pct': round_value(hallucination_pct),
        'avg_speed_mps': round_value(avg_speed_mps),
        'steer_std': round_value(steer_std),
        'outcome': outcome,
    }


def build_error_row(condition, seed, agent):
    metrics = agent.get_metrics_snapshot() if agent is not None else {}
    return build_result_row(
        condition=condition,
        seed=seed,
        metrics=metrics,
        sim_duration=0.0,
        distance_m=0.0,
        offroad_frames=0,
        loop_frames=0,
        collisions=0,
        outcome='error',
    )


def run_headless_loop(args, condition, seed, agent, runner, collision_counter):
    world = runner.world
    vehicle = runner.ego_vehicle
    carla_map = world.get_map()
    sensor_data = runner.sensor_data

    start_sim_time = None
    elapsed = 0.0
    loop_frames = 0
    offroad_frames = 0
    offroad_consecutive = 0
    offroad_terminal_frames = max(
        1,
        int(OFFROAD_TERMINAL_SECONDS / FIXED_DELTA_SECONDS),
    )
    time_to_first_failure = None
    distance_m = 0.0
    last_location = vehicle.get_location()
    outcome = 'timeout'
    max_route_idx_seen = 0
    route_len_at_max_idx = 0

    def update_route_progress(metrics):
        nonlocal max_route_idx_seen, route_len_at_max_idx
        route_idx = max(0, int(metrics.get('route_idx', 0)))
        route_len = int(metrics.get('route_len', 0))
        if route_len > 0 and route_len_at_max_idx <= 0:
            route_len_at_max_idx = route_len
        if route_idx > max_route_idx_seen:
            max_route_idx_seen = route_idx
            route_len_at_max_idx = route_len
        return route_idx, route_len

    update_route_progress(agent.get_metrics_snapshot())

    while elapsed < args.duration:
        step_started_at = time.monotonic()

        frame = world.tick()
        timestamp = world.get_snapshot().timestamp.elapsed_seconds
        if start_sim_time is None:
            start_sim_time = timestamp
        elapsed = timestamp - start_sim_time

        if not vehicle.is_alive:
            raise RuntimeError('Ego vehicle is no longer alive')

        input_data = sensor_data.get_data(frame, vehicle, timeout=0.5)
        control = agent.run_step(input_data, timestamp)
        control = runner.safety_limiter.apply(control, vehicle, carla_map)

        if hasattr(agent, 'ui_violations'):
            agent.ui_violations = getattr(runner.safety_limiter, 'violations', {})
        if hasattr(agent, 'ui_red_light'):
            agent.ui_red_light = runner.safety_limiter.violations.get('red_light', False)
        if runner.safety_limiter.violations.get('red_light', False):
            agent.curr_notice = 'Red light ahead.'

        vehicle.apply_control(control)

        location = vehicle.get_location()
        distance_m += last_location.distance(location)
        last_location = location

        loop_frames += 1
        offroad = is_vehicle_offroad(carla_map, vehicle)
        if offroad:
            offroad_frames += 1
            offroad_consecutive += 1
            if time_to_first_failure is None:
                time_to_first_failure = elapsed
        else:
            offroad_consecutive = 0

        if collision_counter.count > 0:
            if time_to_first_failure is None:
                time_to_first_failure = elapsed
            outcome = 'collision'
            break

        if offroad_consecutive >= offroad_terminal_frames:
            outcome = 'offroad_terminal'
            break

        metrics = agent.get_metrics_snapshot()
        route_idx, route_len = update_route_progress(metrics)
        if route_len > 0 and route_idx >= max(0, route_len - 5):
            outcome = 'completed'
            break

        step_elapsed = time.monotonic() - step_started_at
        sleep_seconds = FIXED_DELTA_SECONDS - step_elapsed
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)

    metrics = agent.get_metrics_snapshot()
    update_route_progress(metrics)
    metrics = metrics.copy()
    metrics['route_idx'] = max_route_idx_seen
    if route_len_at_max_idx > 0:
        metrics['route_len'] = route_len_at_max_idx
    return build_result_row(
        condition=condition,
        seed=seed,
        metrics=metrics,
        sim_duration=elapsed,
        distance_m=distance_m,
        offroad_frames=offroad_frames,
        loop_frames=loop_frames,
        collisions=collision_counter.count,
        outcome=outcome,
    )


def run_one_condition(args, condition, seed, agent, runner):
    collision_counter = None
    try:
        prepare_run_world(runner, condition)
        agent._vehicle = runner.ego_vehicle
        agent.set_runner(runner)

        collision_counter = CollisionCounter(runner.world, runner.ego_vehicle)
        runner.setup_agent_sensors(agent)

        _, world_route = generate_seeded_route(runner, seed)
        agent.set_route(world_route)
        runner.safety_limiter.set_route(world_route)

        run_warmup(runner.world, runner.ego_vehicle, args.warmup)
        row = run_headless_loop(
            args=args,
            condition=condition,
            seed=seed,
            agent=agent,
            runner=runner,
            collision_counter=collision_counter,
        )
        wait_for_cot_idle(agent)
        return row
    except RuntimeError as exc:
        if not is_carla_server_error(exc):
            wait_for_cot_idle(agent)
        raise
    except Exception:
        wait_for_cot_idle(agent)
        raise
    finally:
        teardown_run_actors(runner, collision_counter)
        if agent is not None:
            agent._vehicle = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Headless OpenEMMA LLaMA-4bit CARLA benchmark'
    )
    parser.add_argument('--out', default='benchmark_results.csv',
                        help='CSV path to append benchmark rows')
    parser.add_argument('--duration', type=float, default=150.0,
                        help='Measured simulation seconds per run')
    parser.add_argument('--reps', type=int, default=3,
                        help='Seeds per condition')
    parser.add_argument('--host', default='localhost',
                        help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port')
    parser.add_argument('--config', default=None,
                        help='Optional JSON condition list')
    parser.add_argument('--warmup', type=float, default=3.0,
                        help='Unmeasured simulation warmup seconds')
    args = parser.parse_args()

    if args.duration <= 0:
        raise ValueError('--duration must be positive')
    if args.reps <= 0:
        raise ValueError('--reps must be positive')
    if args.warmup < 0:
        raise ValueError('--warmup must be non-negative')
    return args


def main():
    args = parse_args()
    conditions = load_conditions(args.config)
    jobs = build_jobs(conditions, args.reps)
    all_job_keys = {job_key(condition, seed) for condition, seed in jobs}
    completed_keys = read_completed_keys(args.out)
    already_completed = completed_keys & all_job_keys
    remaining_jobs = [
        (condition, seed)
        for condition, seed in jobs
        if job_key(condition, seed) not in completed_keys
    ]

    print(
        f'[Benchmark] {len(already_completed)}/{len(jobs)} runs already recorded; '
        f'{len(remaining_jobs)} remaining.'
    )
    if not remaining_jobs:
        print(f'[Benchmark] completed={len(already_completed)} remaining=0')
        return 0

    active_town = remaining_jobs[0][0]['town']
    active_jobs = [
        (condition, seed)
        for condition, seed in remaining_jobs
        if condition['town'] == active_town
    ]
    remaining_after_active_town = len(remaining_jobs) - len(active_jobs)
    print(
        f'[Benchmark] Active town for this invocation: {active_town} '
        f'({len(active_jobs)} runs).'
    )

    llama_path = LOCAL_MODELS['llama']
    print(f'[Benchmark] Loading LLaMA 4-bit model once: {llama_path}')
    agent = OpenEMMACarlaAgent(model_path=llama_path, use_4bit=True)

    runner = None
    csv_handle, writer = open_csv_writer(args.out)
    try:
        try:
            runner = create_headless_runner(args)
            load_benchmark_world(runner, active_town)
        except RuntimeError as exc:
            if is_carla_server_error(exc):
                print('[Benchmark] CARLA server connection lost or timed out.')
                print(f'[Benchmark] Exiting with code {SERVER_DEATH_EXIT_CODE}: {exc}')
                csv_handle.flush()
                os.fsync(csv_handle.fileno())
                return SERVER_DEATH_EXIT_CODE
            raise

        for index, (condition, seed) in enumerate(active_jobs, start=1):
            key = job_key(condition, seed)
            print(
                f"[Benchmark] Run {index}/{len(active_jobs)}: "
                f"{condition['town']} {condition['label']} seed={seed}"
            )
            try:
                wait_for_cot_idle(agent)
                agent.reset_run_state()

                row = run_one_condition(args, condition, seed, agent, runner)
            except RuntimeError as exc:
                if is_carla_server_error(exc):
                    print('[Benchmark] CARLA server connection lost or timed out.')
                    print(f'[Benchmark] Exiting with code {SERVER_DEATH_EXIT_CODE}: {exc}')
                    csv_handle.flush()
                    os.fsync(csv_handle.fileno())
                    return SERVER_DEATH_EXIT_CODE
                print(f'[Benchmark] Runtime error in run {key}: {exc}')
                traceback.print_exc()
                row = build_error_row(condition, seed, agent)
            except Exception as exc:
                print(f'[Benchmark] Non-CARLA error in run {key}: {exc}')
                traceback.print_exc()
                row = build_error_row(condition, seed, agent)

            writer.writerow(row)
            csv_handle.flush()
            os.fsync(csv_handle.fileno())
            completed_keys.add(key)
    finally:
        csv_handle.close()
        cleanup_headless(runner)
        agent.destroy()

    completed_count = len(completed_keys & all_job_keys)
    remaining_count = len(jobs) - completed_count
    print(f'[Benchmark] completed={completed_count} remaining={remaining_count}')
    if remaining_after_active_town > 0:
        print(
            f'[Benchmark] Finished {active_town}; '
            f'{remaining_after_active_town} runs remain in other towns.'
        )
        return EXIT_TOWN_DONE
    return 0


if __name__ == '__main__':
    sys.exit(main())

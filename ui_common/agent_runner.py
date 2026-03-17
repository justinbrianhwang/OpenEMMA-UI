"""
Simplified CARLA simulation runner for autonomous driving agents.

Replaces the leaderboard evaluator with a streamlined game loop
that integrates the chase camera UI and info panel.
"""

import copy
import math
import time
import threading
from queue import Queue, Empty

import carla
import numpy as np
import pygame

from .camera import ChaseCameraManager
from .panel import InfoPanel


# ─────────────────────────────────────────────
# Sensor data collection (simplified leaderboard SensorInterface)
# ─────────────────────────────────────────────

class SensorData:
    """Collects sensor data from CARLA via callbacks."""

    def __init__(self):
        self._sensors = {}
        self._data_queue = Queue()
        self._sensor_actors = []
        self._pseudo_sensors = []

    def setup_sensors(self, vehicle, sensor_specs, world):
        """Spawn all sensors defined by the agent and register callbacks."""
        bp_library = world.get_blueprint_library()

        for spec in sensor_specs:
            sensor_type = spec['type']
            sensor_id = spec['id']

            if sensor_type == 'sensor.speedometer':
                # Pseudo-sensor: read directly from vehicle
                self._pseudo_sensors.append(('speedometer', sensor_id, vehicle))
                self._sensors[sensor_id] = True
                continue

            if sensor_type == 'sensor.opendrive_map':
                self._pseudo_sensors.append(('opendrive', sensor_id, world))
                self._sensors[sensor_id] = True
                continue

            # Find blueprint
            bp = bp_library.find(sensor_type)

            # Set attributes based on sensor type
            if sensor_type.startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(spec['width']))
                bp.set_attribute('image_size_y', str(spec['height']))
                bp.set_attribute('fov', str(spec['fov']))
                if not sensor_type.endswith('semantic_segmentation') and not sensor_type.endswith('depth'):
                    bp.set_attribute('lens_circle_multiplier', '3.0')
                    bp.set_attribute('lens_circle_falloff', '3.0')
                    bp.set_attribute('chromatic_aberration_intensity', '0.5')
                    bp.set_attribute('chromatic_aberration_offset', '0')

            elif sensor_type.startswith('sensor.lidar'):
                bp.set_attribute('range', '85')
                bp.set_attribute('rotation_frequency', '10')
                bp.set_attribute('channels', '64')
                bp.set_attribute('upper_fov', '10')
                bp.set_attribute('lower_fov', '-30')
                bp.set_attribute('points_per_second', '600000')
                if not sensor_type.endswith('semantic'):
                    bp.set_attribute('atmosphere_attenuation_rate', '0.004')
                    bp.set_attribute('dropoff_general_rate', '0.45')
                    bp.set_attribute('dropoff_intensity_limit', '0.8')
                    bp.set_attribute('dropoff_zero_intensity', '0.4')

            elif sensor_type.startswith('sensor.other.radar'):
                bp.set_attribute('horizontal_fov', str(spec.get('fov', 30)))
                bp.set_attribute('vertical_fov', str(spec.get('fov', 30)))
                bp.set_attribute('points_per_second', '1500')
                bp.set_attribute('range', '100')

            elif sensor_type.startswith('sensor.other.gnss'):
                bp.set_attribute('noise_alt_bias', '0.0')
                bp.set_attribute('noise_lat_bias', '0.0')
                bp.set_attribute('noise_lon_bias', '0.0')

            elif sensor_type.startswith('sensor.other.imu'):
                bp.set_attribute('noise_accel_stddev_x', '0.001')
                bp.set_attribute('noise_accel_stddev_y', '0.001')
                bp.set_attribute('noise_accel_stddev_z', '0.015')
                bp.set_attribute('noise_gyro_stddev_x', '0.001')
                bp.set_attribute('noise_gyro_stddev_y', '0.001')
                bp.set_attribute('noise_gyro_stddev_z', '0.001')

            # Set sensor_tick if specified (common for GNSS/IMU)
            if 'sensor_tick' in spec and bp.has_attribute('sensor_tick'):
                bp.set_attribute('sensor_tick', str(spec['sensor_tick']))

            # Sensor transform
            loc = carla.Location(
                x=spec.get('x', 0), y=spec.get('y', 0), z=spec.get('z', 0))
            rot = carla.Rotation(
                pitch=spec.get('pitch', 0), roll=spec.get('roll', 0), yaw=spec.get('yaw', 0))
            transform = carla.Transform(loc, rot)

            sensor_actor = world.spawn_actor(bp, transform, attach_to=vehicle)
            self._sensor_actors.append(sensor_actor)
            self._sensors[sensor_id] = True

            # Register callback
            sid = sensor_id
            stype = sensor_type
            sensor_actor.listen(lambda data, _sid=sid, _stype=stype:
                                self._on_sensor_data(data, _sid, _stype))

    def _on_sensor_data(self, data, sensor_id, sensor_type):
        """Callback for CARLA sensor data."""
        if isinstance(data, carla.libcarla.Image):
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = copy.deepcopy(array)
            array = np.reshape(array, (data.height, data.width, 4))
            self._data_queue.put((sensor_id, data.frame, array))

        elif isinstance(data, carla.libcarla.LidarMeasurement):
            points = np.frombuffer(data.raw_data, dtype=np.float32)
            points = copy.deepcopy(points)
            points = np.reshape(points, (-1, 4))
            self._data_queue.put((sensor_id, data.frame, points))

        elif isinstance(data, carla.libcarla.SemanticLidarMeasurement):
            points = np.frombuffer(data.raw_data, dtype=np.float32)
            points = copy.deepcopy(points)
            points = np.reshape(points, (-1, 6))
            self._data_queue.put((sensor_id, data.frame, points))

        elif isinstance(data, carla.libcarla.RadarMeasurement):
            points = np.frombuffer(data.raw_data, dtype=np.float32)
            points = copy.deepcopy(points)
            points = np.reshape(points, (-1, 4))
            points = np.flip(points, 1)
            self._data_queue.put((sensor_id, data.frame, points))

        elif isinstance(data, carla.libcarla.GnssMeasurement):
            array = np.array([data.latitude, data.longitude, data.altitude],
                             dtype=np.float64)
            self._data_queue.put((sensor_id, data.frame, array))

        elif isinstance(data, carla.libcarla.IMUMeasurement):
            array = np.array([
                data.accelerometer.x, data.accelerometer.y, data.accelerometer.z,
                data.gyroscope.x, data.gyroscope.y, data.gyroscope.z,
                data.compass
            ], dtype=np.float64)
            self._data_queue.put((sensor_id, data.frame, array))

    def get_data(self, frame, vehicle, timeout=3.0):
        """Collect data from all sensors for the current frame.

        Drains stale data from previous frames and only accepts data
        matching the current frame (or the latest available for sensors
        with sensor_tick that may not fire every frame).
        """
        data_dict = {}

        # Add pseudo-sensor data
        for ptype, pid, obj in self._pseudo_sensors:
            if ptype == 'speedometer':
                vel = obj.get_velocity()
                transform = obj.get_transform()
                vel_np = np.array([vel.x, vel.y, vel.z])
                pitch = np.deg2rad(transform.rotation.pitch)
                yaw = np.deg2rad(transform.rotation.yaw)
                orientation = np.array([
                    np.cos(pitch) * np.cos(yaw),
                    np.cos(pitch) * np.sin(yaw),
                    np.sin(pitch)
                ])
                speed = np.dot(vel_np, orientation)
                data_dict[pid] = (frame, {'speed': speed})
            elif ptype == 'opendrive':
                data_dict[pid] = (frame, {'opendrive': obj.get_map().to_opendrive()})

        # Collect real sensor data
        real_sensor_count = len(self._sensors) - len(self._pseudo_sensors)
        collected = 0
        start_time = time.time()

        while collected < real_sensor_count:
            if time.time() - start_time > timeout:
                # Find which sensors are missing
                all_ids = set(self._sensors.keys())
                pseudo_ids = {pid for _, pid, _ in self._pseudo_sensors}
                expected_real = all_ids - pseudo_ids
                got_real = set(data_dict.keys()) - pseudo_ids
                missing = expected_real - got_real
                print(f'[WARNING] Sensor timeout. Got {collected}/{real_sensor_count} sensors. Missing: {missing}')
                break
            try:
                sensor_id, data_frame, data = self._data_queue.get(timeout=1.0)
                # Skip stale data from previous frames
                if data_frame < frame:
                    continue
                # Only count as new collection if this sensor_id wasn't seen yet
                is_new = sensor_id not in data_dict
                data_dict[sensor_id] = (data_frame, data)
                if is_new:
                    collected += 1
            except Empty:
                continue

        return data_dict

    def cleanup(self):
        """Destroy all sensor actors."""
        for sensor in self._sensor_actors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        self._sensor_actors.clear()
        self._pseudo_sensors.clear()
        self._sensors.clear()


# ─────────────────────────────────────────────
# Safety Limiter
# ─────────────────────────────────────────────

class SafetyLimiter:
    """
    Enforces traffic rules and safety limits.

    - Speed limits at junctions and normal roads
    - Red traffic light detection and braking
    - Sidewalk invasion prevention
    """

    def __init__(self, max_speed_normal=40.0, max_speed_junction=30.0,
                 soft_limit_normal=35.0, soft_limit_junction=20.0):
        self.max_speed_normal = max_speed_normal
        self.max_speed_junction = max_speed_junction
        self.soft_limit_normal = soft_limit_normal
        self.soft_limit_junction = soft_limit_junction
        self.route = None  # list of (carla.Transform, RoadOption)

    def set_route(self, world_route):
        """Store the route for junction steering assist."""
        self.route = world_route

    def _get_route_steer(self, vehicle):
        """Compute ideal steer from route waypoints ahead of the vehicle.

        Finds the closest route point, then looks ~8m ahead along the route
        to get the target direction. Returns (steer_value, is_valid).
        """
        if not self.route:
            return 0.0, False

        veh_loc = vehicle.get_location()
        veh_transform = vehicle.get_transform()
        veh_fwd = veh_transform.get_forward_vector()

        # Find closest route waypoint
        min_dist = float('inf')
        closest_idx = 0
        for i, (transform, _) in enumerate(self.route):
            loc = transform.location
            d = (loc.x - veh_loc.x)**2 + (loc.y - veh_loc.y)**2
            if d < min_dist:
                min_dist = d
                closest_idx = i

        # Look ahead ~8m along the route (route is sampled at ~1m)
        look_idx = min(closest_idx + 8, len(self.route) - 1)
        target_loc = self.route[look_idx][0].location

        dx = target_loc.x - veh_loc.x
        dy = target_loc.y - veh_loc.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.5:
            return 0.0, False

        # Cross product: positive → target is to the left
        cross = veh_fwd.x * dy - veh_fwd.y * dx
        # Dot product for forward check
        dot = veh_fwd.x * dx + veh_fwd.y * dy

        # Compute steer angle (atan2-based, normalized to [-1, 1])
        steer = math.atan2(cross, dot) / (math.pi / 2.0)
        steer = max(-1.0, min(1.0, steer))

        return steer, True

    def apply(self, control, vehicle, carla_map):
        """Apply safety rules: speed limits, red lights, lane keeping.

        Also sets self.violations dict for debug logging.
        """
        vel = vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        self.violations = {'red_light': False, 'off_road': False, 'wrong_lane': False}

        # ── Red light detection ──
        is_red_light = False
        traffic_light = vehicle.get_traffic_light()
        if traffic_light is not None:
            state = traffic_light.get_state()
            if state == carla.TrafficLightState.Red:
                self.violations['red_light'] = True
                is_red_light = True
                control.throttle = 0.0
                control.brake = 1.0
            elif state == carla.TrafficLightState.Yellow and speed_kmh > 10:
                control.throttle = 0.0
                control.brake = 0.5

        # ── Lane-keeping: preemptive + reactive sidewalk correction ──
        # (skip lane-keeping logic when stopped at red light)
        if is_red_light:
            return control

        veh_loc = vehicle.get_location()
        veh_transform = vehicle.get_transform()
        veh_fwd = veh_transform.get_forward_vector()
        # Right vector (perpendicular to forward, in 2D)
        veh_right_x = veh_fwd.y
        veh_right_y = -veh_fwd.x

        wp = carla_map.get_waypoint(veh_loc, project_to_road=False)
        is_off_road = (wp is None or wp.lane_type != carla.LaneType.Driving)

        # Also check car edges (±1m from center) for width-aware detection
        if not is_off_road:
            for side in (1.0, -1.0):
                edge_loc = carla.Location(
                    x=veh_loc.x + veh_right_x * side,
                    y=veh_loc.y + veh_right_y * side,
                    z=veh_loc.z)
                edge_wp = carla_map.get_waypoint(edge_loc, project_to_road=False)
                if edge_wp is None or edge_wp.lane_type != carla.LaneType.Driving:
                    is_off_road = True
                    break

        if is_off_road:
            self.violations['off_road'] = True

        # Always get the nearest driving-lane waypoint for lane-keeping
        road_wp = carla_map.get_waypoint(veh_loc, project_to_road=True,
                                         lane_type=carla.LaneType.Driving)

        if road_wp:
            road_loc = road_wp.transform.location
            dx = road_loc.x - veh_loc.x
            dy = road_loc.y - veh_loc.y
            dist_to_center = math.sqrt(dx*dx + dy*dy)

            # Cross product for steering direction
            cross = veh_fwd.x * dy - veh_fwd.y * dx

            # Also check ahead: will the car leave the road in ~1 second?
            speed_ms = math.sqrt(vel.x**2 + vel.y**2) + 0.01
            look_ahead = max(3.0, speed_ms * 1.5)  # 1.5s look-ahead, min 3m
            ahead_loc = carla.Location(
                x=veh_loc.x + veh_fwd.x * look_ahead,
                y=veh_loc.y + veh_fwd.y * look_ahead,
                z=veh_loc.z)
            ahead_wp = carla_map.get_waypoint(ahead_loc, project_to_road=False)
            will_leave_road = (ahead_wp is None or
                               ahead_wp.lane_type != carla.LaneType.Driving)

            # Get lane half-width for proximity detection
            lane_hw = road_wp.lane_width / 2.0 if road_wp.lane_width else 1.75

            if is_off_road:
                # REACTIVE: Already off-road — strong correction
                if dist_to_center > 0.1:
                    correction = max(-0.8, min(0.8,
                                    cross / max(dist_to_center, 0.5) * 5.0))
                    control.steer = correction
                control.throttle = min(control.throttle, 0.2)
                control.brake = max(control.brake, 0.3)

            elif will_leave_road or dist_to_center > lane_hw * 0.35:
                # PREEMPTIVE: Near lane edge or heading off-road
                # Gentle correction toward lane center (blended with model steer)
                if dist_to_center > 0.1:
                    # Strength scales with how close to edge
                    edge_ratio = min(1.0, dist_to_center / lane_hw)
                    gain = 2.0 + edge_ratio * 3.0  # 2.0 near center, 5.0 at edge
                    correction = max(-0.5, min(0.5,
                                    cross / max(dist_to_center, 0.5) * gain))
                    # Blend: keep some of model's steer, add correction
                    blend = min(0.8, edge_ratio)  # More override near edge
                    control.steer = (1.0 - blend) * control.steer + blend * correction
                if will_leave_road:
                    control.throttle = min(control.throttle, 0.4)

        # ── Junction steering assist ──
        # ONLY inside junctions (not near), to avoid over-steer at exit.
        if wp is None:
            wp = carla_map.get_waypoint(vehicle.get_location())
        is_junction = wp.is_junction if wp else False

        # Check ahead for speed limit purposes
        near_junction = is_junction
        if not near_junction and wp:
            ahead_wps = wp.next(8.0)
            if ahead_wps and ahead_wps[0].is_junction:
                near_junction = True

        if is_junction and not is_off_road:
            route_steer, valid = self._get_route_steer(vehicle)
            if valid:
                model_steer = control.steer
                steer_diff = abs(route_steer - model_steer)
                if steer_diff > 0.15:
                    # Blend: 50% model + 50% route inside junction
                    blend = 0.5
                    control.steer = (1.0 - blend) * model_steer + blend * route_steer

        # ── Wrong lane detection (driving against traffic) ──
        if wp and wp.lane_type == carla.LaneType.Driving:
            veh_yaw = math.radians(veh_transform.rotation.yaw)
            lane_yaw = math.radians(wp.transform.rotation.yaw)
            diff = abs(veh_yaw - lane_yaw)
            diff = min(diff, 2*math.pi - diff)
            if diff > math.pi * 0.6:  # More than ~108° off → wrong way
                self.violations['wrong_lane'] = True

        # ── Speed limits ──
        # Use near_junction (includes 8m ahead check from above)
        if near_junction:
            max_speed = self.max_speed_junction
            soft_limit = self.soft_limit_junction
        else:
            max_speed = self.max_speed_normal
            soft_limit = self.soft_limit_normal

        # Gradually reduce throttle above soft limit
        if speed_kmh > soft_limit:
            over_ratio = (speed_kmh - soft_limit) / (max_speed - soft_limit + 0.01)
            over_ratio = min(over_ratio, 1.0)
            control.throttle = control.throttle * (1.0 - over_ratio)

        # Force brake if above hard limit
        if speed_kmh > max_speed:
            control.throttle = 0.0
            control.brake = max(control.brake, 0.5)

        return control


# ─────────────────────────────────────────────
# Agent Runner
# ─────────────────────────────────────────────

class AgentRunner:
    """
    Manages the full CARLA simulation loop with an autonomous agent.

    Handles:
    - CARLA connection and world setup
    - Ego vehicle spawning (no NPC traffic for baseline)
    - Agent sensor setup
    - Route generation
    - Safety speed limiting
    - Game loop with chase camera UI
    """

    def __init__(self, host='localhost', port=2000, town='Town01',
                 camera_width=1280, camera_height=720,
                 model_name='Agent', target_fps=20):
        self.host = host
        self.port = port
        self.town = town
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.model_name = model_name
        self.target_fps = target_fps

        # Will be initialized
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.sensor_data = None
        self.chase_camera = None
        self.panel = None
        self.display = None
        self.clock = None
        self.safety_limiter = SafetyLimiter()

    def setup(self):
        """Initialize CARLA world and spawn ego vehicle."""
        print(f'[Runner] Connecting to CARLA at {self.host}:{self.port}...')
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(30.0)

        # Load town
        print(f'[Runner] Loading {self.town}...')
        self.world = self.client.load_world(self.town)

        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.target_fps
        self.world.apply_settings(settings)

        # Traffic manager (needed even without NPCs for sync mode)
        tm = self.client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        # Spawn ego vehicle
        self._spawn_ego_vehicle()

        # Tick once to initialize everything
        self.world.tick()

        print(f'[Runner] Setup complete. Ego vehicle: {self.ego_vehicle.type_id}')

    def _spawn_ego_vehicle(self):
        """Spawn the ego vehicle at the first available spawn point."""
        bp_lib = self.world.get_blueprint_library()

        # Try Tesla Model 3 first, fall back to any 4-wheel vehicle
        candidates = bp_lib.filter('vehicle.tesla.model3')
        if not candidates:
            candidates = bp_lib.filter('vehicle.tesla.*')
        if not candidates:
            candidates = [bp for bp in bp_lib.filter('vehicle.*')
                          if int(bp.get_attribute('number_of_wheels')) == 4]
        if not candidates:
            raise RuntimeError('No suitable vehicle blueprint found!')

        vehicle_bp = candidates[0]
        vehicle_bp.set_attribute('role_name', 'hero')

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError('No spawn points found in the map!')

        # Use first spawn point for ego
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        print(f'[Runner] Spawned ego vehicle at {spawn_points[0].location}')

    def generate_route(self, start_location=None, end_location=None):
        """Generate a route using CARLA's global route planner.

        Returns list of (carla.Transform, RoadOption) for the route.
        Also returns GPS-format plan for agents that need it.
        """
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=1.0)

        spawn_points = self.world.get_map().get_spawn_points()

        if start_location is None:
            start_location = self.ego_vehicle.get_transform().location
        if end_location is None:
            # Pick a distant spawn point as destination
            distances = []
            for sp in spawn_points:
                d = start_location.distance(sp.location)
                distances.append((d, sp))
            distances.sort(key=lambda x: -x[0])
            end_location = distances[0][1].location

        route = grp.trace_route(start_location, end_location)

        # Convert to GPS format (lat, lon) + world coord format
        carla_map = self.world.get_map()
        gps_route = []
        world_route = []

        for waypoint, road_option in route:
            transform = waypoint.transform
            loc = transform.location
            # CARLA GPS conversion (simplified - matches leaderboard convention)
            gps_point = {
                'lat': loc.x / 111324.60662786,
                'lon': loc.y / 111319.490945,
                'z': loc.z
            }
            gps_route.append((gps_point, road_option))
            world_route.append((transform, road_option))

        print(f'[Runner] Generated route with {len(route)} waypoints')
        return gps_route, world_route

    def setup_agent_sensors(self, agent):
        """Set up sensors defined by the agent."""
        self.sensor_data = SensorData()
        sensor_specs = agent.sensors()
        self.sensor_data.setup_sensors(self.ego_vehicle, sensor_specs, self.world)
        self.world.tick()
        print(f'[Runner] Set up {len(sensor_specs)} agent sensors')

    def setup_ui(self):
        """Initialize Pygame UI with chase camera and info panel."""
        pygame.init()

        total_width = InfoPanel.PANEL_WIDTH + self.camera_width
        self.display = pygame.display.set_mode(
            (total_width, self.camera_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption(f'{self.model_name} - CARLA Monitor')

        self.panel = InfoPanel(self.camera_height, model_name=self.model_name)
        self.chase_camera = ChaseCameraManager(
            self.ego_vehicle, self.camera_width, self.camera_height
        )
        self.clock = pygame.time.Clock()
        print('[Runner] UI initialized')

    def run_loop(self, agent, on_step=None):
        """Main simulation loop.

        Agent inference runs in a background thread. The simulation
        pauses (no world.tick()) while inference is running, so the
        vehicle doesn't move with stale control. The UI keeps
        rendering the last camera frame so the window stays responsive.

        Args:
            agent: The autonomous agent (must have run_step(input_data, timestamp))
            on_step: Optional callback(tick_data, control) called each step
        """
        step = 0
        running = True
        carla_map = self.world.get_map()

        # Agent inference state
        _agent_thread = None
        _latest_ctrl = {'throttle': 0.0, 'steer': 0.0, 'brake': 0.0}
        _control_lock = threading.Lock()
        _inference_done = threading.Event()
        _inference_done.set()  # initially ready

        def _run_agent_async(input_data, timestamp):
            """Run agent.run_step in background, store result."""
            nonlocal _latest_ctrl
            try:
                ctrl = agent.run_step(input_data, timestamp)
                if ctrl is not None:
                    with _control_lock:
                        _latest_ctrl = {
                            'throttle': ctrl.throttle,
                            'steer': ctrl.steer,
                            'brake': ctrl.brake,
                            'hand_brake': ctrl.hand_brake,
                            'reverse': ctrl.reverse,
                        }
            except Exception as e:
                import traceback
                print(f'\n{"="*60}')
                print(f'[Runner] AGENT ERROR at step: {e}')
                traceback.print_exc()
                print(f'{"="*60}\n')
            finally:
                _inference_done.set()

        print('[Runner] Starting simulation loop. Press ESC to quit.')

        try:
            while running:
                # Pygame events (always responsive)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print('[Runner] Window closed by user.')
                        running = False
                    elif event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                        print('[Runner] ESC pressed.')
                        running = False

                if not running:
                    break

                # If agent inference is still running, just refresh UI
                if not _inference_done.is_set():
                    self._render_ui(agent)
                    self.clock.tick(self.target_fps)
                    continue

                # ── Inference is done → advance simulation ──
                frame = self.world.tick()
                timestamp = self.world.get_snapshot().timestamp.elapsed_seconds

                # Check ego vehicle
                try:
                    if not self.ego_vehicle.is_alive:
                        print('[Runner] Ego vehicle destroyed. Stopping.')
                        break
                except RuntimeError:
                    print('[Runner] Ego vehicle no longer valid. Stopping.')
                    break

                # Apply latest control from agent
                with _control_lock:
                    control = carla.VehicleControl()
                    control.throttle = _latest_ctrl['throttle']
                    control.steer = _latest_ctrl['steer']
                    control.brake = _latest_ctrl['brake']
                    control.hand_brake = _latest_ctrl.get('hand_brake', False)
                    control.reverse = _latest_ctrl.get('reverse', False)
                    control.manual_gear_shift = False

                # Apply safety rules (red lights, sidewalk, speed limits)
                control = self.safety_limiter.apply(
                    control, self.ego_vehicle, carla_map)

                # Share violation info with agent for debug logging
                if hasattr(agent, 'ui_violations'):
                    agent.ui_violations = getattr(self.safety_limiter, 'violations', {})
                # Share red light state with agent so stall recovery doesn't fight it
                if hasattr(agent, 'ui_red_light'):
                    agent.ui_red_light = self.safety_limiter.violations.get('red_light', False)

                # Reset stuck detector when stopped at red light (prevent false force-move)
                is_red = self.safety_limiter.violations.get('red_light', False)
                fmp = getattr(agent, 'force_move_post_processor', None)
                if is_red and fmp is not None:
                    fmp.stuck_detector = 0
                    fmp.force_move = 0

                # Also share red light notice with agent for UI display
                if is_red:
                    agent.curr_notice = "Red light ahead."

                self.ego_vehicle.apply_control(control)

                # Optional callback
                if on_step:
                    on_step(None, control)

                # Collect sensor data for next inference
                # Process pygame events during collection to keep window responsive
                sensor_timeout = 3.0 if step < 5 else 0.5
                input_data = self.sensor_data.get_data(frame, self.ego_vehicle, timeout=sensor_timeout)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print('[Runner] Window closed by user.')
                        running = False
                    elif event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                        print('[Runner] ESC pressed.')
                        running = False
                if not running:
                    break

                # Launch agent inference in background
                _inference_done.clear()
                _agent_thread = threading.Thread(
                    target=_run_agent_async,
                    args=(input_data, timestamp),
                    daemon=True
                )
                _agent_thread.start()

                # Render UI
                self._render_ui(agent)
                self.clock.tick(self.target_fps)
                step += 1

        except KeyboardInterrupt:
            print('\n[Runner] Interrupted by user.')

        if _agent_thread is not None and _agent_thread.is_alive():
            _agent_thread.join(timeout=10.0)

    def _render_ui(self, agent=None):
        """Render chase camera and info panel. Called every frame."""
        self.display.fill((0, 0, 0))

        cam_surface = self.chase_camera.get_surface()
        if cam_surface:
            self.display.blit(cam_surface, (InfoPanel.PANEL_WIDTH, 0))

        vel = self.ego_vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        ctrl = self.ego_vehicle.get_control()
        if ctrl.brake > 0:
            brake_status = 'BRAKING'
        elif ctrl.throttle > 0:
            brake_status = 'ACCELERATING'
        else:
            brake_status = 'IDLE'

        vehicle_count = len(self.world.get_actors().filter('vehicle.*'))
        pedestrian_count = len(self.world.get_actors().filter('walker.pedestrian.*'))

        # Collect LLM info from agent if available
        llm_info = None
        if agent is not None:
            llm_info = {
                'instruction': getattr(agent, 'curr_instruction', ''),
                'notice': getattr(agent, 'curr_notice', ''),
                'waypoints': getattr(agent, 'ui_waypoints', None),
                'desired_speed': getattr(agent, 'ui_desired_speed', 0.0),
                'curvature': getattr(agent, 'ui_curvature', 0.0),
                'steer': ctrl.steer,
            }

        self.panel.render(
            self.display,
            vehicle_count, pedestrian_count,
            speed_kmh, brake_status,
            llm_info=llm_info
        )

        pygame.display.flip()

    def cleanup(self):
        """Clean up all actors and reset world settings."""
        print('[Runner] Cleaning up...')

        # Chase camera
        if self.chase_camera:
            self.chase_camera.destroy()

        # Agent sensors
        if self.sensor_data:
            self.sensor_data.cleanup()

        # Ego vehicle
        if self.ego_vehicle:
            try:
                self.ego_vehicle.destroy()
            except Exception:
                pass

        # Reset world settings
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        # Pygame
        pygame.quit()
        print('[Runner] Cleanup complete.')

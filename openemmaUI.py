"""
OpenEMMA - CARLA Autonomous Driving UI

Adapts OpenEMMA (originally designed for nuScenes) to work with CARLA 0.9.16.
Uses a VLM (Qwen2-VL, LLaVA, or GPT-4o) for chain-of-thought driving decisions.

Architecture:
    - VLM predicts future speed/curvature actions that are integrated to waypoints
    - Per-frame controller follows the latest VLM trajectory
    - SafetyLimiter handles red lights, lane-keeping (in agent_runner)

Pipeline: Scene Description -> Critical Objects -> Driving Intent -> Motion Prediction
The full CoT reasoning is displayed in the UI panel.

Usage:
    1. Start CARLA server (CarlaUE4.exe)
    2. Run: python openemmaUI.py --llava
       Or:  python openemmaUI.py --town Town02 --llava
       Or:  python openemmaUI.py --model-path /path/to/model

    Set OPENAI_API_KEY environment variable to use GPT-4o backend.

Requirements:
    - CARLA 0.9.16 PythonAPI
    - OpenEMMA dependencies (transformers, torch, etc.)
    - VLM model weights or OpenAI API key (OPENAI_API_KEY) for GPT-4o
"""

import os
import sys
import math
import argparse
import re
import threading

# CARLA 0.9.16 PythonAPI setup (must be first)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui_common'))
from carla_setup import setup_carla_paths
setup_carla_paths()

import carla
import numpy as np

# Path setup for OpenEMMA
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OPENEMMA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'OpenEMMA'))
if not os.path.isdir(OPENEMMA_ROOT):
    OPENEMMA_ROOT = os.path.join(PROJECT_ROOT, 'OpenEMMA')
if OPENEMMA_ROOT not in sys.path:
    sys.path.insert(0, OPENEMMA_ROOT)

from ui_common.agent_runner import AgentRunner
from ui_common.trajectory import (
    EstimateCurvatureFromTrajectory,
    IntegrateCurvatureForPoints,
)

# Default local model paths (in OpenEMMA/local/)
LOCAL_MODEL_DIR = os.path.join(OPENEMMA_ROOT, 'local')
LOCAL_MODELS = {
    'qwen': os.path.join(LOCAL_MODEL_DIR, 'Qwen2-VL-7B-Instruct'),
    'llava': os.path.join(LOCAL_MODEL_DIR, 'llava-v1.5-7b'),
    'llama': os.path.join(LOCAL_MODEL_DIR, 'Llama-3.2-11B-Vision-Instruct'),
}


# Chain-of-Thought Prompt Templates

SCENE_PROMPT = (
    "You are an autonomous vehicle driving in a virtual city (CARLA simulator). "
    "Describe the driving scene in 1-2 sentences: traffic lights (color?), "
    "other vehicles, pedestrians, lane markings, and road geometry (straight/curve/intersection)."
)

OBJECTS_PROMPT = (
    "Based on the driving scene, list 1-3 critical objects the ego car should "
    "focus on (e.g., red traffic light ahead, car braking in front, pedestrian crossing). "
    "For each, briefly explain why it matters. Be concise."
)

INTENT_PROMPT_TEMPLATE = (
    "You are driving at {speed:.1f} m/s. "
    "Scene: {scene}\n"
    "Critical objects: {objects}\n"
    "{prev_intent_str}"
    "What should the ego car do? Answer in one line: "
    "turn left / turn right / go straight AND speed up / maintain speed / slow down / stop."
)

MOTION_SYSTEM = (
    "You are a autonomous driving labeller. You have access to a front-view "
    "camera image of a vehicle, a sequence of past speeds, a sequence of past "
    "curvatures, and a driving rationale. Each speed, curvature is represented "
    "as [v, k], where v corresponds to the speed, and k corresponds to the "
    "curvature. A positive k means the vehicle is turning left. A negative k "
    "means the vehicle is turning right. The larger the absolute value of k, "
    "the sharper the turn. A close to zero k means the vehicle is driving "
    "straight. As a driver on the road, you should follow any common sense "
    "traffic rules. You should try to stay in the middle of your lane. You "
    "should maintain necessary distance from the leading vehicle. You should "
    "observe lane markings and follow them. Your task is to do your best to "
    "predict future speeds and curvatures for the vehicle over the next 10 "
    "timesteps given vehicle intent inferred from the image. Make a best guess "
    "if the problem is too difficult for you."
)

MOTION_PROMPT_TEMPLATE = (
    "These are frames from a video taken by a camera mounted in the front of a "
    "car. The images are taken at a 0.5 second interval.\n"
    "The scene is described as follows: {scene}.\n"
    "The identified critical objects are {objects}.\n"
    "The car's intent is {intent}.\n"
    "The 5 second historical velocities and curvatures of the ego car are "
    "{history}.\n"
    "Infer the association between these numbers and the image sequence. "
    "Generate the predicted future speeds and curvatures in the format "
    "[speed_1, curvature_1], [speed_2, curvature_2],..., "
    "[speed_10, curvature_10]. Write the raw text not markdown or latex. "
    "Future speeds and curvatures:"
)


# OpenEMMA CARLA Adapter

class OpenEMMACarlaAgent:
    """
    Adapts OpenEMMA's VLM-based driving pipeline to work as a CARLA agent.

    Control Strategy:
    - PRIMARY: VLM-predicted speed/curvature actions integrated to a trajectory
    - FALLBACK: Route pure-pursuit before the first valid VLM trajectory
    - SafetyLimiter handles actual red lights and lane-keeping

    Full Chain-of-Thought pipeline (runs every N frames):
    1. Scene Description - what's in the driving scene
    2. Critical Objects - what to pay attention to
    3. Driving Intent - what should the car do
    4. Motion Prediction - 10 future speed/curvature actions at 0.5s spacing
    """

    OBS_LEN = 10
    FUT_LEN = 10
    FRAME_DT = 0.5
    HISTORY_SAMPLE_FRAMES = 10

    # Speed parameters
    CRUISE_SPEED = 4.5      # m/s (~16 km/h) - default target
    MAX_SPEED = 5.5         # m/s (~20 km/h) - max allowed
    MIN_DRIVE_SPEED = 4.5   # m/s - standing-start floor
    TURN_SPEED = 2.5        # m/s (~9 km/h) - speed in sharp turns
    CURVE_SPEED = 3.8       # m/s (~14 km/h) - moderate curves
    SANE_MAX_SPEED_MPS = 8.0
    SANE_MAX_CURV = 0.3
    # Route steering (dual look-ahead)
    LOOK_AHEAD_NEAR = 12    # near target for lane centering
    LOOK_AHEAD_FAR = 22     # far target for turn anticipation
    LOOK_AHEAD_M = 4.0      # VLM trajectory look-ahead in ego-local meters
    STEER_GAIN = 1.3        # steering responsiveness
    OFFROAD_RECOVERY_DIST_M = 2.5
    OFFROAD_RECOVERY_FRAMES = 8

    @staticmethod
    def _fresh_metrics():
        return {
            'frames': 0,
            'vlm_frames': 0,
            'fallback_frames': 0,
            'speed_floor_frames': 0,
            'speed_clamp_events': 0,
            'curv_clamp_events': 0,
            'degenerate_rejections': 0,
            'stuck_events': 0,
            'recovery_events': 0,
            'route_regens': 0,
            'cot_cycles': 0,
            'cot_stop_or_red': 0,
            'steer_sum': 0.0,
            'steer_sq_sum': 0.0,
            'speed_sum': 0.0,
        }

    def __init__(self, model_path, device='cuda', use_4bit=False, debug=False):
        self.model_path = model_path
        self.use_4bit = use_4bit
        self.device = device
        self.debug = debug

        # State
        self.step = 0
        self.route = None  # List of (carla.Transform, RoadOption)
        self._route_idx = 0  # Track progress along route

        # CoT state (persists across frames)
        self.prev_intent = ''
        self.last_scene = ''
        self.last_objects = ''
        self.last_intent = ''
        self.last_motion_raw = ''

        # Ego history sampled every 0.5s. Speeds are per-frame displacement
        # magnitudes in meters, matching OpenEMMA's OBS_LEN/FUT_LEN convention.
        self._ego_history_positions = []
        self.obs_positions = None
        self.obs_velocities = None
        self.obs_vel_norm = None
        self.obs_curvatures = None
        self.obs_initial_heading = 0.0
        self.obs_sample_count = 0

        # VLM trajectory in ego-local coordinates: +x forward, +y left.
        self.vlm_traj = None
        self.vlm_pred_speeds = None  # m/s, converted from 0.5s displacement
        self.vlm_pred_curvatures = None

        # Async VLM inference
        self._cot_running = False
        self._cot_lock = threading.Lock()

        # Control smoothing
        self.prev_steer = 0.0

        # Stuck recovery
        self._stuck_counter = 0
        self._offroad_counter = 0
        self._recovery_counter = 0
        self._realign_counter = 0
        self._recovery_steer = 0.0
        self._stuck_recovery_events = 0
        self._last_recovery_route_idx = None
        self._last_vlm_dbg_source = None

        # UI attributes (read by agent_runner._render_ui)
        self.curr_instruction = ''
        self.curr_notice = ''
        self.ui_waypoints = None
        self.ui_desired_speed = 0.0
        self.ui_curvature = 0.0
        # Must be initialized so agent_runner can update via hasattr() check
        self.ui_red_light = False
        self.ui_violations = {}

        # Cheap always-on counters used by the headless benchmark harness.
        self.metrics = self._fresh_metrics()

        # VLM components
        self.model = None
        self.processor = None
        self.tokenizer = None

        self._load_model()

    def reset_run_state(self):
        """Reset per-run state while keeping the loaded model in memory."""
        self.step = 0
        self.route = None
        self._route_idx = 0

        self.prev_intent = ''
        self.last_scene = ''
        self.last_objects = ''
        self.last_intent = ''
        self.last_motion_raw = ''

        self._ego_history_positions = []
        with self._cot_lock:
            self._cot_running = False
            self.obs_positions = None
            self.obs_velocities = None
            self.obs_vel_norm = None
            self.obs_curvatures = None
            self.obs_initial_heading = 0.0
            self.obs_sample_count = 0
            self.vlm_traj = None
            self.vlm_pred_speeds = None
            self.vlm_pred_curvatures = None

        self.prev_steer = 0.0
        self._stuck_counter = 0
        self._offroad_counter = 0
        self._recovery_counter = 0
        self._realign_counter = 0
        self._recovery_steer = 0.0
        self._stuck_recovery_events = 0
        self._last_recovery_route_idx = None
        self._last_vlm_dbg_source = None

        self.curr_instruction = ''
        self.curr_notice = ''
        self.ui_waypoints = None
        self.ui_desired_speed = 0.0
        self.ui_curvature = 0.0
        self.ui_red_light = False
        self.ui_violations = {}

        self.metrics = self._fresh_metrics()

    def get_metrics_snapshot(self):
        """Return benchmark counters plus route progress."""
        snapshot = self.metrics.copy()
        snapshot['route_idx'] = self._route_idx
        snapshot['route_len'] = len(self.route) if self.route else 0
        return snapshot

    def _record_frame_metrics(self, steer, current_speed):
        self.metrics['frames'] += 1
        self.metrics['steer_sum'] += float(steer)
        self.metrics['steer_sq_sum'] += float(steer) * float(steer)
        self.metrics['speed_sum'] += abs(float(current_speed))

    def set_route(self, world_route):
        """Set the route for pure-pursuit steering."""
        self.route = world_route
        self._route_idx = 0
        self._stuck_recovery_events = 0
        self._last_recovery_route_idx = None

    def set_runner(self, runner):
        """Store runner reference for route regeneration."""
        self._runner = runner

    def _regenerate_route(self, count_metric=False):
        """Generate a new random route from current position."""
        runner = getattr(self, '_runner', None)
        vehicle = getattr(self, '_vehicle', None)
        if runner is None or vehicle is None:
            return
        import random
        spawn_points = runner.world.get_map().get_spawn_points()
        current_loc = vehicle.get_location()
        # Pick a random distant destination (>100m away)
        candidates = [sp for sp in spawn_points
                      if current_loc.distance(sp.location) > 100]
        if not candidates:
            candidates = spawn_points
        dest = random.choice(candidates)
        _, world_route = runner.generate_route(
            start_location=current_loc, end_location=dest.location)
        self.set_route(world_route)
        runner.safety_limiter.set_route(world_route)
        if count_metric:
            self.metrics['route_regens'] += 1
        print(f'[OpenEMMA] New route: {len(world_route)} waypoints')

    def _load_model(self):
        """Load the VLM model based on model_path."""
        import torch

        model_path = self.model_path
        load_kwargs = {'device_map': 'auto'}

        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
            )
            mode_str = '4-bit'
        else:
            load_kwargs['torch_dtype'] = torch.float16
            mode_str = 'fp16'

        if 'gpt' in model_path.lower():
            from openai import OpenAI
            self.model_type = 'gpt'
            self.client = OpenAI()
            print('[OpenEMMA] Using GPT-4o via API')

        elif 'qwen' in model_path.lower():
            self.model_type = 'qwen'
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            print(f'[OpenEMMA] Loaded Qwen2-VL ({mode_str}) from {model_path}')

        elif 'llava' in model_path.lower():
            self.model_type = 'llava'
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            print(f'[OpenEMMA] Loaded LLaVA ({mode_str}) from {model_path}')

        elif 'llama' in model_path.lower():
            self.model_type = 'llama'
            from transformers import MllamaForConditionalGeneration, AutoProcessor
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            print(f'[OpenEMMA] Loaded Llama-Vision ({mode_str}) from {model_path}')

        else:
            raise ValueError(f'Unknown model type: {model_path}. '
                             'Supported: gpt, qwen, llava, llama')

    def _vlm_query(self, prompt, image_path):
        """Send a query to the VLM with an image."""
        import torch

        if self.model_type == 'gpt':
            import base64
            with open(image_path, 'rb') as f:
                b64_image = base64.b64encode(f.read()).decode('utf-8')
            response = self.client.chat.completions.create(
                model='gpt-4o',
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'image_url',
                         'image_url': {'url': f'data:image/jpeg;base64,{b64_image}'}},
                        {'type': 'text', 'text': prompt}
                    ]
                }],
                max_tokens=200
            )
            return response.choices[0].message.content

        elif self.model_type == 'qwen':
            from qwen_vl_utils import process_vision_info
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image_path},
                    {'type': 'text', 'text': prompt}
                ]
            }]
            text = self.processor.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=200)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        elif self.model_type == 'llava':
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            # LLaVA v1.5 prompt format (no apply_chat_template in transformers 4.36)
            text = f"<image>\nUSER: {prompt}\nASSISTANT:"
            inputs = self.processor(images=image, text=text,
                                    return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=200)
            # Decode only new tokens (skip input)
            generated = output_ids[0][inputs['input_ids'].shape[1]:]
            return self.processor.decode(generated, skip_special_tokens=True)

        elif self.model_type == 'llama':
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt}
                ]
            }]
            input_text = self.processor.apply_chat_template(messages,
                                                             add_generation_prompt=True)
            inputs = self.processor(image, input_text, return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=200)
            # Decode only new tokens (skip input)
            generated = output_ids[0][inputs['input_ids'].shape[1]:]
            return self.processor.decode(generated, skip_special_tokens=True)

    def _vlm_query_text(self, prompt):
        """Send a text-only query to the VLM (no image, for intent/motion steps)."""
        if self.model_type == 'gpt':
            response = self.client.chat.completions.create(
                model='gpt-4o',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=150
            )
            return response.choices[0].message.content

        # For local models, reuse the last image for context
        temp_path = os.path.join(OPENEMMA_ROOT, '_temp_frame.jpg')
        if os.path.exists(temp_path):
            return self._vlm_query(prompt, temp_path)
        return ""

    def sensors(self):
        """Define sensors needed by OpenEMMA (front camera + speedometer)."""
        return [
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 800, 'height': 600,
                'fov': 100,
                'id': 'rgb_front',
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps',
            },
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu',
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed',
            },
        ]

    def _get_route_steer(self, vehicle):
        """Compute steering angle from route using pure-pursuit.

        Tracks progress along the route (forward-only search) to avoid
        jumping to wrong waypoints after missing a turn.
        Returns (steer, is_valid).
        """
        if not self.route:
            return 0.0, False

        veh_loc = vehicle.get_location()
        veh_transform = vehicle.get_transform()
        veh_fwd = veh_transform.get_forward_vector()

        # Search window: mostly forward, tiny backward tolerance
        search_start = max(0, self._route_idx - 3)
        search_end = min(len(self.route), self._route_idx + 50)

        min_dist_sq = float('inf')
        closest_idx = self._route_idx
        for i in range(search_start, search_end):
            loc = self.route[i][0].location
            d = (loc.x - veh_loc.x)**2 + (loc.y - veh_loc.y)**2
            if d < min_dist_sq:
                min_dist_sq = d
                closest_idx = i

        # Update route index (forward-only with tiny backward tolerance)
        if closest_idx >= self._route_idx:
            self._route_idx = closest_idx
        elif closest_idx >= self._route_idx - 3:
            self._route_idx = closest_idx

        # Dual look-ahead: near for centering, far for turn anticipation
        def _steer_to(look_ahead):
            idx = min(self._route_idx + look_ahead, len(self.route) - 1)
            loc = self.route[idx][0].location
            dx = loc.x - veh_loc.x
            dy = loc.y - veh_loc.y
            d = math.sqrt(dx*dx + dy*dy)
            if d < 0.5:
                return 0.0
            dot = veh_fwd.x * dx + veh_fwd.y * dy
            if dot < 0:
                # Target behind - extend
                idx2 = min(self._route_idx + look_ahead * 3, len(self.route) - 1)
                loc2 = self.route[idx2][0].location
                dx, dy = loc2.x - veh_loc.x, loc2.y - veh_loc.y
                dot = veh_fwd.x * dx + veh_fwd.y * dy
            cross = veh_fwd.x * dy - veh_fwd.y * dx
            return math.atan2(cross, dot) / (math.pi / 2.0)

        steer_near = _steer_to(self.LOOK_AHEAD_NEAR)
        steer_far = _steer_to(self.LOOK_AHEAD_FAR)

        # Blend: 60% near (centering) + 40% far (anticipation)
        steer = 0.6 * steer_near + 0.4 * steer_far
        steer = max(-0.7, min(0.7, steer * self.STEER_GAIN))

        return steer, True

    def _road_recovery_steer(self, vehicle):
        """Steer toward the nearest driving-lane center for recovery only."""
        runner = getattr(self, '_runner', None)
        world = getattr(runner, 'world', None)
        if vehicle is None or world is None:
            return 0.0

        try:
            carla_map = world.get_map()
            veh_loc = vehicle.get_location()
            wp = carla_map.get_waypoint(
                veh_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            return 0.0

        if wp is None:
            return 0.0

        veh_transform = vehicle.get_transform()
        veh_fwd = veh_transform.get_forward_vector()
        lane_loc = wp.transform.location
        lane_fwd = wp.transform.get_forward_vector()

        def _steer_to_vector(dx, dy):
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-3:
                return 0.0
            dot = veh_fwd.x * dx + veh_fwd.y * dy
            cross = veh_fwd.x * dy - veh_fwd.y * dx
            return math.atan2(cross, dot) / (math.pi / 2.0)

        center_dx = lane_loc.x - veh_loc.x
        center_dy = lane_loc.y - veh_loc.y
        center_dist = math.sqrt(center_dx * center_dx + center_dy * center_dy)
        center_steer = _steer_to_vector(center_dx, center_dy)
        align_steer = _steer_to_vector(lane_fwd.x, lane_fwd.y)

        if center_dist > 2.0:
            steer = 0.8 * center_steer + 0.2 * align_steer
        elif center_dist > 0.5:
            steer = 0.6 * center_steer + 0.4 * align_steer
        else:
            steer = 0.2 * center_steer + 0.8 * align_steer

        return float(np.clip(steer * self.STEER_GAIN, -1.0, 1.0))

    def _update_offroad_recovery_trigger(self, vehicle):
        """Trigger recovery after the vehicle center leaves a driving lane."""
        if (
            vehicle is None
            or self._recovery_counter > 0
            or self._realign_counter > 0
        ):
            return

        runner = getattr(self, '_runner', None)
        world = getattr(runner, 'world', None)
        if world is None:
            return

        try:
            carla_map = world.get_map()
            veh_loc = vehicle.get_location()
            wp = carla_map.get_waypoint(
                veh_loc,
                project_to_road=False,
            )
        except Exception:
            return

        is_offroad = wp is None or wp.lane_type != carla.LaneType.Driving
        if is_offroad:
            self._offroad_counter += 1
        else:
            self._offroad_counter = max(0, self._offroad_counter - 1)

        if self._offroad_counter >= self.OFFROAD_RECOVERY_FRAMES:
            self.metrics['recovery_events'] += 1
            self._recovery_counter = 25
            self._realign_counter = 0
            self._offroad_counter = 0

    def _compute_route_curvature(self):
        """Estimate upcoming route curvature for speed adaptation.

        Checks multiple lookahead distances and returns the max curvature,
        so tight turns ahead are detected early.
        """
        if not self.route:
            return 0.0

        max_curv = 0.0
        # Check curvature at multiple distances ahead
        for offset in [(0, 8, 16), (5, 15, 25), (10, 20, 30)]:
            idx0 = min(self._route_idx + offset[0], len(self.route) - 1)
            idx1 = min(self._route_idx + offset[1], len(self.route) - 1)
            idx2 = min(self._route_idx + offset[2], len(self.route) - 1)
            if idx0 == idx2:
                continue
            p0 = self.route[idx0][0].location
            p1 = self.route[idx1][0].location
            p2 = self.route[idx2][0].location
            v1x, v1y = p1.x - p0.x, p1.y - p0.y
            v2x, v2y = p2.x - p1.x, p2.y - p1.y
            cross = abs(v1x * v2y - v1y * v2x)
            d1 = math.sqrt(v1x*v1x + v1y*v1y)
            d2 = math.sqrt(v2x*v2x + v2y*v2y)
            if d1 * d2 > 0.01:
                curv = cross / (d1 * d2)
                max_curv = max(max_curv, curv)
        return max_curv

    def _update_ego_history(self, vehicle):
        """Sample ego pose every 0.5s and derive OpenEMMA motion history."""
        if vehicle is None or self.step % self.HISTORY_SAMPLE_FRAMES != 0:
            return

        loc = vehicle.get_location()
        pos = np.array([loc.x, loc.y], dtype=float)
        self._ego_history_positions.append(pos)
        self._ego_history_positions = self._ego_history_positions[-self.OBS_LEN:]

        positions = np.asarray(self._ego_history_positions, dtype=float)
        velocities = np.empty((0, 2), dtype=float)
        vel_norm = np.empty((0,), dtype=float)
        curvatures = np.zeros(len(positions), dtype=float)
        heading = self.obs_initial_heading

        if len(positions) >= 2:
            velocities = np.zeros_like(positions)
            velocities[1:] = positions[1:] - positions[:-1]
            velocities[0] = velocities[1]
            vel_norm = np.linalg.norm(velocities, axis=1)
            curvatures = EstimateCurvatureFromTrajectory(positions)
            last_velocity = velocities[-1]
            if np.linalg.norm(last_velocity) > 1e-6:
                heading = math.atan2(last_velocity[1], last_velocity[0])

        with self._cot_lock:
            self.obs_positions = positions
            self.obs_velocities = velocities
            self.obs_vel_norm = vel_norm
            self.obs_curvatures = curvatures
            self.obs_initial_heading = heading
            self.obs_sample_count = len(positions)

    def _motion_history_string(self, current_speed):
        """Return OpenEMMA [speed, curvature*100] history string."""
        with self._cot_lock:
            sample_count = self.obs_sample_count
            vel_norm = None if self.obs_vel_norm is None else self.obs_vel_norm.copy()
            curvatures = None if self.obs_curvatures is None else self.obs_curvatures.copy()

        if sample_count >= self.OBS_LEN and vel_norm is not None and curvatures is not None:
            pairs = zip(vel_norm, curvatures)
            return ", ".join(f"[{v:.1f},{c * 100:.1f}]" for v, c in pairs)

        # Cold-start fallback: speed here must be a 0.5s displacement, not m/s.
        fallback_speed = abs(float(current_speed)) * self.FRAME_DT
        return ", ".join(
            f"[{fallback_speed:.1f},{0.0:.1f}]"
            for _ in range(self.OBS_LEN)
        )

    def _parse_motion_pairs(self, response):
        """Parse up to FUT_LEN [speed, curvature] pairs from raw VLM text."""
        pattern = r'\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]'
        pairs = re.findall(pattern, response or "")
        if not pairs:
            return None
        parsed = np.asarray(
            [[float(speed), float(curvature)] for speed, curvature in pairs[:self.FUT_LEN]],
            dtype=float,
        )
        if parsed.size == 0 or not np.all(np.isfinite(parsed)):
            return None
        return parsed

    def _pursue_vlm_trajectory(self, traj, pred_speeds, pred_curvatures, current_speed):
        """Return steer/speed from ego-local VLM waypoints (+x forward, +y left)."""
        if traj is None or pred_speeds is None or len(traj) == 0:
            return None

        traj = np.asarray(traj, dtype=float)
        pred_speeds = np.asarray(pred_speeds, dtype=float)
        if traj.ndim != 2 or traj.shape[1] < 2 or len(pred_speeds) == 0:
            return None
        if not np.all(np.isfinite(traj)) or not np.all(np.isfinite(pred_speeds)):
            return None

        look_ahead = max(self.LOOK_AHEAD_M, min(12.0, abs(current_speed) * 1.5))
        candidates = np.where((traj[:, 0] >= look_ahead) & (traj[:, 0] > 0.0))[0]
        if len(candidates) > 0:
            idx = int(candidates[0])
        else:
            idx = int(np.argmax(np.linalg.norm(traj[:, :2], axis=1)))

        forward = float(traj[idx, 0])
        lateral = float(traj[idx, 1])
        if abs(forward) < 0.5 and abs(lateral) < 0.5:
            steer = 0.0
        else:
            angle = math.atan2(lateral, max(forward, 0.1))
            # OpenEMMA local +y is left; this UI/CARLA controller uses negative steer for left.
            steer = -angle / (math.pi / 2.0) * self.STEER_GAIN
            steer = float(np.clip(steer, -0.7, 0.7))

        speed_idx = min(idx, len(pred_speeds) - 1)
        target_speed = float(np.clip(pred_speeds[speed_idx], 0.0, self.MAX_SPEED))

        curvature = 0.0
        if pred_curvatures is not None and len(pred_curvatures) > 0:
            curv_idx = min(idx, len(pred_curvatures) - 1)
            curvature = float(pred_curvatures[curv_idx])

        return steer, target_speed, curvature, idx

    def _route_fallback_control(self, route_steer, route_valid, route_curvature):
        """Cold-start fallback before a valid VLM trajectory exists."""
        steer_mag = abs(route_steer)
        if route_curvature > 0.3 or steer_mag > 0.5:
            target_speed = self.TURN_SPEED
        elif route_curvature > 0.12 or steer_mag > 0.25:
            curve_factor = max(route_curvature / 0.3, steer_mag / 0.5)
            curve_factor = min(curve_factor, 1.0)
            target_speed = self.CRUISE_SPEED - curve_factor * (self.CRUISE_SPEED - self.CURVE_SPEED)
        else:
            target_speed = self.CRUISE_SPEED

        if route_valid:
            steer = route_steer
        else:
            steer = self.prev_steer * 0.9
        return steer, target_speed

    def _apply_speed_control(self, control, target_speed, current_speed):
        """Proportional longitudinal control toward target_speed in m/s."""
        if target_speed <= 0.05:
            control.throttle = 0.0
            control.brake = 0.6 if abs(current_speed) > 0.2 else 0.3
            return

        speed_error = target_speed - abs(current_speed)
        if speed_error > 0.3:
            control.throttle = float(np.clip(speed_error * 0.3, 0.1, 0.75))
            control.brake = 0.0
        elif speed_error < -0.1:
            control.throttle = 0.0
            control.brake = float(np.clip(-speed_error * 0.5, 0.1, 1.0))
        else:
            control.throttle = 0.15
            control.brake = 0.0

    def _set_ui_waypoints_from_vlm(self, traj):
        """Convert VLM [forward, left] waypoints to panel [right, forward]."""
        if traj is None or len(traj) == 0:
            self.ui_waypoints = None
            return

        traj = np.asarray(traj, dtype=float)
        if len(traj) > 5:
            start = 1 if len(traj) > 1 else 0
            sample_idx = np.linspace(start, len(traj) - 1, min(5, len(traj) - start), dtype=int)
            sampled = traj[sample_idx]
        else:
            sampled = traj

        self.ui_waypoints = np.column_stack((-sampled[:, 1], sampled[:, 0]))

    def run_step(self, input_data, timestamp):
        """Execute one step using VLM trajectory primary control."""
        self.step += 1
        control = carla.VehicleControl()

        speed = input_data.get('speed', (0, {'speed': 0.0}))[1]
        if isinstance(speed, dict):
            speed = speed.get('speed', 0.0)
        current_speed = float(speed)

        vehicle = getattr(self, '_vehicle', None)
        self._update_ego_history(vehicle)

        # First few steps: brake to stabilize before either controller acts.
        if self.step < 10:
            control.brake = 1.0
            return control

        if self._recovery_counter == 0 and self._realign_counter == 0:
            self._update_offroad_recovery_trigger(vehicle)

        if self._recovery_counter > 0:
            self._recovery_counter -= 1
            if vehicle is not None:
                self._recovery_steer = -self._road_recovery_steer(vehicle)
            control.throttle = 0.5
            control.brake = 0.0
            control.reverse = True
            control.steer = self._recovery_steer
            if self._recovery_counter == 0:
                self._stuck_counter = 0
                self.prev_steer = 0.0
                self._realign_counter = 15
            self.curr_instruction = "Road recovery reverse"
            self.ui_desired_speed = self.TURN_SPEED
            self.ui_curvature = 0.0
            self._record_frame_metrics(control.steer, current_speed)
            return control

        if self._realign_counter > 0:
            self._realign_counter -= 1
            realign_steer = (
                self._road_recovery_steer(vehicle)
                if vehicle is not None else 0.0
            )
            control.reverse = False
            control.steer = realign_steer
            self.prev_steer = realign_steer
            self._apply_speed_control(control, self.TURN_SPEED, current_speed)
            self.curr_instruction = "Road recovery realign"
            self.ui_desired_speed = self.TURN_SPEED
            self.ui_curvature = 0.0
            if self._realign_counter == 0:
                self._stuck_counter = 0
            self._record_frame_metrics(control.steer, current_speed)
            return control

        is_red = getattr(self, 'ui_red_light', False)

        rgb_data = input_data.get('rgb_front', (0, None))[1]
        if rgb_data is not None and self.step % 20 == 0 and not self._cot_running:
            try:
                import cv2
                temp_path = os.path.join(OPENEMMA_ROOT, '_temp_frame.jpg')
                img = rgb_data[:, :, :3]
                cv2.imwrite(temp_path, img)

                self._cot_running = True
                t = threading.Thread(
                    target=self._run_cot_async,
                    args=(temp_path, current_speed),
                    daemon=True
                )
                t.start()
            except Exception as e:
                self._cot_running = False
                print(f'[OpenEMMA] CoT launch error: {e}')

        # Route is maintained for cold-start fallback, route regeneration, and
        # SafetyLimiter junction assist. It is not the primary controller here.
        route_steer = 0.0
        route_valid = False
        route_curvature = 0.0
        if vehicle is not None:
            route_steer, route_valid = self._get_route_steer(vehicle)
            route_curvature = self._compute_route_curvature()

        route_remaining = len(self.route) - self._route_idx if self.route else 999
        if route_remaining < 20:
            self._regenerate_route()
            if vehicle is not None:
                route_steer, route_valid = self._get_route_steer(vehicle)
                route_curvature = self._compute_route_curvature()

        with self._cot_lock:
            vlm_traj = None if self.vlm_traj is None else self.vlm_traj.copy()
            vlm_speeds = None if self.vlm_pred_speeds is None else self.vlm_pred_speeds.copy()
            vlm_curvatures = (
                None if self.vlm_pred_curvatures is None
                else self.vlm_pred_curvatures.copy()
            )

        vlm_result = self._pursue_vlm_trajectory(
            vlm_traj, vlm_speeds, vlm_curvatures, current_speed
        )
        if vlm_result is not None:
            steer, target_speed, ui_curvature, _ = vlm_result
            control_source = "VLM trajectory"
            self.metrics['vlm_frames'] += 1
            self._set_ui_waypoints_from_vlm(vlm_traj)
        else:
            steer, target_speed = self._route_fallback_control(
                route_steer, route_valid, route_curvature
            )
            ui_curvature = route_curvature
            control_source = "Route fallback"
            self.metrics['fallback_frames'] += 1
            if vehicle is not None and self.route:
                self._generate_route_waypoints(vehicle)

        intent_lower = self.last_intent.lower()
        scene_lower = self.last_scene.lower()
        explicit_stop = (
            'stop' in intent_lower
            or 'red' in scene_lower
            or 'red' in intent_lower
        )
        legitimate_stop = is_red or explicit_stop
        target_speed_before_floor = target_speed
        if not legitimate_stop:
            target_speed = max(target_speed, self.MIN_DRIVE_SPEED)
            if target_speed_before_floor < self.MIN_DRIVE_SPEED:
                self.metrics['speed_floor_frames'] += 1

        steer_diff = abs(steer - self.prev_steer)
        if steer_diff > 0.15:
            alpha = 0.15
        elif steer_diff > 0.05:
            alpha = 0.35
        else:
            alpha = 0.5
        steer = alpha * self.prev_steer + (1.0 - alpha) * steer
        self.prev_steer = steer
        control.steer = steer

        self._apply_speed_control(control, target_speed, current_speed)

        if (
            control.throttle > 0.1
            and abs(current_speed) < 0.3
            and self.step > 20
            and not legitimate_stop
        ):
            self._stuck_counter += 1
        else:
            self._stuck_counter = max(0, self._stuck_counter - 2)

        if self._stuck_counter > 40:
            self.metrics['stuck_events'] += 1
            self.metrics['recovery_events'] += 1
            old_idx = self._route_idx
            if (
                self._last_recovery_route_idx is not None
                and old_idx <= self._last_recovery_route_idx
            ):
                self._stuck_recovery_events += 1
            else:
                self._stuck_recovery_events = 1
            self._last_recovery_route_idx = old_idx

            if self._stuck_recovery_events >= 3:
                print(f'[OpenEMMA] Stuck recovery loop at step {self.step}, '
                      f'route_idx {old_idx}, regenerating route...')
                self._regenerate_route(count_metric=True)
                self._stuck_recovery_events = 0
                self._last_recovery_route_idx = None
            else:
                self._route_idx = max(0, self._route_idx - 40)
                print(f'[OpenEMMA] Stuck at step {self.step}, '
                      f'route_idx {old_idx}->{self._route_idx}, reversing...')

            self._recovery_steer = (
                -self._road_recovery_steer(vehicle)
                if vehicle is not None else 0.0
            )
            self._recovery_counter = 25
            self._realign_counter = 0
            self._stuck_counter = 0

        if self.step % 40 == 0 or control_source != self._last_vlm_dbg_source:
            if self.debug:
                print(
                    f"[VLM-DBG] src={control_source} target_speed={target_speed:.1f} "
                    f"steer={steer:+.2f} cur_speed={current_speed:.1f}"
                )
            self._last_vlm_dbg_source = control_source

        self.curr_instruction = (
            f"Intent: {self.last_intent[:80]}" if self.last_intent else control_source
        )
        if self.last_scene:
            self.curr_notice = self.last_scene[:100]
        self.ui_desired_speed = target_speed
        self.ui_curvature = ui_curvature

        self._record_frame_metrics(control.steer, current_speed)
        return control

    def _generate_route_waypoints(self, vehicle):
        """Generate UI waypoints from route ahead of vehicle."""
        if not self.route:
            return

        veh_loc = vehicle.get_location()
        veh_transform = vehicle.get_transform()
        veh_fwd = veh_transform.get_forward_vector()
        veh_right = veh_transform.get_right_vector()

        # Find closest
        min_dist = float('inf')
        closest_idx = 0
        for i, (transform, _) in enumerate(self.route):
            loc = transform.location
            d = (loc.x - veh_loc.x)**2 + (loc.y - veh_loc.y)**2
            if d < min_dist:
                min_dist = d
                closest_idx = i

        # Sample 5 waypoints ahead
        waypoints = []
        for offset in [2, 5, 8, 12, 16]:
            idx = min(closest_idx + offset, len(self.route) - 1)
            wp_loc = self.route[idx][0].location
            # Convert to ego-relative coords
            dx = wp_loc.x - veh_loc.x
            dy = wp_loc.y - veh_loc.y
            # Forward = y, Right = x in ego frame
            local_x = dx * veh_right.x + dy * veh_right.y
            local_y = dx * veh_fwd.x + dy * veh_fwd.y
            waypoints.append([local_x, local_y])

        self.ui_waypoints = np.array(waypoints)

    def _run_cot_async(self, image_path, current_speed):
        """Wrapper that runs CoT pipeline in background thread."""
        try:
            self._run_cot_pipeline(image_path, current_speed)
        except Exception as e:
            print(f'[OpenEMMA] CoT error: {e}')
        finally:
            self._cot_running = False

    def _run_cot_pipeline(self, image_path, current_speed):
        """Run the full Chain-of-Thought pipeline."""
        self.metrics['cot_cycles'] += 1

        # Step 1: Scene Description (with image)
        scene = self._vlm_query(SCENE_PROMPT, image_path)
        self.last_scene = scene.strip()[:200]


        # Step 2: Critical Objects (with image)
        objects = self._vlm_query(OBJECTS_PROMPT, image_path)
        self.last_objects = objects.strip()[:200]


        # Step 3: Driving Intent (uses scene + objects context)
        prev_str = f"Previous intent: {self.prev_intent}\n" if self.prev_intent else ""
        intent_prompt = INTENT_PROMPT_TEMPLATE.format(
            speed=abs(current_speed),
            scene=self.last_scene,
            objects=self.last_objects,
            prev_intent_str=prev_str
        )
        intent = self._vlm_query_text(intent_prompt)
        self.last_intent = intent.strip()[:150]
        self.prev_intent = self.last_intent

        intent_lower = self.last_intent.lower()
        scene_lower = self.last_scene.lower()
        explicit_stop = (
            'stop' in intent_lower
            or 'red' in scene_lower
            or 'red' in intent_lower
        )
        if explicit_stop:
            self.metrics['cot_stop_or_red'] += 1

        # Step 4: Motion Prediction
        history = self._motion_history_string(current_speed)
        motion_prompt = MOTION_SYSTEM + "\n" + MOTION_PROMPT_TEMPLATE.format(
            scene=self.last_scene,
            objects=self.last_objects,
            intent=self.last_intent,
            history=history,
        )
        parsed = None
        motion_response = ""
        for _ in range(3):
            motion_response = self._vlm_query(motion_prompt, image_path)
            self.last_motion_raw = (motion_response or "").strip()[:100]
            parsed = self._parse_motion(motion_response)
            if parsed is not None:
                break

        if parsed is None:
            with self._cot_lock:
                self.vlm_traj = None
                self.vlm_pred_speeds = None
                self.vlm_pred_curvatures = None
            return

        raw_pred_speeds = parsed[:, 0]
        raw_pred_speeds_mps = raw_pred_speeds / self.FRAME_DT
        speed_clamped = bool(np.any(raw_pred_speeds_mps > self.SANE_MAX_SPEED_MPS))
        pred_speeds_mps = np.clip(
            raw_pred_speeds_mps,
            0.0,
            self.SANE_MAX_SPEED_MPS,
        )
        if len(pred_speeds_mps) >= 3:
            padded = np.pad(pred_speeds_mps, (1, 1), mode='edge')
            pred_speeds_mps = np.convolve(
                padded,
                np.ones(3, dtype=float) / 3.0,
                mode='valid',
            )
            pred_speeds_mps = np.clip(
                pred_speeds_mps,
                0.0,
                self.SANE_MAX_SPEED_MPS,
            )

        pred_speeds = pred_speeds_mps * self.FRAME_DT
        raw_curvatures = parsed[:, 1] / 100.0
        curv_clamped = bool(np.any(np.abs(raw_curvatures) > self.SANE_MAX_CURV))
        pred_curvatures = np.clip(
            raw_curvatures,
            -self.SANE_MAX_CURV,
            self.SANE_MAX_CURV,
        )
        if speed_clamped:
            self.metrics['speed_clamp_events'] += 1
        if curv_clamped:
            self.metrics['curv_clamp_events'] += 1
        raw_speed_preview = [round(float(v), 2) for v in raw_pred_speeds_mps[:3]]
        sane_speed_preview = [round(float(v), 2) for v in pred_speeds_mps[:3]]

        if len(pred_speeds_mps) == 0 or float(np.max(pred_speeds_mps)) < 0.1:
            self.metrics['degenerate_rejections'] += 1
            if self.debug:
                print(
                    f"[VLM-DBG] intent='{self.last_intent[:50]}' pairs={len(parsed)} "
                    f"pred_speeds_mps={raw_speed_preview} "
                    f"sane_speeds_mps={sane_speed_preview} rejected=degenerate"
                )
            with self._cot_lock:
                self.vlm_traj = None
                self.vlm_pred_speeds = None
                self.vlm_pred_curvatures = None
            return

        traj = IntegrateCurvatureForPoints(
            pred_curvatures,
            pred_speeds,
            (0.0, 0.0),
            0.0,
            len(pred_speeds),
        )
        traj_fwd_max = 0.0
        if traj is not None and len(traj) > 0:
            traj_fwd_max = round(float(np.max(np.asarray(traj)[:, 0])), 2)
        if self.debug:
            print(
                f"[VLM-DBG] intent='{self.last_intent[:50]}' pairs={len(parsed)} "
                f"pred_speeds_mps={raw_speed_preview} "
                f"sane_speeds_mps={sane_speed_preview} traj_fwd_max={traj_fwd_max}"
            )

        with self._cot_lock:
            self.vlm_traj = traj
            self.vlm_pred_speeds = pred_speeds_mps
            self.vlm_pred_curvatures = pred_curvatures

    def _parse_motion(self, response):
        """Parse VLM [speed, curvature] actions without changing controller state."""
        return self._parse_motion_pairs(response)

    def destroy(self):
        """Cleanup."""
        # Remove temp frame
        temp_path = os.path.join(OPENEMMA_ROOT, '_temp_frame.jpg')
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description='OpenEMMA CARLA Autonomous Driving UI',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # CARLA settings
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--town', default='Town01',
                        help='CARLA town to load')
    # Camera settings
    parser.add_argument('--cam-width', type=int, default=1280,
                        help='Chase camera width')
    parser.add_argument('--cam-height', type=int, default=720,
                        help='Chase camera height')

    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--gpt', action='store_true',
                             help='Use GPT-4o via API (requires OPENAI_API_KEY)')
    model_group.add_argument('--qwen', action='store_true',
                             help='Use Qwen2-VL-7B from OpenEMMA/local/')
    model_group.add_argument('--llava', action='store_true',
                             help='Use LLaVA-v1.5-7B from OpenEMMA/local/')
    model_group.add_argument('--llama', action='store_true',
                             help='Use Llama-3.2-11B-Vision from OpenEMMA/local/')
    model_group.add_argument('--model-path', default=None,
                             help='Custom path to VLM model directory')

    # Performance options
    parser.add_argument('--4bit', dest='use_4bit', action='store_true',
                        help='Use 4-bit quantization (for 8GB VRAM GPUs)')
    parser.add_argument('--debug', action='store_true',
                        help='Print [VLM-DBG] diagnostic logs each frame/CoT cycle')

    args = parser.parse_args()

    # Resolve model path
    if args.model_path:
        model_path = args.model_path
    elif args.qwen:
        model_path = LOCAL_MODELS['qwen']
    elif args.llava:
        model_path = LOCAL_MODELS['llava']
    elif args.llama:
        model_path = LOCAL_MODELS['llama']
    elif args.gpt:
        model_path = 'gpt'
    else:
        model_path = LOCAL_MODELS['qwen']

    runner = None
    agent = None

    try:
        # 1. Set up CARLA
        runner = AgentRunner(
            host=args.host, port=args.port, town=args.town,
            camera_width=args.cam_width, camera_height=args.cam_height,
            model_name=f'OpenEMMA ({os.path.basename(model_path)})',
            target_fps=20
        )
        runner.setup()

        # 2. Create OpenEMMA agent
        agent = OpenEMMACarlaAgent(
            model_path=model_path,
            use_4bit=args.use_4bit,
            debug=args.debug,
        )

        # 3. Generate route and pass to agent for steering
        _, world_route = runner.generate_route()
        agent.set_route(world_route)

        # Also set route on safety limiter for junction assist
        runner.safety_limiter.set_route(world_route)

        # Give agent access to ego vehicle and runner for route steering/regeneration
        agent._vehicle = runner.ego_vehicle
        agent.set_runner(runner)

        # 4. Set up agent sensors
        runner.setup_agent_sensors(agent)

        # 5. Set up UI
        runner.setup_ui()

        # 6. Run simulation
        runner.run_loop(agent)

    except Exception as e:
        print(f'[ERROR] {e}')
        import traceback
        traceback.print_exc()

    finally:
        if agent:
            agent.destroy()
        if runner:
            runner.cleanup()


if __name__ == '__main__':
    main()

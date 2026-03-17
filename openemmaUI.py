"""
OpenEMMA - CARLA Autonomous Driving UI

Adapts OpenEMMA (originally designed for nuScenes) to work with CARLA 0.9.16.
Uses a VLM (Qwen2-VL, LLaVA, or GPT-4o) for chain-of-thought driving decisions.

Architecture:
    - Route-based pure-pursuit controller for steering (primary)
    - VLM Chain-of-Thought for scene understanding (advisory, displayed in UI)
    - SafetyLimiter handles red lights, lane-keeping (in agent_runner)

Pipeline: Scene Description → Critical Objects → Driving Intent → Motion Prediction
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

# ── CARLA 0.9.16 PythonAPI setup (must be first) ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui_common'))
from carla_setup import setup_carla_paths
setup_carla_paths()

import carla
import numpy as np

# ── Path setup for OpenEMMA ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OPENEMMA_ROOT = os.path.join(PROJECT_ROOT, 'OpenEMMA')
if OPENEMMA_ROOT not in sys.path:
    sys.path.insert(0, OPENEMMA_ROOT)

from ui_common.agent_runner import AgentRunner

# ── Default local model paths (in OpenEMMA/local/) ──
LOCAL_MODEL_DIR = os.path.join(OPENEMMA_ROOT, 'local')
LOCAL_MODELS = {
    'qwen': os.path.join(LOCAL_MODEL_DIR, 'Qwen2-VL-7B-Instruct'),
    'llava': os.path.join(LOCAL_MODEL_DIR, 'llava-v1.5-7b'),
    'llama': os.path.join(LOCAL_MODEL_DIR, 'Llama-3.2-11B-Vision-Instruct'),
}


# ─────────────────────────────────────────────
# Chain-of-Thought Prompt Templates
# ─────────────────────────────────────────────

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
    "You are an autonomous driving motion planner in CARLA simulator. "
    "Given the scene understanding and driving intent, predict the ego vehicle's "
    "future trajectory as speed and curvature pairs for the next 3 seconds. "
    "Output format: speed_1,curvature_1;speed_2,curvature_2;speed_3,curvature_3\n"
    "- speed: m/s (0-15), curvature: turning rate (-0.15 to 0.15, negative=left, positive=right, 0=straight)\n"
    "- If red light or obstacle: predict speed=0\n"
    "- Output ONLY the numbers, no explanation."
)

MOTION_PROMPT_TEMPLATE = (
    "Current speed: {speed:.1f} m/s.\n"
    "Scene: {scene}\n"
    "Objects: {objects}\n"
    "Intent: {intent}\n"
    "Predict speed,curvature for next 3 seconds:"
)


# ─────────────────────────────────────────────
# OpenEMMA CARLA Adapter
# ─────────────────────────────────────────────

class OpenEMMACarlaAgent:
    """
    Adapts OpenEMMA's VLM-based driving pipeline to work as a CARLA agent.

    Control Strategy:
    - PRIMARY: Route-based pure-pursuit steering + cruise speed
    - ADVISORY: VLM CoT provides scene understanding for UI display
    - VLM intent can modulate speed (slow down / speed up)
    - SafetyLimiter handles actual red lights and lane-keeping

    Full Chain-of-Thought pipeline (runs every N frames):
    1. Scene Description - what's in the driving scene
    2. Critical Objects - what to pay attention to
    3. Driving Intent - what should the car do
    4. Motion Prediction - speed/curvature for next 3 seconds
    """

    # Cruise speed parameters
    CRUISE_SPEED = 4.5      # m/s (~16 km/h) - default target
    MAX_SPEED = 5.5         # m/s (~20 km/h) - max allowed
    TURN_SPEED = 2.5        # m/s (~9 km/h) - speed in sharp turns
    CURVE_SPEED = 3.8       # m/s (~14 km/h) - moderate curves
    # Route steering (dual look-ahead)
    LOOK_AHEAD_NEAR = 12    # near target for lane centering
    LOOK_AHEAD_FAR = 22     # far target for turn anticipation
    STEER_GAIN = 1.3        # steering responsiveness

    def __init__(self, model_path, device='cuda', use_4bit=False):
        self.model_path = model_path
        self.use_4bit = use_4bit
        self.device = device

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

        # VLM speed/curvature predictions
        self.vlm_speed = None
        self.vlm_curvature = None

        # Async VLM inference
        self._cot_running = False
        self._cot_lock = threading.Lock()

        # Control smoothing
        self.prev_steer = 0.0

        # Stuck recovery
        self._stuck_counter = 0
        self._recovery_counter = 0
        self._recovery_steer = 0.0

        # UI attributes (read by agent_runner._render_ui)
        self.curr_instruction = ''
        self.curr_notice = ''
        self.ui_waypoints = None
        self.ui_desired_speed = 0.0
        self.ui_curvature = 0.0
        # Must be initialized so agent_runner can update via hasattr() check
        self.ui_red_light = False
        self.ui_violations = {}

        # VLM components
        self.model = None
        self.processor = None
        self.tokenizer = None

        self._load_model()

    def set_route(self, world_route):
        """Set the route for pure-pursuit steering."""
        self.route = world_route
        self._route_idx = 0

    def set_runner(self, runner):
        """Store runner reference for route regeneration."""
        self._runner = runner

    def _regenerate_route(self):
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

    def run_step(self, input_data, timestamp):
        """Execute one step of OpenEMMA driving.

        Uses route-based steering as primary control.
        VLM CoT runs every 20 frames for scene understanding (UI display).
        """
        self.step += 1
        control = carla.VehicleControl()

        # First few steps: brake to stabilize
        if self.step < 10:
            control.brake = 1.0
            return control

        # Get current speed
        speed = input_data.get('speed', (0, {'speed': 0.0}))[1]
        if isinstance(speed, dict):
            speed = speed.get('speed', 0.0)
        current_speed = float(speed)

        # ── Stuck recovery: if speed near 0 for too long, reverse briefly ──
        if self._recovery_counter > 0:
            self._recovery_counter -= 1
            control.throttle = 0.5
            control.brake = 0.0
            control.reverse = True
            # Steer away from wall: use route direction after reset
            control.steer = self._recovery_steer
            if self._recovery_counter == 0:
                self._stuck_counter = 0
                self.prev_steer = 0.0
            return control

        is_red = getattr(self, 'ui_red_light', False)
        if abs(current_speed) < 0.3 and self.step > 20 and not is_red:
            self._stuck_counter += 1
        else:
            self._stuck_counter = max(0, self._stuck_counter - 2)

        if self._stuck_counter > 40:  # ~2 seconds stuck
            # Reset _route_idx backward to re-find the turn we missed
            old_idx = self._route_idx
            self._route_idx = max(0, self._route_idx - 40)
            print(f'[OpenEMMA] Stuck at step {self.step}, '
                  f'route_idx {old_idx}→{self._route_idx}, reversing...')
            # Compute recovery steer: steer toward next route point
            self._recovery_steer = 0.3  # default slight right
            vehicle = getattr(self, '_vehicle', None)
            if vehicle is not None:
                rs, valid = self._get_route_steer(vehicle)
                if valid:
                    self._recovery_steer = -rs  # reverse steer = opposite
            self._recovery_counter = 25  # ~1.2 seconds reverse
            self._stuck_counter = 0

        # Save front camera image and launch async VLM inference
        rgb_data = input_data.get('rgb_front', (0, None))[1]

        # Launch CoT in background thread every 20 frames (non-blocking)
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

        # ── PRIMARY CONTROL: Route-based steering + cruise speed ──
        route_steer = 0.0
        route_valid = False

        # Get vehicle reference for route steering
        vehicle = getattr(self, '_vehicle', None)
        if vehicle is not None:
            route_steer, route_valid = self._get_route_steer(vehicle)
            route_curvature = self._compute_route_curvature()
        else:
            route_curvature = 0.0

        # ── Route end: generate new random route ──
        route_remaining = len(self.route) - self._route_idx if self.route else 999
        if route_remaining < 20:
            self._regenerate_route()
            # Re-compute steer with new route
            if vehicle is not None:
                route_steer, route_valid = self._get_route_steer(vehicle)
                route_curvature = self._compute_route_curvature()

        # Determine target speed based on upcoming curvature + current steer
        steer_mag = abs(route_steer)
        if route_curvature > 0.3 or steer_mag > 0.5:
            # Sharp turn (intersection/U-turn)
            target_speed = self.TURN_SPEED
        elif route_curvature > 0.12 or steer_mag > 0.25:
            # Moderate curve - proportional slowdown
            curve_factor = max(route_curvature / 0.3, steer_mag / 0.5)
            curve_factor = min(curve_factor, 1.0)
            target_speed = self.CRUISE_SPEED - curve_factor * (self.CRUISE_SPEED - self.CURVE_SPEED)
        else:
            target_speed = self.CRUISE_SPEED

        # VLM advisory: modulate speed based on intent
        intent_lower = self.last_intent.lower()
        if 'slow' in intent_lower:
            target_speed = min(target_speed, 3.0)
        elif 'speed up' in intent_lower:
            target_speed = min(target_speed + 2.0, self.MAX_SPEED)

        # Steering: use route steering as primary
        if route_valid:
            steer = route_steer
        else:
            # Route invalid: keep last steer direction (NOT VLM fallback which causes flip-flops)
            steer = self.prev_steer * 0.9  # gently decay toward straight

        # Adaptive smoothing: stable on straights, responsive on curves
        steer_diff = abs(steer - self.prev_steer)
        if steer_diff > 0.15:
            # Curve entry/exit: fast response
            alpha = 0.15
        elif steer_diff > 0.05:
            # Moderate correction
            alpha = 0.35
        else:
            # Straight: smooth/stable
            alpha = 0.5
        steer = alpha * self.prev_steer + (1.0 - alpha) * steer
        self.prev_steer = steer
        control.steer = steer

        # Speed control (proportional)
        speed_error = target_speed - abs(current_speed)
        if speed_error > 0.3:
            # Need to accelerate
            control.throttle = float(np.clip(speed_error * 0.3, 0.1, 0.75))
            control.brake = 0.0
        elif speed_error < -0.1:
            # Need to slow down (brake proportional to overspeed)
            control.throttle = 0.0
            control.brake = float(np.clip(-speed_error * 0.5, 0.1, 1.0))
        else:
            # Near target: gentle throttle to maintain
            control.throttle = 0.15
            control.brake = 0.0

        # Update UI attributes
        self.curr_instruction = f"Intent: {self.last_intent[:80]}" if self.last_intent else "Route following"
        if self.last_scene:
            self.curr_notice = self.last_scene[:100]
        self.ui_desired_speed = target_speed * 3.6
        self.ui_curvature = route_curvature

        # Generate UI waypoints from route
        if vehicle is not None and self.route:
            self._generate_route_waypoints(vehicle)

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


        # Step 4: Motion Prediction
        motion_prompt = MOTION_SYSTEM + "\n" + MOTION_PROMPT_TEMPLATE.format(
            speed=abs(current_speed),
            scene=self.last_scene,
            objects=self.last_objects,
            intent=self.last_intent,
        )
        motion_response = self._vlm_query_text(motion_prompt)
        self.last_motion_raw = motion_response.strip()[:100]


        self._parse_motion(motion_response)

    def _parse_motion(self, response):
        """Parse speed and curvature predictions from VLM response."""
        try:
            numbers = re.findall(r'[-+]?\d*\.?\d+', response)
            if len(numbers) >= 2:
                self.vlm_speed = float(np.clip(float(numbers[0]), 0, 15))
                self.vlm_curvature = float(np.clip(float(numbers[1]), -0.15, 0.15))
            else:
                intent_lower = self.last_intent.lower()
                if 'stop' in intent_lower or 'red' in intent_lower:
                    self.vlm_speed = 0.0
                    self.vlm_curvature = 0.0
        except Exception:
            pass

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
        agent = OpenEMMACarlaAgent(model_path=model_path, use_4bit=args.use_4bit)

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

"""Main Pygame window manager that composes camera view and info panel."""

import time
import pygame

from .carla_utils import CarlaConnection
from .camera import ChaseCameraManager
from .panel import InfoPanel


class UIRenderer:
    """Orchestrates the full UI: camera view + info panel."""

    def __init__(self, carla_conn, window_title='CARLA AV Monitor',
                 camera_width=1280, camera_height=720,
                 model_name='', role_name='hero'):
        """Initialize the full UI.

        Args:
            carla_conn: An established CarlaConnection instance
            window_title: Pygame window title
            camera_width: Width of the camera view area
            camera_height: Height of the camera view area
            model_name: Model name displayed on the info panel
            role_name: Ego vehicle role_name attribute to search for
        """
        self._conn = carla_conn
        self._camera_width = camera_width
        self._camera_height = camera_height

        # Initialize pygame
        pygame.init()
        total_width = InfoPanel.PANEL_WIDTH + camera_width
        self._display = pygame.display.set_mode(
            (total_width, camera_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption(window_title)

        # Info panel
        self._panel = InfoPanel(camera_height, model_name=model_name)

        # Find ego vehicle (retry for up to 30 seconds)
        self._ego = self._wait_for_ego(role_name, timeout=30)

        # Chase camera
        self._camera = ChaseCameraManager(
            self._ego, camera_width, camera_height
        )

        self._clock = pygame.time.Clock()

    def _wait_for_ego(self, role_name, timeout=30):
        """Wait for the ego vehicle to appear in the simulation."""
        start = time.time()
        while time.time() - start < timeout:
            ego = self._conn.get_ego_vehicle(role_name)
            if ego is not None:
                print(f'[UI] Found ego vehicle: {ego.type_id} (id={ego.id})')
                return ego
            print(f'[UI] Waiting for ego vehicle (role_name={role_name})...')
            time.sleep(1.0)
        raise RuntimeError(
            f'Ego vehicle with role_name="{role_name}" not found after {timeout}s. '
            'Make sure the driving agent is running.'
        )

    def run(self, target_fps=30):
        """Main event loop. Blocks until user closes window or presses ESC."""
        running = True
        while running:
            self._clock.tick(target_fps)

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                    running = False

            # Check ego vehicle still alive
            if not self._ego.is_alive:
                print('[UI] Ego vehicle destroyed. Exiting.')
                break

            # Query world state
            vehicle_count = self._conn.get_vehicle_count()
            pedestrian_count = self._conn.get_pedestrian_count()
            speed_kmh = self._conn.get_vehicle_speed_kmh(self._ego)
            brake_status = self._conn.get_brake_status(self._ego)

            # Clear
            self._display.fill((0, 0, 0))

            # Camera view (right side)
            surface = self._camera.get_surface()
            if surface:
                self._display.blit(surface, (InfoPanel.PANEL_WIDTH, 0))

            # Info panel (left side)
            self._panel.render(
                self._display,
                vehicle_count, pedestrian_count,
                speed_kmh, brake_status
            )

            pygame.display.flip()

    def cleanup(self):
        """Destroy camera sensor and quit pygame."""
        if self._camera:
            self._camera.destroy()
        pygame.quit()

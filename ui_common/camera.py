"""Chase camera manager for CARLA vehicles."""

import weakref
import numpy as np
import pygame
import carla


class ChaseCameraManager:
    """Third-person chase camera attached to the ego vehicle.

    Camera is positioned behind and above the vehicle, looking slightly
    downward so the rear license plate and road ahead are visible.
    """

    def __init__(self, vehicle, image_width=1280, image_height=720):
        self._surface = None
        self._vehicle = vehicle
        world = vehicle.get_world()

        # Camera blueprint
        bp_lib = world.get_blueprint_library()
        bp = bp_lib.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(image_width))
        bp.set_attribute('image_size_y', str(image_height))
        bp.set_attribute('fov', '100')

        # Chase camera transform: behind and above, angled to see road ahead
        transform = carla.Transform(
            carla.Location(x=-8.0, y=0.0, z=3.5),
            carla.Rotation(pitch=-5.0)
        )

        # SpringArmGhost gives smooth camera following
        self._sensor = world.spawn_actor(
            bp, transform,
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.SpringArmGhost
        )

        # Use weakref to avoid preventing garbage collection
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda image: ChaseCameraManager._on_image(weak_self, image))

    @staticmethod
    def _on_image(weak_self, carla_image):
        """Convert CARLA image to pygame Surface."""
        self = weak_self()
        if self is None:
            return

        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        array = array[:, :, :3].copy()  # Drop alpha + copy (buffer may be reused)
        array = array[:, :, ::-1]       # BGR -> RGB (returns view, but make_surface copies)
        self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def get_surface(self):
        """Return the latest camera frame as a pygame Surface, or None."""
        return self._surface

    def destroy(self):
        """Destroy the CARLA camera sensor actor."""
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None

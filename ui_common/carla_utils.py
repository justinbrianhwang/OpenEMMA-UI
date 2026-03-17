"""CARLA server connection and world query utilities."""

import carla
import math


class CarlaConnection:
    """Manages connection to CARLA server and provides world query utilities."""

    def __init__(self, host='localhost', port=2000, timeout=10.0):
        self._client = carla.Client(host, port)
        self._client.set_timeout(timeout)
        self._world = self._client.get_world()

    def get_world(self):
        return self._world

    def get_client(self):
        return self._client

    def get_vehicle_count(self):
        """Count all vehicles in the simulation."""
        actors = self._world.get_actors().filter('vehicle.*')
        return len(actors)

    def get_pedestrian_count(self):
        """Count all pedestrians in the simulation."""
        actors = self._world.get_actors().filter('walker.pedestrian.*')
        return len(actors)

    def get_ego_vehicle(self, role_name='hero'):
        """Find the ego vehicle by role_name attribute."""
        vehicles = self._world.get_actors().filter('vehicle.*')
        for v in vehicles:
            if v.attributes.get('role_name') == role_name:
                return v
        return None

    def get_vehicle_speed_kmh(self, vehicle):
        """Get vehicle speed in km/h."""
        vel = vehicle.get_velocity()
        speed_ms = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        return 3.6 * speed_ms

    def get_vehicle_control(self, vehicle):
        """Get current vehicle control state."""
        return vehicle.get_control()

    def get_brake_status(self, vehicle):
        """Return 'BRAKING', 'ACCELERATING', or 'IDLE'."""
        control = vehicle.get_control()
        if control.brake > 0.0:
            return 'BRAKING'
        elif control.throttle > 0.0:
            return 'ACCELERATING'
        return 'IDLE'

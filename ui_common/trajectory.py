"""Trajectory helpers ported from the original OpenEMMA utils.py."""

from scipy.integrate import cumulative_trapezoid
import numpy as np


def EstimateCurvatureFromTrajectory(traj):
    traj = traj[:, :2]
    curvature = np.zeros(len(traj))
    for i in range(1, len(traj) - 1):
        x1, y1 = traj[i - 1]; x2, y2 = traj[i]; x3, y3 = traj[i + 1]
        v1 = np.array([x2 - x1, y2 - y1]); v2 = np.array([x3 - x2, y3 - y2])
        L1 = np.linalg.norm(v1); L2 = np.linalg.norm(v2)
        L3 = np.linalg.norm(np.array([x3 - x1, y3 - y1]))
        area_signed = 0.5 * ((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))
        if L1 > 0 and L2 > 0 and L3 > 0:
            curvature[i] = 4 * area_signed / (L1 * L2 * L3)
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature


def IntegrateCurvatureForPoints(curvatures, velocities_norm, initial_position, initial_heading, time_span):
    t = np.linspace(0, time_span, time_span)
    x0, y0 = initial_position[0], initial_position[1]
    theta0 = initial_heading
    theta = cumulative_trapezoid(curvatures * velocities_norm, t, initial=0)
    theta += theta0
    v_x = velocities_norm * np.cos(theta)
    v_y = velocities_norm * np.sin(theta)
    x = cumulative_trapezoid(v_x, t, initial=0)
    y = cumulative_trapezoid(v_y, t, initial=0)
    x += x0; y += y0
    return np.stack((x, y), axis=1)

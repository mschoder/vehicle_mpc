import math
import numpy as np
import shapely.geometry as geom
from shapely import affinity
import itertools
from matplotlib import pyplot as plt
from descartes import PolygonPatch

from vehicle_mpc import trajectory_gen as tgen
from vehicle_mpc import scenarios


class Environment:
    def __init__(self, obstacles, start, goal_region, bounds=None):
        self.environment_loaded = False
        self.obstacles = obstacles  # list of lists of tuples
        self.bounds = bounds
        self.start = start  # (x,y)
        self.goal_region = goal_region  # list of tuples defining corner points
        self.control_points = []
        self.calculate_scene_dimensions()

    def add_obstacles(self, obstacles):
        self.obstacles += obstacles
        self.calculate_scene_dimensions()

    def set_goal_region(self, goal_region):
        self.goal_region = goal_region
        self.calculate_scene_dimensions()

    def add_control_points(self, points):
        self.control_points += points
        self.calculate_scene_dimensions()

    def calculate_scene_dimensions(self):
        """Compute scene bounds from obstacles, start, and goal """
        points = []
        for elem in self.obstacles:
            points = points + elem
        if self.start:
            points += [self.start]
        if self.goal_region:
            points += self.goal_region
        if len(self.control_points) > 0:
            points += self.control_points
        mp = geom.MultiPoint(points)
        self.bounds = mp.bounds


def plot_environment(env, bounds=None, figsize=None, margin=1.0):
    if bounds is None and env.bounds:
        minx, miny, maxx, maxy = env.bounds
        minx -= margin
        miny -= margin
        maxx += margin
        maxy += margin
    elif bounds:
        minx, miny, maxx, maxy = bounds
    else:
        minx, miny, maxx, maxy = (-10, -5, 10, 5)
    max_width, max_height = 12, 5.5
    if figsize is None:
        width, height = max_width, (maxy-miny)*max_width/(maxx-minx)
        if height > 5:
            width, height = (maxx-minx)*max_height/(maxy-miny), max_height
        figsize = (width, height)

    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111)
    # obstacles
    for i, obs in enumerate(env.obstacles):
        poly = geom.Polygon(obs)
        patch = PolygonPatch(poly, fc='orange', ec='black',
                             alpha=0.5, zorder=20)
        ax.add_patch(patch)

    # start / goal
    goal_poly = geom.Polygon(env.goal_region)
    ax.add_patch(PolygonPatch(goal_poly, fc='green',
                 ec='green', alpha=0.5, zorder=1))
    start = geom.Point(env.start).buffer(0.2, resolution=3)
    ax.add_patch(PolygonPatch(start, fc='red',
                 ec='black', alpha=0.7, zorder=1))

    # control points
    for pt in env.control_points:
        ax.add_patch(PolygonPatch(geom.Point(pt).buffer(
            0.1, resolution=3), fc='black', ec='black', alpha=1, zorder=1))

    plt.xlim([minx, maxx])
    plt.ylim([miny, maxy])
    ax.set_aspect('equal', adjustable='box')
    return ax

    # Helpers for obstacle constraint handling

    def centroid(obstacle):
        ''' 
        Averages all vertices in a given obstacle. Average of x's and y's is
        guaranteed to lie inside polygon
        '''
        x_avg = sum([v[0] for v in obstacle])/len(obstacle)
        y_avg = sum([v[1] for v in obstacle])/len(obstacle)
        return (x_avg, y_avg)

    def linear_obstacle_constraints(obs):
        '''
        Given polygonal obstsacle, returns a list of values for a, b, c
        Constraints take form: cy <= ax + b - buffer + Mz
        Assumes obstacles are given as consecutive ordered list of vertices
        '''
        centroid = centroid(obstacle)
        for i, v in enumerate(obs):
            v1 = obs[i]
            # get next vertex; loop back to first for last constraint
            v2 = obs[(i+1) % len(obs)]
            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]

            if dx == 0:     # vertical constaint case; x <= b
                c = 0
                if centroid[0] <= v1[0]:  # flip constraint
                    a, b, c = 1, -v1[0], 0
                else:
                    a, b, c = -1, v1[0], 0

            else:           # non-vertical constraint; cy <= ax + b
                a = dy / dx
                b = v1[1] - a * v1[0]
                if centroid[1] < a * centroid[0] + b:  # flip constraint
                    a, b, c = -1, -1, -1
                else:
                    a, b, c = 1, 1, 1
            return a, b, c


# Test
if __name__ == '__main__':

    scene = scenarios.two_obstacle
    control_pts = [(3, 0), (4.5, 1), (5, 2.8),
                   (3.5, 3.4), (2, 5.1), (3.7, 6.7), (5.5, 6.3)]
    env = Environment(scene['obs_list'],
                      scene['start'], scene['goal'])
    env.add_control_points(control_pts)
    ax = plot_environment(env)

    xc = [p[0] for p in control_pts]
    yc = [p[1] for p in control_pts]
    v = 5
    dt = 0.1
    bc_headings = (np.pi/8, -np.pi/12)
    xs, ys, psi = tgen.sample_trajectory(xc, yc, bc_headings, v, dt)
    print(psi)
    print(np.rad2deg(psi))
    ax.plot(xs, ys, 'ob', alpha=0.8, markersize=4, color='darkblue')

    plt.show()

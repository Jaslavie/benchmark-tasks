"""RGB views with no labels, user paths, or evaluator overlays."""
import numpy as np
import mujoco

VIEWS = {
    "overview": (120, -25, 1.9, [-.02, 0, 1.0]),
    "closeup": (110, -18, .34, [0, -.025, 1.045]),
    "front": (60, -15, .6, [0, 0, 1.03]),
}


class Views:
    def __init__(self, env, size=480):
        self.env = env
        self.renderer = mujoco.Renderer(env.model, height=size, width=size)

    def image(self, name):
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation, camera.distance, lookat = VIEWS[name]
        camera.lookat[:] = lookat
        self.renderer.update_scene(self.env.data, camera=camera)
        return self.renderer.render().copy()

    def composite(self):
        return np.concatenate([self.image("overview"), self.image("closeup")], axis=1)

    def close(self):
        self.renderer.close()

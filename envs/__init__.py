from stable_baselines3.common.monitor import Monitor

from utils.frame_stack import FrameStack

import gymnasium as gym
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from tactile_envs.utils.resize_dict import ResizeDict
from tactile_envs.utils.add_tactile import AddTactile

import numpy as np

def make_env(
    env_name,
    rank,
    seed=0,
    state_type="vision_and_touch",
    camera_idx=0,
    objects=["square"],
    holders=["holder2"],
    frame_stack=1,
    no_rotation=True,
    skip_frame=2,
    **kwargs,
):
    """
    Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    def _init():
        if env_name in ["Door"]:
            import robosuite as suite
            from robosuite.wrappers.tactile_wrapper import TactileWrapper
            from robosuite import load_controller_config

            config = load_controller_config(default_controller="OSC_POSE")

            # Notice how the environment is wrapped by the wrapper
            if env_name == "Door":
                robots = ["PandaTactile"]
                placement_initializer = None
                init_qpos = [-0.073, 0.016, -0.392, -2.502, 0.240, 2.676, 0.189]
                env_config = kwargs.copy()
                env_config["robot_configs"] = [{"initial_qpos": init_qpos}]
                env_config["initialization_noise"] = None

            env = TactileWrapper(
                suite.make(
                    env_name,
                    robots=robots,  # use PandaTactile robot
                    use_camera_obs=True,  # use pixel observations
                    use_object_obs=False,
                    has_offscreen_renderer=True,  # needed for pixel obs
                    has_renderer=False,  # not needed due to offscreen rendering
                    reward_shaping=True,  # use dense rewards
                    camera_names="agentview",
                    horizon=300,
                    controller_configs=config,
                    placement_initializer=placement_initializer,
                    camera_heights=64,
                    camera_widths=64,
                    **env_config,
                ),
                env_id=rank,
                state_type=state_type,
            )
            env = FrameStack(env, frame_stack)
        elif env_name in ["HandManipulateBlockRotateZFixed-v1", "HandManipulateEggRotateFixed-v1", "HandManipulatePenRotateFixed-v1"]:
            env = gym.make(env_name, render_mode="rgb_array", reward_type='dense')
            env = PixelObservationWrapper(env, pixel_keys=('image',))
            env = ResizeDict(env, 64, pixel_key='image')
            if state_type == "vision_and_touch":
                env = AddTactile(env)
            env = FrameStack(env, frame_stack)
        else:
            
            env = gym.make(
                env_name,
                state_type=state_type,
                camera_idx=camera_idx,
                symlog_tactile=True,
                env_id=rank,
                holders=holders,
                objects=objects,
                no_rotation=no_rotation,
                skip_frame=skip_frame,
            )
            env = FrameStack(env, frame_stack)

        env = Monitor(env)
        np.random.seed(seed + rank)
        return env

    return _init

import wandb
import numpy as np

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.vec_env import DummyVecEnv

import envs
from utils.wandb_logger import WandbLogger
from utils.pretrain_utils import log_videos

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record("rollout/avg_success", np.mean(self.model.ep_success_buffer))
        return True


class EvalCallback(BaseCallback):
    def __init__(
        self,
        env,
        state_type,
        no_tactile=False,
        representation=True,
        eval_every=1,
        verbose=0,
        config=None,
        objects=["square"],
        holders=["holder2"],
        camera_idx=0,
        frame_stack=1,
    ):
        super(EvalCallback, self).__init__(verbose)
        self.n_samples = 4
        self.eval_seed = 100
        self.no_tactile = no_tactile
        self.representation = representation

        env_config = {"use_latch": config.use_latch}

        self.test_env = DummyVecEnv(
            [
                envs.make_env(
                    env,
                    0,
                    self.eval_seed,
                    state_type=state_type,
                    objects=objects,
                    holders=holders,
                    camera_idx=camera_idx,
                    frame_stack=frame_stack,
                    no_rotation=config.no_rotation,
                    **env_config
                )
            ]
        )
        self.count = 0
        self.eval_every = eval_every

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self.count += 1
        if self.count >= self.eval_every:
           
            ret, obses, rewards_per_step = self.eval_model()
            frame_stack = obses[0]["image"].shape[1]
            self.logger.record("eval/return", ret)

            log_videos(
                obses,
                rewards_per_step,
                self.logger,
                self.model.num_timesteps,
                frame_stack=frame_stack,
            )
            self.count = 0

    def eval_model(self):
        print("Collect eval rollout")
        obs = self.test_env.reset()
        dones = [False]
        reward = 0
        obses = []
        rewards_per_step = []
        while not dones[0]:
            action, _ = self.model.predict(obs, deterministic=False)
            obs, rewards, dones, info = self.test_env.step(action)
            reward += rewards[0]
            rewards_per_step.append(rewards[0])
            obses.append(obs)
        
        return reward, obses, rewards_per_step


def create_callbacks(config, model, num_tactiles, objects, holders):
    no_tactile = num_tactiles == 0
    project_name = "MultimodalLearning"
    if config.env in ["Door"]:
        project_name += "_robosuite"

    callbacks = []

    eval_callback = EvalCallback(
        config.env,
        config.state_type,
        no_tactile=no_tactile,
        representation=config.representation,
        eval_every=config.eval_every // config.rollout_length,
        config=config,
        objects=objects,
        holders=holders,
        camera_idx=config.camera_idx,
        frame_stack=config.frame_stack,
    )
    callbacks.append(eval_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.save_freq // config.n_envs, 1),
        save_path="./logs/",
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(TensorboardCallback())

    default_logger = configure_logger(
        verbose=1, tensorboard_log=model.tensorboard_log, tb_log_name="PPO"
    )
    wandb.init(
        project=project_name,
        config=config,
        save_code=True,
        name=default_logger.dir.split("/")[-1],
        dir=config.wandb_dir,
        id=config.wandb_id,
        entity=config.wandb_entity,
    )
    logger = WandbLogger(
        default_logger.dir, default_logger.output_formats, log_interval=1000
    )
    model.set_logger(logger)
    checkpoint_callback.save_path = wandb.run.dir

    return callbacks

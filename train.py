import argparse
import torch

from stable_baselines3 import PPO
from models.ppo_mae import PPO_MAE

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

import tactile_envs
import envs
from utils.callbacks import create_callbacks
from models.pretrain_models import VTT, VTMAE, MAEPolicy

def str2bool(v):
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    raise ValueError(f"boolean argument should be either True or False (got {v})")

def main():
    parser = argparse.ArgumentParser("M3L")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=int(1e5))
    parser.add_argument("--eval_every", type=int, default=int(2e5))
    parser.add_argument("--total_timesteps", type=int, default=int(3e6))
    
    parser.add_argument("--wandb_dir", type=str, default="./wandb/")
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
 
    # Environment.
    parser.add_argument(
        "--env",
        type=str,
        default="tactile_envs/Insertion-v0",
        choices=[
            "tactile_envs/Insertion-v0",
            "Door",
            "HandManipulateBlockRotateZFixed-v1",
            "HandManipulateEggRotateFixed-v1",
            "HandManipulatePenRotateFixed-v1"
        ],
    )
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument(
        "--state_type",
        type=str,
        default="vision_and_touch",
        choices=["vision", "touch", "vision_and_touch"]
    )
    parser.add_argument("--norm_reward", type=str2bool, default=True)
    parser.add_argument("--use_latch", type=str2bool, default=True)
    
    parser.add_argument("--camera_idx", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--no_rotation", type=str2bool, default=True)

    # MAE.
    parser.add_argument("--representation", type=str2bool, default=True)
    parser.add_argument("--early_conv_masking", type=str2bool, default=True)
    
    parser.add_argument("--dim_embedding", type=int, default=256)
    parser.add_argument("--use_sincosmod_encodings", type=str2bool, default=True)
    parser.add_argument("--masking_ratio", type=float, default=0.95)
    
    parser.add_argument("--mae_batch_size", type=int, default=32)
    parser.add_argument("--train_mae_every", type=int, default=1)

    # PPO.
    parser.add_argument("--rollout_length", type=int, default=32768)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--lr_ppo", type=float, default=1e-4)
    parser.add_argument("--vision_only_control", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=512)

    # PPO-MAE.
    parser.add_argument("--separate_optimizer", type=str2bool, default=False)

    config = parser.parse_args()

    set_random_seed(config.seed)

    num_tactiles = 0
    if config.state_type == "vision_and_touch" or config.state_type == "touch":
        num_tactiles = 2
        if config.env == "HandManipulateBlockRotateZFixed-v1" or config.env == "HandManipulateEggRotateFixed-v1" or config.env == "HandManipulatePenRotateFixed-v1":
            num_tactiles = 1
            
    env_config = {
        "use_latch": config.use_latch,
    }
    
    objects = [
        "square",
        "triangle",
        "horizontal",
        "vertical",
        "trapezoidal",
        "rhombus",
    ]
    holders = ["holder1", "holder2", "holder3"]

    env_list = [
        envs.make_env(
            config.env,
            i,
            config.seed,
            config.state_type,
            objects=objects,
            holders=holders,
            camera_idx=config.camera_idx,
            frame_stack=config.frame_stack,
            no_rotation=config.no_rotation,
            **env_config,
        )
        for i in range(config.n_envs)
    ]

    if config.n_envs < 100:
        env = SubprocVecEnv(env_list)
    else:
        env = DummyVecEnv(env_list)
    env = VecNormalize(env, norm_obs=False, norm_reward=config.norm_reward)

    v = VTT(
        image_size=(64, 64),
        tactile_size=(32, 32),
        image_patch_size=8,
        tactile_patch_size=4,
        dim=config.dim_embedding,
        depth=4,
        heads=4,
        mlp_dim=config.dim_embedding * 2,
        num_tactiles=num_tactiles,
        image_channels=3*config.frame_stack,
        tactile_channels=3*config.frame_stack,
        frame_stack=config.frame_stack,
    )

    mae = VTMAE(
        encoder=v,
        masking_ratio=config.masking_ratio, 
        decoder_dim=config.dim_embedding,  
        decoder_depth=3, 
        decoder_heads=4,
        num_tactiles=num_tactiles,
        early_conv_masking=config.early_conv_masking,
        use_sincosmod_encodings=config.use_sincosmod_encodings,
        frame_stack = config.frame_stack
    )
    if torch.cuda.is_available():
        mae.cuda()
    mae.eval()

    if config.representation:
        mae.initialize_training({"lr": 1e-4, "batch_size": config.mae_batch_size})
        
        policy = MAEPolicy
        policy_kwargs={
            "mae_model": mae,
            "dim_embeddings": config.dim_embedding,
            "vision_only_control": config.vision_only_control,
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "frame_stack": config.frame_stack,
        }

        model = PPO_MAE(
            policy,
            env,
            verbose=1,
            learning_rate=config.lr_ppo,
            tensorboard_log=config.wandb_dir+"tensorboard/",
            batch_size=config.batch_size,
            n_steps=config.rollout_length // config.n_envs,
            n_epochs=config.ppo_epochs,
            mae_batch_size=config.mae_batch_size,
            separate_optimizer=config.separate_optimizer,
            policy_kwargs=policy_kwargs,
            mae=mae,
        )
    
        
        callbacks = create_callbacks(
            config, model, num_tactiles, objects, holders
        )
        model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
    else:
        
        model = PPO(
            MAEPolicy,
            env,
            verbose=1,
            learning_rate=config.lr_ppo,
            tensorboard_log=config.wandb_dir+"ppo_privileged_tensorboard/",
            batch_size=config.batch_size,
            n_steps=config.rollout_length // config.n_envs,
            n_epochs=config.ppo_epochs,
            policy_kwargs={
                "mae_model": mae,
                "dim_embeddings": config.dim_embedding,
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
                "frame_stack": config.frame_stack,
            },
        )
        callbacks = create_callbacks(
            config, model, num_tactiles, objects, holders
        )
        model.learn(total_timesteps=config.total_timesteps, callback=callbacks)


if __name__ == "__main__":
    main()

import cv2
import numpy as np
from tqdm import tqdm
import torch
from stable_baselines3.common.logger import Video

def vt_load(
    x, image_normalization=[0, 1], tactile_normalization=[-1, 1], squeeze=False, frame_stack=1
):
    ### Load and normalize to [0,1] ###

    if isinstance(x, str):
        path = x
        x = np.load(path, allow_pickle=True).item()

    if "image" in x:
        if len(x["image"].shape) == 3:
            x["image"] = x["image"][None, :, :, :]  # Add batch dimension
    
    if "tactile" in x:
        if len(x["tactile"].shape) == 3:
            x["tactile"] = x["tactile"][None, :, :, :]  # Add batch dimension

    # Preprocess the image
    if "image" in x:
        assert x["image"].shape[-1] == 3*frame_stack
        x["image"] = torch.Tensor(x["image"]).permute(0, 3, 1, 2)
        x["image"] = (x["image"] - image_normalization[0]) / (
            image_normalization[1] - image_normalization[0]
        )
    
    # Preprocess the tactile
    if "tactile" in x:
        assert x["tactile"].shape[1] == 3*frame_stack or x["tactile"].shape[1] == 6*frame_stack or x["tactile"].shape[1] == 12*frame_stack

        idx = []
        n_tactiles = x["tactile"].shape[1] // frame_stack
        for i in range(frame_stack):
            idx.append(i*n_tactiles+0)
            idx.append(i*n_tactiles+1)
            idx.append(i*n_tactiles+2)
        idx = np.array(idx).flatten()
        
        n_sensors = n_tactiles//3
        for tactile_idx in range(n_sensors):
            x["tactile"+str(tactile_idx+1)] = torch.Tensor(x["tactile"][:, idx+3*tactile_idx])
            x["tactile"+str(tactile_idx+1)] = (x["tactile"+str(tactile_idx+1)] - tactile_normalization[0]) / (
                tactile_normalization[1] - tactile_normalization[0]
            )

        del x["tactile"]
    
    if squeeze:
        for key in x:
            x[key] = x[key].squeeze()

    return x


def train(model, train_loader, optimizer, epoch, writer, normalize_image=False):
    model.train()

    t = tqdm(train_loader, desc="Iteration".format(ncols=80))

    for iter, data in enumerate(t):
        x = data[0]

        if normalize_image:
            x = x.float() / 255
        if torch.cuda.is_available():
            if isinstance(x, dict):
                for key in x:
                    x[key] = x[key].cuda()
            else:
                x = x.cuda()
        optimizer.zero_grad()
        r_loss = model(x)
        r_loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", r_loss, epoch * len(train_loader) + iter)

        t.set_description(
            "Epoch {}, rloss: {}, lr: {}, Progress: ".format(
                epoch, r_loss, optimizer.param_groups[0]["lr"]
            )
        )


def eval_loss(model, loader, normalize_image=False):
    model.eval()

    r_loss = 0
    with torch.no_grad():
        for data in loader:
            x = data[0]
            if normalize_image:
                x = x.float() / 255
            if torch.cuda.is_available():
                if isinstance(x, dict):
                    for key in x:
                        x[key] = x[key].cuda()
                else:
                    x = x.cuda()
            r_loss = model(x).item() * len(x)

    return r_loss / len(loader.dataset)

def annotate_frame(step, frame, rew, info={}):

    """Renders a video frame and adds caption."""
    if np.max(frame) <= 1.0:
        frame *= 255.0
    frame = frame.astype(np.uint8)

    # Set the minimum size of frames to (`S`, `S`) for caption readibility.
    # S = 512
    S = 128
    if frame.shape[0] < S:
        frame = cv2.resize(frame, (int(S * frame.shape[1] / frame.shape[0]), S))
    h, w = frame.shape[:2]

    # Add caption.
    frame = np.concatenate([frame, np.zeros((64, w, 3), np.uint8)], 0)
    scale = h / S
    font_size = 0.4 * scale
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(5 * scale), h + int(10 * scale)
    add_text = lambda x, y, c, t: cv2.putText(
        frame, t, (x, y), font_face, font_size, c, 1, cv2.LINE_AA
    )

    add_text(x, y, (255, 255, 0), f"{step:5} {rew:.3f}")
    for i, k in enumerate(info.keys()):
        key_text = f"{k}: "
        key_width = cv2.getTextSize(key_text, font_face, font_size, 1)[0][0]
        offset = int(12 * scale) * (i + 2)
        add_text(x, y + offset, (66, 133, 244), key_text)
        value_text = str(info[k])
        if isinstance(info[k], np.ndarray):
            value_text = np.array2string(
                info[k], precision=2, separator=", ", floatmode="fixed"
            )
        add_text(x + key_width, y + offset, (255, 255, 255), value_text)

    return frame

def log_videos(
    obses,
    rewards_per_step,
    logger,
    num_timesteps,
    frame_stack=1
):
    
    image_video = []
    reward_video = []
    
    episode_return = 0.0
    for (x, reward) in zip(obses, rewards_per_step):  # For each timestep
        
        if 'image' in x:
            x['image'] = x['image'].transpose((0, 2, 3, 1, 4))
            x['image'] = x['image'].reshape((x['image'].shape[0], x['image'].shape[1], x['image'].shape[2], -1))
        if 'tactile' in x:
            x['tactile'] = x['tactile'].reshape((x['tactile'].shape[0], -1, x['tactile'].shape[3], x['tactile'].shape[4]))
        
        x = vt_load(x, frame_stack=frame_stack)
        
        if torch.cuda.is_available():
            for key in ['image', 'tactile1', 'tactile2']:
                if key in x:
                    x[key] = x[key].cuda()

        # if frame_stack > 1: # image is (1, frame_stack*channels, height, width)
        image = x["image"].reshape((frame_stack, -1, x["image"].shape[2], x["image"].shape[3]))
        image = image[-1].detach().cpu()
        
        image_video.append(image)

        episode_return += reward
        reward_video.append(episode_return)

    image_video = [
        annotate_frame(i, frame.numpy().transpose(1, 2, 0), reward_video[i])
        for i, frame in enumerate(image_video)
    ]

    logger.record(
        "eval/image_video",
        Video(np.stack(image_video).transpose(0, 3, 1, 2)[None], fps=int(40)),
        exclude=("stdout", "log", "json", "csv"),
    )

    logger.dump(step=num_timesteps)

    return True


# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

import argparse
import logging
import os
import time
import warnings

warnings.filterwarnings("ignore")  # mute warnings, live dangerously ;)

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import cv2

from torchbeast.monobeast import create_env
from torchbeast.core.environment import Environment
from torchbeast.models.attention_augmented_agent import AttentionAugmentedAgent

cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser(
    description="Visualizations for the Attention-Augmented Agent"
)

parser.add_argument(
    "--model_load_path",
    default="./logs/torchbeast",
    help="Path to the model that should be used for the visualizations.",
)
parser.add_argument(
    "--env", type=str, default="PongNoFrameskip-v4", help="Gym environment."
)
parser.add_argument(
    "--frame_height", type=int, default=84, help="Height to which frames are rescaled."
)
parser.add_argument(
    "--frame_width", type=int, default=84, help="Width to which frames are rescaled."
)
parser.add_argument(
    "--aaa_input_format",
    type=str,
    default="gray_stack",
    choices=["gray_stack, rgb_last, rgb_stack"],
    help="Color format of the frames as input for the AAA.",
)
parser.add_argument("--num_frames", default=50, type=int, help=".")
parser.add_argument("--first_frame", default=200, type=int, help=".")
parser.add_argument("--resolution", default=75, type=int, help=".")
parser.add_argument("--density", default=2, type=int, help=".")
parser.add_argument("--radius", default=2, type=int, help=".")
parser.add_argument("--save_dir", default="~/logs/aaa-vis", help=".")

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def extract_data(data):
    if type(data) is tuple:
        return tuple(extract_data(d) for d in data)
    elif type(data) is dict:
        return None
    return data.detach()


def rollout(model, env, max_ep_len=3e3, render=False):
    history = {
        "observation": [],
        "policy_logits": [],
        "baseline": [],
        "agent_state": [],
        "image": [],
        "attention_maps": [],
    }
    episode_length, epr, eploss, done = 0, 0, 0, False  # bookkeeping

    activations = {}

    def get_activation(layer_name):
        if layer_name not in activations:
            activations[layer_name] = []

        def hook(model, input, output):
            activations[layer_name].append(extract_data(output))

        return hook

    for name, module in model.named_modules():
        module.register_forward_hook(get_activation(name))

    observation = env.initial()
    with torch.no_grad():
        agent_state = model.initial_state(batch_size=1)
        while not done and episode_length <= max_ep_len:
            episode_length += 1
            agent_output, agent_state, attention_maps = model(
                observation, agent_state, return_attention_maps=True
            )
            observation = env.step(agent_output["action"])
            done = observation["done"]

            history["observation"].append(observation)
            history["policy_logits"].append(
                agent_output["policy_logits"].detach().numpy()[0]
            )
            history["baseline"].append(agent_output["baseline"].detach().numpy()[0])
            history["agent_state"].append(tuple(s.data.numpy()[0] for s in agent_state))
            history["attention_maps"].append(attention_maps.detach().numpy()[0])
            history["image"].append(env.gym_env.render(mode="rgb_array"))
    history["activations"] = activations

    return history


def visualize_aaa(model, env, flags):
    video_title = "{}_{}_{}_{}.mp4".format(
        "aaa-vis", flags.env, flags.first_frame, flags.num_frames
    )
    max_ep_len = flags.first_frame + flags.num_frames + 1
    torch.manual_seed(0)
    history = rollout(model, env, max_ep_len=max_ep_len)

    start = time.time()
    ffmpeg_writer = manimation.writers["ffmpeg"]
    metadata = dict(
        title=video_title, artist="", comment="atari-attention-augmented-agent-video"
    )
    writer = ffmpeg_writer(fps=8, metadata=metadata)

    total_frames = len(history["observation"])
    f = plt.figure(figsize=[(4 / 1.3) * 2, 4], dpi=flags.resolution)
    axis_f = f.add_subplot(1, 2, 1)
    axis_a = f.add_subplot(1, 2, 2)
    axis_f.axis("off")
    axis_a.axis("off")

    video_path = os.path.expandvars(os.path.expanduser(flags.save_dir))
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    with writer.saving(f, video_path + "/" + video_title, flags.resolution):
        for i in range(flags.num_frames):
            ix = flags.first_frame + i
            if (
                ix < total_frames
            ):  # prevent loop from trying to process a frame ix greater than rollout length
                frame = history["image"][ix]
                attention_maps = history["attention_maps"][ix]
                attention_map = attention_maps[:, :, 0]
                attention_map = cv2.resize(
                    attention_map,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                axis_f.imshow(frame)
                axis_a.imshow(attention_map, cmap="gray")
                f.suptitle(flags.env, fontsize=15, fontname="DejaVuSans")

                writer.grab_frame()
                f.clear()
                axis_f = f.add_subplot(1, 2, 1)
                axis_a = f.add_subplot(1, 2, 2)
                axis_f.axis("off")
                axis_a.axis("off")

                time_str = time.strftime(
                    "%Hh %Mm %Ss", time.gmtime(time.time() - start)
                )
                print(
                    "\ttime: {} | progress: {:.1f}%".format(
                        time_str, 100 * i / min(flags.num_frames, total_frames)
                    ),
                    end="\r",
                )
    print("\nFinished.")


if __name__ == "__main__":
    flags = parser.parse_args()

    gym_env = create_env(
        flags.env,
        frame_height=flags.frame_height,
        frame_width=flags.frame_width,
        gray_scale=(flags.aaa_input_format == "gray_stack"),
    )
    env = Environment(gym_env)
    model = AttentionAugmentedAgent(
        gym_env.observation_space.shape,
        gym_env.action_space.n,
        rgb_last=(flags.aaa_input_format == "rgb_last"),
    )
    model.eval()
    checkpoint = torch.load(flags.model_load_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    logging.info("Visualizing AAA using checkpoint at %s.", flags.model_load_path)
    visualize_aaa(model, env, flags)

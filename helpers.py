import json
from ai2thor.controller import Controller
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image
import numpy as np
import copy
import random


def angular_std_dev(angles):
    # Convert angles to radians
    angles_rad = np.radians(angles)

    # Convert each angle to a vector on the unit circle
    x = np.cos(angles_rad)
    y = np.sin(angles_rad)

    # Compute the mean vector
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Compute the mean angle
    mean_angle = np.arctan2(mean_y, mean_x)

    # Calculate angular deviations
    angular_devs = np.arctan2(
        np.sin(angles_rad - mean_angle), np.cos(angles_rad - mean_angle)
    )

    # Calculate the standard deviation
    std_dev = np.sqrt(np.mean(angular_devs**2))
    return std_dev


def get_top_down_frame(controller: Controller):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )

    top_down_frame = event.events[0].third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)


def visualize_frames(
    rgb_frames: List[np.ndarray], title: str = "", figsize: Tuple[int, int] = (8, 2)
):
    """Plots the rgb_frames for each agent."""
    fig, axs = plt.subplots(
        1, len(rgb_frames), figsize=figsize, facecolor="white", dpi=300
    )
    for i, frame in enumerate(rgb_frames):
        ax = axs[i]
        ax.imshow(frame)
        ax.set_title(f"AgentId: {i}")
        ax.axis("off")
    if title:
        fig.suptitle(title)
    return fig


def visualize_frame(frame: np.ndarray, figsize: Tuple[int, int] = (2, 2)):
    """Plots the rgb_frames for each agent."""
    fig, axs = plt.subplots(1, 1, figsize=figsize, facecolor="white", dpi=500)
    axs.imshow(frame)
    axs.axis("off")

    return fig


def reachable_positions_map(controller: Controller, i: int):
    event = controller.step(action="GetReachablePositions")
    event.metadata["actionReturn"]
    reachable_positions = event.metadata["actionReturn"]

    subset = random.choices(reachable_positions, k=500)
    xs = [rp["x"] for rp in subset]
    zs = [rp["z"] for rp in subset]

    fig, ax = plt.subplots(1, 1)
    ax.scatter(xs, zs)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    # ax.set_title("Reachable Positions in the Scene")
    ax.set_aspect("equal")
    fig.savefig(f"reachable{str(i)}.png")


# img = get_top_down_frame()
# img.save("topdown.png")

# img2 = Image.fromarray(event.frame)
# img2.save("rand.png")


# type(controller.last_event)
# controller.last_event.events
# rgb_frames = [event.frame for event in controller.last_event.events]

# fig = visualize_frames(rgb_frames)
# fig.savefig()

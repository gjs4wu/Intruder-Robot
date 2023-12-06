import argparse
from collections import OrderedDict
import pprint
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict, Sequence, Tuple
import numpy as np
from math import dist
import os
import random

import ray
from ray import air, tune, train
from ray.rllib.env.env_context import EnvContext
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.utils.typing import TensorStructType

from thor_env import initializeEnvironment
from intruder import Intruder
from robot import Robot
from math import inf, trunc
from ai2thor.controller import Controller
from PIL import Image
from helpers import angular_std_dev

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

# Check for torch
if torch == None:
    print("Torch not found")
    raise Exception


class AI2ThorEnv(gym.Env):
    def __init__(self, config: EnvContext):
        super().__init__()
        self.ROBOT_COUNT: int = config["robot_count"]  # Should always be 1
        self.MAX_STEPS: int = config["max_steps"]
        self.controller: Controller | None = None
        self.step_count = 0  # Keep track of # of steps for current environment
        self.iter = 0  # Keep track of current iteration
        self.cum_reward = 0  # Cumulative reward across episodes
        self.seen_objects = (
            []
        )  # All object_id seen by the agent for the current environment
        self.last_observations = None
        self.prev_positions: list[list[float]] = []  # 50 previous positions
        self.prev_angles: list[float] = []  # 50 previous angles
        self.visited_positions: set[
            str
        ] = set()  # All positions visited in current episode

        self.curr_frame = 0
        self.curr_house = 0
        self.env_id = 0
        self.collisions = 0

        self.action_space = Discrete(6)  # 6 possible actions
        self.observation_space = Box(  # Observation space contains 303 floats:
            low=-inf,  #       Current robot position [x, z]
            high=inf,  #       Current robot rotation [y]
            shape=(303, 1),  #       100 random reachable positions [x,z]
            dtype=np.float32,  #       50 previous robot positions [x,z]
        )

        self.reset(seed=config.worker_index * config.num_workers + random.randint(1, 100))  # type: ignore

    def step(self, action: int):
        done = False

        # Perform action
        if action == 0:
            ev = self.robots[0].moveAhead()
        elif action == 1:
            ev = self.robots[0].moveBack()
        elif action == 2:
            ev = self.robots[0].moveRight()
        elif action == 3:
            ev = self.robots[0].moveLeft()
        elif action == 1:
            ev = self.robots[0].rotateRight()
        elif action == 2:
            ev = self.robots[0].rotateLeft()
        else:
            ev = self.robots[0].controller.step("Done")

        # Calculate number of new objects seen by robot
        new_objects = 0
        for obj in ev.metadata["objects"]:
            if obj["visible"]:
                if not obj["name"] in self.seen_objects:
                    new_objects += 1
                    self.seen_objects.append(obj["name"])

        pos = [
            self.robots[0].position["x"],
            self.robots[0].position["z"],
        ]  # Current robot position

        # Add position to position history
        if len(self.prev_positions) >= 50:
            self.prev_positions.pop(0)
        self.prev_positions.append(pos)

        # Add rotation to angle history
        if len(self.prev_angles) >= 50:
            self.prev_angles.pop(0)
        self.prev_angles.append(self.robots[0].rotation["y"])

        # Check if position has been visited before
        visited = True
        pos_str = "".join(map(str, pos))
        if pos_str not in self.visited_positions:
            self.visited_positions.add(pos_str)
            visited = False

        # Check if robot can see intruder
        if self.robots[0].isIntruderVisible(ev):
            print("Found intruder")
            reward = 300
            done = True
        else:
            reward = 0

            if action == 0 and has_error(ev):
                reward = -10
            else:
                stdev = np.std(self.prev_positions, axis=0)
                reward += 7.5 * (stdev[0] + stdev[1]) - 2.5
                ang_std = angular_std_dev(self.prev_angles)
                reward += 5 * ang_std

                reward += new_objects
                reward += 4 if not visited else 0

                if action == 0 and close_to_wall(ev):
                    reward -= 10
                reward -= 1

        observations = (
            [
                self.robots[0].position["x"],
                self.robots[0].position["z"],
                self.robots[0].rotation["y"],
            ]
            + [i for j in [[i[0], i[2]] for i in self.reachable_positions] for i in j]
            + [i for j in self.prev_positions for i in j]
        )

        if len(observations) < 303:
            observations += [*self.prev_positions[-1]] * int(
                (303 - len(observations)) / 2
            )

        self.last_observations = observations

        self.step_count += 1

        if has_error(ev):
            self.collisions += 1

        # self.cum_reward += reward

        # print("Episode Cumulative Reward:", self.cum_reward)
        info = {
            "steps": self.step_count,
            "env_id": self.env_id,
            "collisions": self.collisions,
        }
        return (
            np.array([observations]).transpose(),
            reward,
            done,
            self.step_count == self.MAX_STEPS,
            info,
        )

    def reset(self, *, seed=None, options=None):
        self.iter += 1
        print()
        print("----------------------------")
        print("Iteration:", self.iter)
        random.seed(seed)

        self.env_id = random.randint(0, 9999)
        reachable = []
        (
            self.controller,
            self.intruder,
            self.robots,
            reachable,
            self.env_id,
        ) = initializeEnvironment(
            self.env_id, self.ROBOT_COUNT, self.controller if self.controller else None
        )
        self.curr_house = self.env_id
        print(f"Running agent on House %d" % self.env_id)

        self.reachable_positions = [list(pos.values()) for pos in reachable]

        observations = (
            [
                self.robots[0].position["x"],
                self.robots[0].position["z"],
                self.robots[0].rotation["y"],
            ]
            + [i for j in [[i[0], i[2]] for i in self.reachable_positions] for i in j]
            + [self.robots[0].position["x"], self.robots[0].position["z"]] * 50
        )

        print(
            f"Intruder at position: (%f, %f, %f)"
            % tuple(self.intruder.position.values())
        )

        self.last_observations = observations
        self.visited_positions = set()
        self.prev_positions = []
        self.prev_angles = []
        self.cum_reward = 0
        self.seen_objects = []
        self.curr_frame = 0
        self.step_count = 0
        self.collisions = 0
        info = {"steps": 0, "env_id": self.env_id, "collisions": self.collisions}
        return np.array([observations]).transpose(), info

    def render(self, mode="human"):
        # Implement rendering if necessary
        pass

    def close(self):
        if self.controller:
            self.controller.stop()

    def get_frames(self):
        self.intruder.setAgent()
        event = self.intruder.controller.step("Done")

        for i in range(self.ROBOT_COUNT):
            if self.curr_frame == 0:
                os.makedirs(
                    f"./frames/robot%d/house%d" % (i, self.curr_house), exist_ok=True
                )
            frame = event.third_party_camera_frames[i]
            Image.fromarray(frame).save(
                f"./frames/robot%d/house%d/frame%d.png"
                % (i, self.curr_house, self.curr_frame)
            )

        self.robots[0].setAgent()
        self.intruder.setPlaceholder()
        self.curr_frame += 1


def close_to_wall(event):
    for obj in event.metadata["objects"]:
        if obj["visible"] and obj["distance"] < 1:
            obj_type = obj["name"].partition("|")[0]
            if obj_type == "wall" or obj_type == "door" or obj_type == "window":
                # print("Too close to", obj["name"])
                return True
    return False


def has_error(event):
    return event.metadata["errorMessage"] != ""


if __name__ == "__main__":
    ray.init(num_gpus=1)

    config = (
        get_trainable_cls("PPO")
        .get_default_config()
        .environment(AI2ThorEnv, env_config={"robot_count": 1, "max_steps": 256})
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=1)
    )

    config["num_sgd_iter"] = 30
    config["sgd_minibatch_size"] = 128
    config["num_cpus_per_learner_worker"] = 0
    config["num_cpus_per_worker"] = 5
    config["train_batch_size"] = 16384 / 2
    config["model"]["fcnet_hiddens"] = [256, 256]
    config["num_gpus_per_learner_worker"] = 1
    config["num_gpus_per_worker"] = 0

    stop = {
        "training_iteration": 10,
        "episode_reward_mean": 50000,
    }

    print("Training automatically with Ray Tune")
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop,
            # failure_config=train.FailureConfig(fail_fast=True),
        ),
    )
    # tuner = tuner.restore(path="/home/gabriel/ray_results/PPO", trainable="PPO")

    results = tuner.fit()

    best_result = results.get_best_result()

    loaded_ppo = Algorithm.from_checkpoint(best_result.checkpoint)  # type: ignore
    loaded_policy = loaded_ppo.get_policy()
    testing = AI2ThorEnv(
        EnvContext({"robot_count": 1, "max_steps": 1000}, worker_index=0, num_workers=1)
    )

    for _ in range(256):
        print("Choosing action:", end=" ")
        action, _, _ = loaded_policy.compute_single_action(
            obs=testing.last_observations
        )
        print(action)
        obs, reward, done, term, _ = testing.step(action)  # type: ignore
        if done:
            print("I DID IT")
            break

    ray.shutdown()

import random
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_context import EnvContext
from rl import AI2ThorEnv
from math import dist


def test():
    loaded_ppo = Algorithm.from_checkpoint(
        "FILENAME"
    )
    loaded_policy = loaded_ppo.get_policy()

    info = {}
    done = False
    ray.shutdown()
    ray.init(num_gpus=1)

    TESTING_ID = 72

    for i in range(1):
        testing = AI2ThorEnv(
            EnvContext(
                {"robot_count": 1, "max_steps": 256},
                worker_index=random.randint(1, 10000),
                num_workers=1,
            )
        )

        start_dist = dist(
            [testing.intruder.position["x"], testing.intruder.position["z"]],
            {testing.robots[0].position["x"], testing.robots[0].position["z"]},
        )

        for _ in range(256):
            print("Choosing action:")
            action, _, _ = loaded_policy.compute_single_action(
                obs=testing.last_observations
            )
            print(action)
            obs, reward, done, term, info = testing.step(action)  # type: ignore
            if done:
                print("I DID IT")
                break

        if done:
            with open(f"results{TESTING_ID}.txt", "a") as f:
                f.write(
                    f'{info["env_id"]}:{info["steps"]}:{start_dist}:{info["collisions"]}:\n'
                )
        else:
            with open(f"results{TESTING_ID}.txt", "a") as f:
                f.write(f'{info["env_id"]}:-1:{start_dist}:{info["collisions"]}:\n')

        if testing.controller:
            testing.controller.stop()


if __name__ == "__main__":
    test()

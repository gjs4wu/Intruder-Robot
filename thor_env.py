import termios, fcntl, sys, os
from typing import List
from ai2thor.controller import Controller
from PIL import Image
import random
import json
import matplotlib.pyplot as plt
from robot import Robot
from intruder import Intruder


def initializeEnvironment(
    env_id: int, robotCount: int, ctrl: Controller | None = None
) -> tuple[Controller, Intruder, list[Robot], list[dict[str, float]], int]:
    controller: Controller | None = None
    if ctrl:
        controller = ctrl

    # Pick random environment
    filename = f"./houses/house_%d.json" % env_id
    with open(filename, "r") as f:
        house = json.load(f)
        s = house["objects"][-1]["id"].split("|")
        intruder_id = f"BasketBall|%s|%d" % (s[1], int(s[2]) + 1)
        placeholder = {
            "assetId": "Basketball_1",
            "id": intruder_id,
            "kinematic": False,
            "position": {
                "x": 0,
                "y": 0,
                "z": 0,
            },
            "rotation": {"x": 0, "y": 0, "z": 0},
            # "layer": "Procedural2",
            "material": None,
        }
        house["objects"].append(placeholder)

    # Initalize AI2Thor Environment
    DISPLAY_SIZE = 700
    GRID_SIZE = 0.05
    if controller:
        print("Resetting...")
        controller.reset(
            scene=house,
            gridSize=GRID_SIZE,
            width=DISPLAY_SIZE,
            height=DISPLAY_SIZE,
            visibilityDistance=5,
        )
    else:
        controller = Controller(
            scene=house,
            gridSize=GRID_SIZE,
            width=DISPLAY_SIZE,
            height=DISPLAY_SIZE,
            visibilityDistance=5,
        )

    # print("Getting positions...")
    try:
        e = controller.step(action="GetReachablePositions")
    except:
        print("ERROR: No reachable positions found, trying again")
        env_id = random.randint(0, 9999)
        return initializeEnvironment(env_id, robotCount, controller)

    reachable_positions: list[dict[str, float]] = (
        e.metadata["actionReturn"] if e.metadata["actionReturn"] else []
    )

    # print("Number of valid positions:", len(reachable_positions))

    if not reachable_positions or len(reachable_positions) == 0:
        print("No reachable positions found, trying again")
        env_id = random.randint(0, 9999)
        return initializeEnvironment(env_id, robotCount, controller)

    object_poses: list[dict[str, str | dict[str, float]]] = []
    for obj in e.metadata["objects"]:
        if obj["name"] != intruder_id:
            object_poses.append(
                {
                    "objectName": obj["name"],
                    "rotation": obj["rotation"],
                    "position": obj["position"],
                }
            )

    # reachable_positions_map(controller, env_id)

    # Initialize Intruder
    intruder = Intruder(controller, intruder_id, reachable_positions, object_poses)

    # Initialize security robots
    robots: List[Robot] = []
    for i in range(robotCount):
        robots.append(Robot(i, controller, intruder_id, reachable_positions))
    intruder.setPlaceholder()

    return (
        controller,
        intruder,
        robots,
        random.choices(reachable_positions, k=100),
        env_id,
    )


if __name__ == "__main__":
    ROBOT_COUNT = 1

    controller, intruder, robots, reachable_positions, env_id = initializeEnvironment(
        random.randint(0, 9999), ROBOT_COUNT
    )

    # reachable_positions_map(controller)

    fd = sys.stdin.fileno()
    oldterm = termios.tcgetattr(fd)  # type: ignore
    newattr = termios.tcgetattr(fd)  # type: ignore
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO  # type: ignore
    termios.tcsetattr(fd, termios.TCSANOW, newattr)  # type: ignore

    oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)  # type: ignore
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)  # type: ignore

    print("Ready to move")
    try:
        while 1:
            try:
                c = sys.stdin.read(1)
                if c:
                    if c == "t":
                        controller.stop()
                        break

                    if c == "m":
                        intruder.setAgent()
                        event = intruder.controller.step("Done")

                        for i in range(ROBOT_COUNT):
                            frame = event.third_party_camera_frames[i]
                            Image.fromarray(frame).save("Agent" + str(i) + ".png")

                        robots[0].setAgent()
                        intruder.setPlaceholder()
                        # robots[0].setAgent()

                    if c == "w":
                        robots[0].moveAhead()
                    if c == "s":
                        robots[0].moveBack()
                    if c == "a":
                        robots[0].moveLeft()
                    if c == "d":
                        robots[0].moveRight()
                    if c == "q":
                        robots[0].rotateLeft()
                    if c == "e":
                        robots[0].rotateRight()

                    if c == "i":
                        robots[-1].moveAhead()
                    if c == "k":
                        robots[-1].moveBack()
                    if c == "j":
                        robots[-1].moveLeft()
                    if c == "l":
                        robots[-1].moveRight()
                    if c == "u":
                        robots[-1].rotateLeft()
                    if c == "o":
                        robots[-1].rotateRight()

                    # intruder.setAgent()

            except IOError:
                pass
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)  # type: ignore
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)  # type: ignore


# with open('temp.json', 'w') as f:
#     json.dump(house, f)

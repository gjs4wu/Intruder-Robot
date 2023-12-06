import json
from ai2thor.controller import Controller
from ai2thor.server import MultiAgentEvent
import random


class Robot(object):
    """Security Robot"""

    def __init__(
        self,
        agent_num: int,
        ctrl: Controller,
        intruder_id: str,
        starting_positions: list[dict[str, float]],
    ):
        self.id = agent_num
        self.name = "robot" + str(agent_num)
        self.controller = ctrl
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.rotation = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.cam_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.intruder_id = intruder_id
        self.set = False
        self.initializePosition(starting_positions)

    def initializePosition(self, starting_positions: list[dict[str, float]]):
        self.position = random.choice(starting_positions)
        self.rotation["y"] = float(random.choice(range(360)))

        event = self.controller.step(
            action="Teleport", position=self.position, rotation=self.rotation
        )

        print(
            "Robot",
            self.id,
            "at",
            self.position,
        )

        self.cam_position = event.metadata["cameraPosition"]

        self.controller.step(
            action="AddThirdPartyCamera",
            position=self.cam_position,
            rotation=self.rotation,
        )

    def setAgent(self):
        print_error(
            self.controller.step(
                action="Teleport", position=self.position, rotation=self.rotation
            )
        )

    def updatePosition(self, event: MultiAgentEvent):
        self.position = event.metadata["agent"]["position"]
        self.rotation = event.metadata["agent"]["rotation"]

    def updateCamera(self, event: MultiAgentEvent):
        self.cam_position = event.metadata["cameraPosition"]

        event = self.controller.step(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=self.id,
            rotation=self.rotation,
            position=self.cam_position,
        )

    def moveAhead(self):
        # print(f"Move agent %d ahead" % self.id)
        # self.setAgent()
        event = self.controller.step(action="MoveAhead", moveMagnitude=0.25)
        self.updatePosition(event)
        self.updateCamera(event)
        # print_error(event)
        return event

    def moveBack(self):
        # print(f"Move agent %d back" % self.id)
        # self.setAgent()
        event = self.controller.step(action="MoveBack", moveMagnitude=0.25)
        self.updatePosition(event)
        self.updateCamera(event)
        # print_error(event)
        return event

    def moveRight(self):
        # print(f"Move agent %d right" % self.id)
        # self.setAgent()
        event = self.controller.step(action="MoveRight", moveMagnitude=0.25)
        self.updatePosition(event)
        self.updateCamera(event)
        # print_error(event)
        return event

    def moveLeft(self):
        # print(f"Move agent %d left" % self.id)
        # self.setAgent()
        event = self.controller.step(action="MoveLeft", moveMagnitude=0.25)
        self.updatePosition(event)
        self.updateCamera(event)
        # print_error(event)
        return event

    def rotateRight(self):
        # print(f"Rotate agent %d right" % self.id)
        # self.setAgent()
        event = self.controller.step(action="RotateRight", degrees=15)
        self.updatePosition(event)
        self.updateCamera(event)
        # print_error(event)
        return event

    def rotateLeft(self):
        # print(f"Rotate agent %d left" % self.id)
        # self.setAgent()
        event = self.controller.step(action="RotateLeft", degrees=15)
        self.updatePosition(event)
        self.updateCamera(event)
        # print_error(event)
        return event

    def isIntruderVisible(self, event):
        for obj in event.metadata["objects"]:
            if obj["name"] == self.intruder_id:
                if obj["visible"]:
                    print("Agent", self.id, "can see the intruder")
                    return True
        return False

    def getReachablePositions(self) -> list[dict[str, float]]:
        reachable = self.controller.step(action="GetReachablePositions").metadata[
            "actionReturn"
        ]
        if not reachable:
            return []
        return random.choices(reachable, k=100)


def print_error(event):
    if event.metadata["errorMessage"] != "":
        print(event.metadata["errorMessage"])


def dump_event(event):
    with open("event.json", "w") as f:
        json.dump(event.metadata, f)

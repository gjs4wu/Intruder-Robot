from ai2thor.controller import Controller
import random

class Intruder(object):
    def __init__(
        self,
        ctrl: Controller,
        intruder_id: str,
        starting_positions: list[dict[str, float]],
        object_poses: list[dict[str, str | dict[str, float]]],
    ):
        self.id = 0
        self.name = "intuder"
        self.controller = ctrl
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.rotation = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.cam_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.intruder_id = intruder_id
        self.object_poses = object_poses
        self.placeholder_set = False
        self.initializePosition(starting_positions)

    def initializePosition(self, starting_positions: list[dict[str, float]]):
        self.position = random.choice(starting_positions)
        self.rotation["y"] = float(random.choice(range(360)))

        event = self.controller.step(
            action="Teleport", position=self.position, rotation=self.rotation
        )

        print("Intruder at", self.position)

        self.cam_position = event.metadata["cameraPosition"]

        self.object_poses.append(
            {
                "objectName": self.intruder_id,
                "position": self.cam_position,
                "rotation": self.rotation,
            }
        )

    def setAgent(self):
        self.unsetPlaceholder()
        self.controller.step(
            action="Teleport", position=self.position, rotation=self.rotation
        )

    def setPlaceholder(self):
        if not self.placeholder_set:
            self.controller.step(action="SetObjectPoses", objectPoses=self.object_poses)
            self.placeholder_set = True

        self.controller.step(
            action="EnableObject",
            objectId=self.intruder_id,
        )

    def unsetPlaceholder(self):
        self.controller.step(
            action="DisableObject",
            objectId=self.intruder_id,
        )
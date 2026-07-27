import torch

from src.hereditary_resnet18 import HereditaryResNet18


class RelationshipPredictor:

    def __init__(
        self,
        checkpoint_path="checkpoints/best_hereditary_resnet18.pth"
    ):

        self.device = torch.device("cpu")

        self.class_names = {
            0: "Father-Son",
            1: "Father-Daughter",
            2: "Mother-Son",
            3: "Mother-Daughter"
        }

        self.model = HereditaryResNet18(num_classes=4)

        self.model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=self.device
            )
        )

        self.model.eval()

    def predict(self, img1_tensor, img2_tensor):

        with torch.no_grad():

            outputs = self.model(
                img1_tensor,
                img2_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        class_id = predicted.item()

        return {
            "relationship": self.class_names[class_id],
            "confidence": confidence.item(),
            "class_id": class_id
        }
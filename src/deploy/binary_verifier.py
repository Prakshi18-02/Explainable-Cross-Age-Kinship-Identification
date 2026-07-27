import torch

from src.binary_fusion_model import BinaryFusionModel


class BinaryKinVerifier:

    def __init__(
        self,
        checkpoint_path="checkpoints/best_binary_fusion.pth"
    ):

        self.device = torch.device("cpu")

        self.model = BinaryFusionModel()

        self.model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=self.device
            )
        )

        self.model.eval()

    def predict(
        self,
        face1,
        face2,
        eyes1,
        nose1,
        lips1,
        eyes2,
        nose2,
        lips2
    ):

        with torch.no_grad():

            output, weights = self.model(
                face1,
                face2,
                eyes1,
                nose1,
                lips1,
                eyes2,
                nose2,
                lips2
            )

            kin_prob = torch.sigmoid(output).item()

        return kin_prob, weights
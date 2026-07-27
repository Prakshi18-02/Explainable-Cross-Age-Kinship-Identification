import os
import cv2
import torch
import numpy as np


class GradCAMExplainer:

    def __init__(self, relationship_model):

        self.model = relationship_model.model
        self.model.eval()

        self.activations = []

        target_layer = self.model.hereditary_attention

        target_layer.register_forward_hook(
            self.save_activation
        )

    def save_activation(self, module, input, output):

        if output.requires_grad:
            output.retain_grad()

        self.activations.append(output)

    def generate_pair_cams(
        self,
        img1_tensor,
        img2_tensor,
        target_class
    ):

        self.model.zero_grad()
        self.activations = []

        output = self.model(
            img1_tensor,
            img2_tensor
        )

        score = output[0, target_class]

        score.backward()

        if len(self.activations) < 2:
            raise ValueError(
                "Could not capture activations for both images."
            )

        cam1 = self.compute_cam(
            self.activations[0]
        )

        cam2 = self.compute_cam(
            self.activations[1]
        )

        return cam1, cam2

    def compute_cam(self, activation_tensor):

        gradients = activation_tensor.grad[0]
        activations = activation_tensor.detach()[0]

        weights = torch.mean(
            gradients,
            dim=(1, 2)
        )

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        )

        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(
            cam,
            (224, 224)
        )

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam

    def save_overlay_from_tensor(
        self,
        face_tensor,
        cam,
        output_path
    ):

        img = face_tensor.squeeze(0).permute(1, 2, 0).numpy()
        img = np.uint8(img * 255)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            img,
            0.6,
            heatmap,
            0.4,
            0
        )

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        cv2.imwrite(
            output_path,
            overlay
        )
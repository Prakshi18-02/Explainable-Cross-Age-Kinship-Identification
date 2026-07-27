import os
import cv2
import torch
import numpy as np


class AttentionExplainer:

    def __init__(self, relationship_predictor):

        self.model = relationship_predictor.model
        self.model.eval()

        self.attention_map = None

        target_layer = self.model.hereditary_attention.spatial_attention

        target_layer.register_forward_hook(
            self.save_attention
        )

    def save_attention(self, module, inputs, output):

        self.attention_map = output.detach()

    def generate_single_attention_map(
        self,
        image_tensor
    ):

        with torch.no_grad():

            _ = self.model.extract(
                image_tensor
            )

        if self.attention_map is None:

            raise ValueError(
                "Attention map was not captured."
            )

        attn = self.attention_map[0, 0].cpu().numpy()

        attn = cv2.resize(
            attn,
            (224, 224)
        )

        attn = attn - attn.min()
        attn = attn / (attn.max() + 1e-8)

        return attn

    def save_attention_overlay(
        self,
        face_tensor,
        attention_map,
        output_path
    ):

        img = face_tensor.squeeze(0).permute(1, 2, 0).numpy()
        img = np.uint8(img * 255)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        heatmap = cv2.applyColorMap(
            np.uint8(255 * attention_map),
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
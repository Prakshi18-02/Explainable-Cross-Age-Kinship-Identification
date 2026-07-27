import os
import cv2
import numpy as np


class ComparisonGenerator:

    def __init__(self):

        pass

    def tensor_to_image(self, tensor):

        img = tensor.squeeze(0).permute(1, 2, 0).numpy()
        img = np.uint8(img * 255)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def resize_img(self, img, size=(224, 224)):

        return cv2.resize(img, size)

    def add_title(self, img, title):

        canvas = np.ones(
            (270, 224, 3),
            dtype=np.uint8
        ) * 255

        canvas[40:264, 0:224] = img

        cv2.putText(
            canvas,
            title,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        return canvas

    def create_comparison(
        self,
        face_tensor,
        eyes_tensor,
        nose_tensor,
        lips_tensor,
        attention_path,
        gradcam_path,
        output_path,
        relationship,
        kin_probability,
        relationship_confidence
    ):

        face = self.tensor_to_image(face_tensor)
        eyes = self.tensor_to_image(eyes_tensor)
        nose = self.tensor_to_image(nose_tensor)
        lips = self.tensor_to_image(lips_tensor)

        attention = cv2.imread(attention_path)
        gradcam = cv2.imread(gradcam_path)

        if attention is None:
            raise ValueError("Attention image not found.")

        if gradcam is None:
            raise ValueError("Grad-CAM image not found.")

        face = self.resize_img(face)
        eyes = self.resize_img(eyes)
        nose = self.resize_img(nose)
        lips = self.resize_img(lips)
        attention = self.resize_img(attention)
        gradcam = self.resize_img(gradcam)

        panels = [
            self.add_title(face, "Face"),
            self.add_title(eyes, "Eyes"),
            self.add_title(nose, "Nose"),
            self.add_title(lips, "Lips"),
            self.add_title(attention, "Attention"),
            self.add_title(gradcam, "Grad-CAM")
        ]

        top_row = np.hstack(panels)

        info_height = 130
        info = np.ones(
            (info_height, top_row.shape[1], 3),
            dtype=np.uint8
        ) * 255

        text_lines = [
            "Explainable Cross-Age Kinship Identification",
            f"Kin Probability: {kin_probability * 100:.2f}%",
            f"Relationship: {relationship}",
            f"Relationship Confidence: {relationship_confidence * 100:.2f}%"
        ]

        y = 30

        for line in text_lines:

            cv2.putText(
                info,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )

            y += 30

        final_img = np.vstack(
            [
                top_row,
                info
            ]
        )

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        cv2.imwrite(
            output_path,
            final_img
        )

        return output_path
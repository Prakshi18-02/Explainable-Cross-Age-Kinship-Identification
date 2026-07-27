import os
import cv2
import numpy as np


class PairReportGenerator:

    def tensor_to_image(self, tensor):

        img = tensor.squeeze(0).permute(1, 2, 0).numpy()
        img = np.uint8(img * 255)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return cv2.resize(img, (224, 224))

    def add_title(self, img, title):

        canvas = np.ones((260, 224, 3), dtype=np.uint8) * 255
        canvas[35:259, 0:224] = img

        cv2.putText(
            canvas,
            title,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 0),
            2
        )

        return canvas

    def read_image(self, path):

        img = cv2.imread(path)

        if img is None:
            raise ValueError(f"Could not read image: {path}")

        return cv2.resize(img, (224, 224))

    def create_report(
        self,
        data1,
        data2,
        p1_attention_path,
        p2_attention_path,
        p1_gradcam_path,
        p2_gradcam_path,
        output_path,
        kin_probability,
        relationship,
        relationship_confidence
    ):

        panels = [
            self.add_title(self.tensor_to_image(data1["face"]), "P1 Face"),
            self.add_title(self.tensor_to_image(data2["face"]), "P2 Face"),

            self.add_title(self.tensor_to_image(data1["eyes"]), "P1 Eyes"),
            self.add_title(self.tensor_to_image(data2["eyes"]), "P2 Eyes"),

            self.add_title(self.tensor_to_image(data1["nose"]), "P1 Nose"),
            self.add_title(self.tensor_to_image(data2["nose"]), "P2 Nose"),

            self.add_title(self.tensor_to_image(data1["lips"]), "P1 Lips"),
            self.add_title(self.tensor_to_image(data2["lips"]), "P2 Lips"),

            self.add_title(self.read_image(p1_attention_path), "P1 Attention"),
            self.add_title(self.read_image(p2_attention_path), "P2 Attention"),

            self.add_title(self.read_image(p1_gradcam_path), "P1 Grad-CAM"),
            self.add_title(self.read_image(p2_gradcam_path), "P2 Grad-CAM"),
        ]

        row1 = np.hstack(panels[0:2])
        row2 = np.hstack(panels[2:6])
        row3 = np.hstack(panels[6:10])
        row4 = np.hstack(panels[10:12])

        width = max(
            row1.shape[1],
            row2.shape[1],
            row3.shape[1],
            row4.shape[1]
        )

        def pad(row):
            extra = width - row.shape[1]

            if extra <= 0:
                return row

            pad_img = np.ones(
                (row.shape[0], extra, 3),
                dtype=np.uint8
            ) * 255

            return np.hstack([row, pad_img])

        row1 = pad(row1)
        row2 = pad(row2)
        row3 = pad(row3)
        row4 = pad(row4)

        info = np.ones((170, width, 3), dtype=np.uint8) * 255

        lines = [
            "Explainable Cross-Age Kinship Identification",
            f"Kin Probability: {kin_probability * 100:.2f}%",
            f"Predicted Relationship: {relationship}",
            f"Relationship Confidence: {relationship_confidence * 100:.2f}%",
            "Explanation: region crops, attention maps and Grad-CAM are shown for both faces."
        ]

        y = 35

        for line in lines:

            cv2.putText(
                info,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2
            )

            y += 30

        final_img = np.vstack([row1, row2, row3, row4, info])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, final_img)

        return output_path
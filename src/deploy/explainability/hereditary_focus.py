import os
import cv2


class HereditaryFocusMap:

    def create_focus_map(self, original_image_path, landmarks, bbox, output_path):

        img = cv2.imread(original_image_path)

        if img is None:
            raise ValueError(f"Could not read image: {original_image_path}")

        h, w, _ = img.shape
        overlay = img.copy()

        # Use same actual eye landmarks as region extractor
        eye_points = landmarks[33:53]

        ex1 = int(eye_points[:, 0].min())
        ey1 = int(eye_points[:, 1].min())
        ex2 = int(eye_points[:, 0].max())
        ey2 = int(eye_points[:, 1].max())

        eye_center_y = (ey1 + ey2) // 2
        eye_height = max(18, int((ey2 - ey1) * 1.8))

        # Expand horizontally using face bbox so both eyes are covered
        bx1, by1, bx2, by2 = bbox.astype(int)
        face_width = bx2 - bx1

        eye_box = (
            max(0, ex1 - int(0.65 * face_width)),
            max(0, eye_center_y - eye_height // 2),
            min(w, ex2 + int(0.65 * face_width)),
            min(h, eye_center_y + eye_height // 2)
        )

        nose_box = self.get_box(landmarks[72:87], w, h, 25, 25)
        lips_box = self.get_box(landmarks[52:72], w, h, 25, 18)

        regions = {
            "Eyes": eye_box,
            "Nose": nose_box,
            "Lips": lips_box
        }

        for name, (x1, y1, x2, y2) in regions.items():

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.putText(
                img,
                name,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        focus = cv2.addWeighted(img, 0.82, overlay, 0.18, 0)
        focus = cv2.resize(focus, (224, 224))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, focus)

        return output_path

    def get_box(self, points, image_width, image_height, pad_x, pad_y):

        x1 = int(points[:, 0].min()) - pad_x
        y1 = int(points[:, 1].min()) - pad_y
        x2 = int(points[:, 0].max()) + pad_x
        y2 = int(points[:, 1].max()) + pad_y

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        return x1, y1, x2, y2
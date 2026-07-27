import cv2
from PIL import Image
from torchvision import transforms
from insightface.app import FaceAnalysis


class FaceRegionExtractor:

    def __init__(self):

        self.app = FaceAnalysis(
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(ctx_id=-1)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def crop_region(self, img, points, padding=10):

        h, w, _ = img.shape

        x1 = int(points[:, 0].min()) - padding
        y1 = int(points[:, 1].min()) - padding
        x2 = int(points[:, 0].max()) + padding
        y2 = int(points[:, 1].max()) + padding

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(crop)
        crop = self.transform(crop)

        return crop.unsqueeze(0)

    def crop_face(self, img, bbox, padding=20):

        h, w, _ = img.shape

        x1, y1, x2, y2 = bbox.astype(int)

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        face_crop = img[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_crop = Image.fromarray(face_crop)
        face_crop = self.transform(face_crop)

        return face_crop.unsqueeze(0)

    def process_image(self, image_path):

        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        faces = self.app.get(img)

        if len(faces) == 0:
            raise ValueError(
                f"No face detected in image: {image_path}"
            )

        face = faces[0]
        landmarks = face.landmark_2d_106

        face_tensor = self.crop_face(
            img,
            face.bbox
        )

        eyes = self.crop_region(
            img,
            landmarks[33:53]
        )

        nose = self.crop_region(
            img,
            landmarks[72:87]
        )

        lips = self.crop_region(
            img,
            landmarks[52:72]
        )

        if face_tensor is None or eyes is None or nose is None or lips is None:
            raise ValueError(
                "Could not extract face or facial regions."
            )

        return {
        "face": face_tensor,
        "eyes": eyes,
        "nose": nose,
        "lips": lips,
        "landmarks": landmarks,
        "age": face.age,
        "gender": face.gender
        }
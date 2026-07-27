from src.deploy.face_region_extractor import FaceRegionExtractor
from src.deploy.feature_extractor import ResNetFeatureExtractor
from src.deploy.binary_verifier import BinaryKinVerifier
from src.deploy.relationship_predictor import RelationshipPredictor


NOT_KIN_THRESHOLD = 0.31
KIN_PASS_THRESHOLD = 0.70


def main():

    print("=" * 60)
    print("DEPLOYED KINSHIP PREDICTION PIPELINE")
    print("=" * 60)

    img1_path = input("Enter first image path: ")
    img2_path = input("Enter second image path: ")

    print()
    print("Loading modules...")

    region_extractor = FaceRegionExtractor()
    feature_extractor = ResNetFeatureExtractor()
    binary_verifier = BinaryKinVerifier()
    relationship_predictor = RelationshipPredictor()

    print("Processing images...")

    data1 = region_extractor.process_image(img1_path)
    data2 = region_extractor.process_image(img2_path)

    face1 = feature_extractor.extract(data1["face"])
    face2 = feature_extractor.extract(data2["face"])

    eyes1 = feature_extractor.extract(data1["eyes"])
    nose1 = feature_extractor.extract(data1["nose"])
    lips1 = feature_extractor.extract(data1["lips"])

    eyes2 = feature_extractor.extract(data2["eyes"])
    nose2 = feature_extractor.extract(data2["nose"])
    lips2 = feature_extractor.extract(data2["lips"])

    kin_probability, fusion_weights = binary_verifier.predict(
        face1,
        face2,
        eyes1,
        nose1,
        lips1,
        eyes2,
        nose2,
        lips2
    )

    print()
    print("=" * 60)
    print("STAGE 1: KIN / NOT-KIN VERIFICATION")
    print("=" * 60)
    print(f"Kin Probability: {kin_probability * 100:.2f}%")

    if kin_probability < NOT_KIN_THRESHOLD:
        print("Final Decision: Not Kin")
        print("Relationship Prediction: Not Applicable")
        return

    if kin_probability < KIN_PASS_THRESHOLD:
        print("Final Decision: Uncertain")
        print("Relationship Prediction: Not performed")
        print("Reason: Kin probability is not high enough.")
        return

    print("Final Decision: Kin")
    print("Proceeding to relationship classification...")

    result = relationship_predictor.predict(
        data1["face"],
        data2["face"]
    )

    print()
    print("=" * 60)
    print("STAGE 2: RELATIONSHIP CLASSIFICATION")
    print("=" * 60)
    print("Predicted Relationship:", result["relationship"])
    print(f"Relationship Confidence: {result['confidence'] * 100:.2f}%")

    print()
    print("=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print("Kinship Decision: Kin")
    print("Relationship:", result["relationship"])
    print(f"Confidence: {result['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()
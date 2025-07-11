from deepface import DeepFace
import os
import cv2

# Folder paths
KNOWN_DIR = "face_db"
CAPTURED_DIR = "captures"

# Matching threshold — lower = stricter, 0.6–0.95 is typical
THRESHOLD = 0.75
MODEL_NAME = "ArcFace"

for captured_img in os.listdir(CAPTURED_DIR):
    captured_path = os.path.join(CAPTURED_DIR, captured_img)

    if cv2.imread(captured_path) is None:
        print(f"[SKIP] Can't read {captured_img}")
        continue

    match_found = False

    for known_img in os.listdir(KNOWN_DIR):
        known_path = os.path.join(KNOWN_DIR, known_img)

        if cv2.imread(known_path) is None:
            print(f"[SKIP] Can't read {known_img}")
            continue

        try:
            result = DeepFace.verify(
                img1_path=captured_path,
                img2_path=known_path,
                model_name=MODEL_NAME,
                enforce_detection=False  # Both are cropped, so skip detection
            )

            distance = result["distance"]
            print(f"[COMPARE] {captured_img} vs {known_img} → Distance: {distance:.4f}")

            if distance < THRESHOLD:
                print(f"✅ MATCH FOUND: {captured_img} matches with {known_img} (Distance: {distance:.4f})")
                match_found = True
                break

        except Exception as e:
            print(f"[ERROR] Failed to compare {captured_img} and {known_img}: {e}")

    if not match_found:
        print(f"❌ NO MATCH: {captured_img} does not match any known face.\n")


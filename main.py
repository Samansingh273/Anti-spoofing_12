


# import cv2
# import tensorflow as tf
# import numpy as np
# import imutils
# import pickle
# import os

# # Load encoders and model
# print("[INFO] loading encodings...")
# with open("encoded_faces.pickle", "rb") as file:
#     encoded_data = pickle.load(file)

# print("[INFO] loading face detector...")
# proto_path = os.path.sep.join(["face_detector", "deploy.prototxt"])
# model_path = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
# detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# print("[INFO] loading liveness model...")
# liveness_model = tf.keras.models.load_model("liveness_model.h5")


# print("[INFO] loading label encoder...")
# le = pickle.loads(open("label_encoder.pickle", "rb").read())

# # Start webcam
# cap = cv2.VideoCapture(0)

# print("[INFO] starting video stream...")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = imutils.resize(frame, width=800)
#     (h, w) = frame.shape[:2]

#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#                                  (300, 300), (104.0, 177.0, 123.0))
#     detector_net.setInput(blob)
#     detections = detector_net.forward()

#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             startX = max(0, startX - 20)
#             startY = max(0, startY - 20)
#             endX = min(w, endX + 20)
#             endY = min(h, endY + 20)

#             face = frame[startY:endY, startX:endX]
#             try:
#                 face = cv2.resize(face, (32, 32))
#             except:
#                 continue

#             face = face.astype("float") / 255.0
#             face = tf.keras.preprocessing.image.img_to_array(face)
#             face = np.expand_dims(face, axis=0)

#             preds = liveness_model.predict(face)[0]
#             j = np.argmax(preds)
#             label_name = le.classes_[j]
#             label = f"{label_name}: {preds[j]:.4f}"

#             color = (0, 0, 255) if label_name == "fake" else (0, 255, 0)
#             cv2.putText(frame, label, (startX, startY - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#             cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

#     cv2.imshow("Liveness Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()



# import os
# import cv2
# import time
# import numpy as np
# import pickle
# import tensorflow as tf
# import mediapipe as mp
# from ultralytics import YOLO
# from datetime import datetime

# # Ensure 'captures' folder exists
# os.makedirs("captures", exist_ok=True)

# # Auto-download YOLOv8n if missing
# if not os.path.exists("yolov8n.pt"):
#     from ultralytics.utils.downloads import attempt_download
#     print("[INFO] Downloading YOLOv8n model...")
#     attempt_download("yolov8n.pt")
# yolo = YOLO("yolov8n.pt")

# # Load MediaPipe Pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

# # Load models
# liveness_model = tf.keras.models.load_model("liveness_model.h5")
# with open("label_encoder.pickle", "rb") as f:
#     le = pickle.load(f)

# # Load OpenCV DNN face detector
# net = cv2.dnn.readNetFromCaffe(
#     os.path.join("face_detector", "deploy.prototxt"),
#     os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
# )

# # Video stream
# cap = cv2.VideoCapture(0)
# burst_count = 0
# MAX_BURSTS = 2
# capture_done = False

# print("[INFO] Starting live feed...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (800, 600))
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     h, w = frame.shape[:2]

#     # Detect motion
#     results = pose.process(rgb)
#     is_moving = results.pose_landmarks is not None

#     # Face detection
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#                                  (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()

#     for i in range(detections.shape[2]):
#         conf = detections[0, 0, i, 2]
#         if conf < 0.6:
#             continue

#         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#         (startX, startY, endX, endY) = box.astype("int")
#         startX, startY = max(0, startX - 20), max(0, startY - 20)
#         endX, endY = min(w, endX + 20), min(h, endY + 20)

#         face = frame[startY:endY, startX:endX]
#         try:
#             input_face = cv2.resize(face, (32, 32)).astype("float") / 255.0
#         except:
#             continue

#         input_face = tf.keras.preprocessing.image.img_to_array(input_face)
#         input_face = np.expand_dims(input_face, axis=0)

#         preds = liveness_model.predict(input_face, verbose=0)[0]
#         label_index = np.argmax(preds)
#         label_name = le.classes_[label_index]
#         label_conf = preds[label_index]

#         label_text = f"{label_name.upper()} ({label_conf:.2f})"
#         color = (0, 255, 0) if label_name == "real" else (0, 0, 255)

#         # Draw on screen only
#         cv2.putText(frame, label_name.upper(), (startX, startY - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#         cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

#         # Handle capture only if not already done
#         if not capture_done and label_name == "real" and is_moving:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"captures/real_{timestamp}_{burst_count+1}.jpg"
#             cv2.imwrite(filename, face)
#             burst_count += 1
#             if burst_count >= MAX_BURSTS:
#                 capture_done = True

#     cv2.imshow("Live Feed", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()



# eye blink main.py 

import os
import cv2
import time
import numpy as np
import pickle
import tensorflow as tf
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime

# Ensure 'captures' folder exists
os.makedirs("captures", exist_ok=True)

# Auto-download YOLOv8n if missing
if not os.path.exists("yolov8n.pt"):
    from ultralytics.utils.downloads import attempt_download
    print("[INFO] Downloading YOLOv8n model...")
    attempt_download("yolov8n.pt")
yolo = YOLO("yolov8n.pt")

# Load MediaPipe Pose and FaceMesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load models
liveness_model = tf.keras.models.load_model("liveness_model.h5")
with open("label_encoder.pickle", "rb") as f:
    le = pickle.load(f)

# Load OpenCV DNN face detector
net = cv2.dnn.readNetFromCaffe(
    os.path.join("face_detector", "deploy.prototxt"),
    os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
)

# EAR computation functions
def euclidean_dist(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def compute_ear(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Eye indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

# Blink detection threshold
BLINK_THRESHOLD = 0.18
blink_started = False
capture_ready = False

# Video stream
cap = cv2.VideoCapture(0)
burst_count = 0
MAX_BURSTS = 2
capture_done = False

print("[INFO] Starting live feed...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # Detect motion
    results = pose.process(rgb)
    is_moving = results.pose_landmarks is not None

    # Detect blink
    capture_ready = False
    face_results = face_mesh.process(rgb)
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        ih, iw = frame.shape[:2]

        try:
            left_eye = [(int(landmarks[i].x * iw), int(landmarks[i].y * ih)) for i in LEFT_EYE_IDX]
            right_eye = [(int(landmarks[i].x * iw), int(landmarks[i].y * ih)) for i in RIGHT_EYE_IDX]

            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < BLINK_THRESHOLD:
                blink_started = True  # Eyes are closed
            elif blink_started and avg_ear >= BLINK_THRESHOLD:
                capture_ready = True  # Blink just finished, eyes reopened
                blink_started = False
                print("[INFO] Blink complete – eyes reopened.")
        except:
            pass

    # Face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < 0.6:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        startX, startY = max(0, startX - 20), max(0, startY - 20)
        endX, endY = min(w, endX + 20), min(h, endY + 20)

        face = frame[startY:endY, startX:endX]
        try:
            input_face = cv2.resize(face, (32, 32)).astype("float") / 255.0
        except:
            continue

        input_face = tf.keras.preprocessing.image.img_to_array(input_face)
        input_face = np.expand_dims(input_face, axis=0)

        preds = liveness_model.predict(input_face, verbose=0)[0]
        label_index = np.argmax(preds)
        label_name = le.classes_[label_index]
        label_conf = preds[label_index]

        label_text = f"{label_name.upper()} ({label_conf:.2f})"
        color = (0, 255, 0) if label_name == "real" else (0, 0, 255)

        # Draw on screen only
        cv2.putText(frame, label_name.upper(), (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Capture after blink ends
        if not capture_done and label_name == "real" and is_moving and capture_ready:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captures/real_{timestamp}_{burst_count+1}.jpg"
            cv2.imwrite(filename, face)
            burst_count += 1
            print(f"[INFO] Image captured after blink: {filename}")
            if burst_count >= MAX_BURSTS:
                capture_done = True

    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




# finall




# import cv2
# import tensorflow as tf
# import numpy as np
# import pickle
# import os
# import mediapipe as mp
# from scipy.spatial import distance as dist
# from datetime import datetime

# # EAR calculation
# def eye_aspect_ratio(eye):
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)

# # EAR thresholding
# EYE_AR_THRESH = 0.25
# EYE_AR_CONSEC_FRAMES = 2
# blink_counter = 0
# blink_detected = False
# blink_released = False

# # Motion detection
# prev_gray = None
# captured_images = 0
# MAX_IMAGES = 2

# # Load models
# print("[INFO] loading encodings...")
# with open("encoded_faces.pickle", "rb") as file:
#     encoded_data = pickle.load(file)

# print("[INFO] loading face detector...")
# proto_path = os.path.sep.join(["face_detector", "deploy.prototxt"])
# model_path = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
# detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# print("[INFO] loading liveness model...")
# liveness_model = tf.keras.models.load_model("liveness_model.h5")

# print("[INFO] loading label encoder...")
# le = pickle.loads(open("label_encoder.pickle", "rb").read())

# # Initialize MediaPipe FaceMesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
#                                    max_num_faces=1,
#                                    refine_landmarks=True,
#                                    min_detection_confidence=0.5,
#                                    min_tracking_confidence=0.5)

# # Eye landmark indices from MediaPipe
# LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# # Start webcam
# cap = cv2.VideoCapture(0)
# print("[INFO] starting video stream...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (800, 600))
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     h, w = frame.shape[:2]

#     # Detect motion
#     motion_detected = False
#     if prev_gray is not None:
#         frame_delta = cv2.absdiff(prev_gray, gray)
#         thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
#         motion_score = np.sum(thresh)
#         motion_detected = motion_score > 50000
#     prev_gray = gray

#     # Face detection via OpenCV DNN
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#                                  (300, 300), (104.0, 177.0, 123.0))
#     detector_net.setInput(blob)
#     detections = detector_net.forward()

#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#             startX = max(0, startX - 20)
#             startY = max(0, startY - 20)
#             endX = min(w, endX + 20)
#             endY = min(h, endY + 20)

#             face = frame[startY:endY, startX:endX]
#             try:
#                 face_resized = cv2.resize(face, (32, 32))
#             except:
#                 continue

#             face_input = face_resized.astype("float") / 255.0
#             face_input = tf.keras.preprocessing.image.img_to_array(face_input)
#             face_input = np.expand_dims(face_input, axis=0)

#             preds = liveness_model.predict(face_input, verbose=0)[0]
#             j = np.argmax(preds)
#             label_name = le.classes_[j]
#             label = f"{label_name}: {preds[j]:.4f}"

#             color = (0, 0, 255) if label_name == "fake" else (0, 255, 0)
#             cv2.putText(frame, label, (startX, startY - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#             cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

#             # Blink detection with MediaPipe
#             if label_name == "real":
#                 results = face_mesh.process(rgb_frame)
#                 if results.multi_face_landmarks:
#                     landmarks = results.multi_face_landmarks[0].landmark

#                     left_eye = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in LEFT_EYE_IDX]
#                     right_eye = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in RIGHT_EYE_IDX]

#                     leftEAR = eye_aspect_ratio(left_eye)
#                     rightEAR = eye_aspect_ratio(right_eye)
#                     ear = (leftEAR + rightEAR) / 2.0

#                     if ear < EYE_AR_THRESH:
#                         blink_counter += 1
#                         blink_released = False
#                     else:
#                         if blink_counter >= EYE_AR_CONSEC_FRAMES:
#                             blink_detected = True
#                         blink_counter = 0

#                         if blink_detected and not blink_released:
#                             if motion_detected and captured_images < MAX_IMAGES:
#                                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                                 filename = f"captures/real_{timestamp}.jpg"
#                                 cv2.imwrite(filename, frame)
#                                 print(f"[INFO] Captured: {filename}")
#                                 captured_images += 1
#                             blink_detected = False
#                             blink_released = True

#                     for (x, y) in left_eye + right_eye:
#                         cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)

#     cv2.imshow("Liveness Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()








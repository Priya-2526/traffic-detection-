# -------- IMPORT LIBRARIES --------
import cv2
import numpy as np
import os
import urllib.request


# -------- FILE PATHS --------
# get current folder path
base_dir = os.path.dirname(__file__)

# model files
weights = os.path.join(base_dir, "yolov3-tiny.weights")
cfg = os.path.join(base_dir, "yolov3-tiny.cfg")
names_path = os.path.join(base_dir, "coco.names")


# -------- DOWNLOAD WEIGHTS IF NOT PRESENT --------
# this is useful for render / new pc

if not os.path.exists(weights):
    print("Downloading weights...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pjreddie/darknet/master/yolov3.weights",
        weights
    )


# -------- LOAD YOLO MODEL --------
net = cv2.dnn.readNet(weights, cfg)

# load class names
with open(names_path) as f:
    classes = [line.strip() for line in f.readlines()]


# get output layers
layer_names = net.getLayerNames()
unconnected = net.getUnconnectedOutLayers()

output_layers = []

# fix for different opencv versions
for i in unconnected:
    try:
        output_layers.append(layer_names[i[0] - 1])
    except:
        output_layers.append(layer_names[i - 1])


# -------- MAIN FUNCTION --------
def detect_traffic(image_path):

    # read image
    img = cv2.imread(image_path)

    if img is None:
        return None

    h, w, _ = img.shape


    # -------- WEATHER DETECTION --------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()

    if contrast < 35:
        weather = "Foggy"
    else:
        weather = "Clear"


    # -------- YOLO INPUT --------
    blob = cv2.dnn.blobFromImage(img, 1/255, (608, 608), swapRB=True)

    net.setInput(blob)

    outputs = net.forward(output_layers)


    # lists for NMS
    boxes = []
    confidences = []
    class_ids = []


    # -------- DETECTION LOOP --------
    for output in outputs:
        for detection in output:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # avoid index error
            if class_id >= len(classes):
                continue

            label = classes[class_id]

            # detect only vehicles
            if confidence > 0.3 and label in ["car", "bus", "truck", "motorbike", "motorcycle"]:

                cx = int(detection[0] * w)
                cy = int(detection[1] * h)

                bw = int(detection[2] * w)
                bh = int(detection[3] * h)

                x = int(cx - bw / 2)
                y = int(cy - bh / 2)

                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    # -------- NMS (REMOVE DUPLICATE BOXES) --------
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    vehicle_count = 0

    for i in indexes:

        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i

        x, y, bw, bh = boxes[i]

        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        vehicle_count += 1


    # -------- TRAFFIC LEVEL --------
    if vehicle_count < 20:
        traffic = "Low"
    elif vehicle_count < 50:
        traffic = "Medium"
    else:
        traffic = "High"


    # -------- RISK CALCULATION --------
    risk_score = 0

    if traffic == "Medium":
        risk_score += 1
    elif traffic == "High":
        risk_score += 2

    if weather == "Foggy":
        risk_score += 2


    if risk_score <= 1:
        risk = "LOW RISK"
    elif risk_score <= 3:
        risk = "MEDIUM RISK"
    else:
        risk = "HIGH RISK"


    # -------- WRITE TEXT --------
    text = f"Vehicles: {vehicle_count}  Traffic: {traffic}  Risk: {risk}"

    y = img.shape[0] - 20

    cv2.putText(img, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)

    cv2.putText(img, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    # -------- SAVE RESULT --------
    output_path = image_path.replace("uploads", "uploads/result")

    cv2.imwrite(output_path, img)


    # -------- RETURN RESULT --------
    return {
        "vehicles": vehicle_count,
        "traffic": traffic,
        "weather": weather,
        "risk": risk,
        "output_image": output_path
    }
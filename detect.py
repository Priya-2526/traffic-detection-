# -------- IMPORT LIBRARIES --------
import cv2
import numpy as np
import os
import urllib.request


# -------- FILE PATHS --------
base_dir = os.path.dirname(__file__)

weights = os.path.join(base_dir, "yolov3-tiny.weights")
cfg = os.path.join(base_dir, "yolov3-tiny.cfg")
names_path = os.path.join(base_dir, "coco.names")


# -------- DOWNLOAD WEIGHTS IF NOT PRESENT --------
if not os.path.exists(weights):
    print("Downloading weights...")
    urllib.request.urlretrieve(
        "https://pjreddie.com/media/files/yolov3-tiny.weights",
        weights
    )


# -------- LOAD YOLO --------
net = cv2.dnn.readNet(weights, cfg)

with open(names_path) as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
unconnected = net.getUnconnectedOutLayers()

output_layers = []

for i in unconnected:
    try:
        output_layers.append(layer_names[i[0] - 1])
    except:
        output_layers.append(layer_names[i - 1])


# -------- MAIN FUNCTION --------
def detect_traffic(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    h, w, _ = img.shape


    # WEATHER
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()

    weather = "Foggy" if contrast < 35 else "Clear"


    # YOLO INPUT
    blob = cv2.dnn.blobFromImage(img, 1/255, (608, 608), swapRB=True)

    net.setInput(blob)

    outputs = net.forward(output_layers)


    boxes = []
    confidences = []
    class_ids = []


    # DETECTION
    for output in outputs:
        for detection in output:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id >= len(classes):
                continue

            label = classes[class_id]

            if confidence > 0.3 and label in [
                "car", "bus", "truck", "motorbike", "motorcycle"
            ]:

                cx = int(detection[0] * w)
                cy = int(detection[1] * h)

                bw = int(detection[2] * w)
                bh = int(detection[3] * h)

                x = int(cx - bw / 2)
                y = int(cy - bh / 2)

                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    # NMS
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    vehicle_count = 0

    for i in indexes:

        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i

        x, y, bw, bh = boxes[i]

        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        vehicle_count += 1


    # TRAFFIC
    if vehicle_count < 20:
        traffic = "Low"
    elif vehicle_count < 50:
        traffic = "Medium"
    else:
        traffic = "High"


    # RISK
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


    # TEXT
    text = f"Vehicles: {vehicle_count}  Traffic: {traffic}  Risk: {risk}"

    y = img.shape[0] - 20

    cv2.putText(img, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)

    cv2.putText(img, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    # SAVE
  filename = os.path.basename(image_path)

output_path = os.path.join(
    os.path.dirname(image_path),
    "result",
    filename
)

    cv2.imwrite(output_path, img)


    # RETURN
    return {
        "vehicles": vehicle_count,
        "traffic": traffic,
        "weather": weather,
        "risk": risk,
        "output_image": output_path
    }

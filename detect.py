import cv2
import numpy as np
import os


base_dir = os.path.dirname(__file__)

weights = os.path.join(base_dir, "yolov3-tiny.weights")
cfg = os.path.join(base_dir, "yolov3-tiny.cfg")
names_path = os.path.join(base_dir, "coco.names")


print("WEIGHTS:", weights)
print("CFG:", cfg)


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


def detect_traffic(image_path):

    print("IMAGE:", image_path)

    img = cv2.imread(image_path)

    if img is None:
        print("Image not loaded")
        return None

    h, w, _ = img.shape


    blob = cv2.dnn.blobFromImage(
        img,
        1/255,
        (416, 416),
        swapRB=True
    )

    net.setInput(blob)

    outputs = net.forward(output_layers)


    boxes = []
    confidences = []
    class_ids = []


    for output in outputs:
        for detection in output:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:

                cx = int(detection[0] * w)
                cy = int(detection[1] * h)

                bw = int(detection[2] * w)
                bh = int(detection[3] * h)

                x = int(cx - bw / 2)
                y = int(cy - bh / 2)

                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        0.3,
        0.4
    )


    vehicle_count = len(indexes)


    for i in indexes:

        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i

        x, y, bw, bh = boxes[i]

        cv2.rectangle(
            img,
            (x, y),
            (x + bw, y + bh),
            (0, 255, 0),
            2
        )


    if vehicle_count < 20:
        traffic = "Low"
    elif vehicle_count < 50:
        traffic = "Medium"
    else:
        traffic = "High"


    weather = "Clear"
    risk = "MEDIUM RISK"


    filename = os.path.basename(image_path)

    output_path = os.path.join(
        os.path.dirname(image_path),
        "result",
        filename
    )

    print("SAVE:", output_path)

    cv2.imwrite(output_path, img)


    return {
        "vehicles": vehicle_count,
        "traffic": traffic,
        "weather": weather,
        "risk": risk,
        "output_image": output_path
    }

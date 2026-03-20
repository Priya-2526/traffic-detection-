import cv2
import os
import urllib.request

base_dir = os.path.dirname(__file__)

weights = os.path.join(base_dir, "yolov3-tiny.weights")
cfg = os.path.join(base_dir, "yolov3-tiny.cfg")
names = os.path.join(base_dir, "coco.names")


def download_weights():

    if not os.path.exists(weights):

        print("Downloading weights...")

        urllib.request.urlretrieve(
            "https://pjreddie.com/media/files/yolov3-tiny.weights",
            weights
        )


download_weights()

net = cv2.dnn.readNet(weights, cfg)

with open(names, "r") as f:
    classes = f.read().splitlines()


def detect_traffic(image_path):

    img = cv2.imread(image_path)

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(
        img, 1/255, (416, 416), swapRB=True, crop=False
    )

    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(output_layers)

    vehicle_count = 0

    for output in outputs:

        for detection in output:

            scores = detection[5:]
            class_id = scores.argmax()

            label = classes[class_id]

            if label in ["car", "bus", "truck", "motorbike"]:
                vehicle_count += 1

    return vehicle_count

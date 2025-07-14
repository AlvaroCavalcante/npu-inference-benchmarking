import openvino as ov
from collections import deque
import cv2
import numpy as np
import time
import torch

BACKEND = "openvino"
DEVICE_OPENVINO = "NPU"
DEVICE_TORCH = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45


if BACKEND == "openvino":
    core = ov.Core()
    compiled_model = core.compile_model("model/yolov5m.xml", DEVICE_OPENVINO)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
else:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    model.eval()


def non_max_suppression_opencv(predictions, conf_thres=0.25, iou_thres=0.45):
    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        pred = pred[pred[:, 4] >= conf_thres]
        if not pred.shape[0]:
            continue
        scores = pred[:, 4] * pred[:, 5:].max(1)[0]
        classes = pred[:, 5:].argmax(1)
        for i in range(len(pred)):
            x, y, w, h = pred[i][:4]
            x1 = x - w / 2
            y1 = y - h / 2
            # formato: x, y, w, h
            boxes.append([int(x1), int(y1), int(w), int(h)])
            confidences.append(float(scores[i]))
            class_ids.append(int(classes[i]))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)

    final = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            x2 = x + w
            y2 = y + h
            final.append([x, y, x2, y2, confidences[i], class_ids[i]])

    return final


def preprocess_frame(frame):
    if BACKEND == "openvino":
        img_input = cv2.resize(frame, (640, 640))
        img_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
        img_input = img_input.transpose((2, 0, 1))  # HWC → CHW
        img_input = np.expand_dims(img_input, axis=0).astype(
            np.float32) / 255.0  # Normalização
        return img_input

    return frame


cap = cv2.VideoCapture("c:/Users/leand/Downloads/walking_2.mp4")
fps_history = deque(maxlen=45)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    original_height, original_width = frame.shape[:2]
    input_data = preprocess_frame(frame.copy())
    start_time = time.perf_counter()

    if BACKEND == "openvino":
        result = compiled_model([input_data])[output_layer]
    else:
        preds = model(input_data)
        preds = preds.pandas().xyxy[0]

    end_time = time.perf_counter()

    inference_time = end_time - start_time
    fps = 1.0 / inference_time if inference_time > 0 else 0
    fps_history.append(fps)
    avg_fps = sum(fps_history) / len(fps_history)

    if BACKEND == "openvino":
        scale_x = original_width / 640
        scale_y = original_height / 640

        pred = result.squeeze(0)
        boxes = non_max_suppression_opencv(
            [pred], CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

        for box in boxes:
            x1, y1, x2, y2, score, cls_id = box

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            label = f"{int(cls_id)} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    else:
        for _, row in preds.iterrows():
            x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS Médio: {avg_fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Detecção de Objetos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Media de FPS: {avg_fps:.2f}")
cap.release()
cv2.destroyAllWindows()

import openvino as ov
from collections import deque
import cv2
import numpy as np
import time
import queue
import threading
import torch

BACKEND = "openvino"
DEVICE_OPENVINO = "NPU"
DEVICE_TORCH = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45

TARGET_FPS = 25
FRAME_INTERVAL = 1.0 / TARGET_FPS


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


class FrameProducer(threading.Thread):
    def __init__(self, video_path, frame_queue):
        super().__init__()
        self.cap = cv2.VideoCapture(video_path)
        self.frame_queue = frame_queue
        self.running = True

    def run(self):
        while self.running and self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

            elapsed = time.time() - start_time
            sleep_time = FRAME_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.cap.release()
        self.frame_queue.put(None)

    def stop(self):
        self.running = False


class FrameConsumer:
    def __init__(self, frame_queue):
        self.frame_queue = frame_queue
        self.fps_history = deque(maxlen=40)
        self.out = None

        if BACKEND == "openvino":
            self.core = ov.Core()
            self.compiled_model = self.core.compile_model(
                "c:/Users/leand/OneDrive/Área de Trabalho/camera-recall/yolov5l.xml", DEVICE_OPENVINO)
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
        else:
            self.model = torch.hub.load(
                'ultralytics/yolov5', 'yolov5l', pretrained=True)
            self.model.eval()

    def preprocess_frame(self, frame):
        if BACKEND == "openvino":
            img_input = cv2.resize(frame, (640, 640))
            img_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
            img_input = img_input.transpose((2, 0, 1))
            img_input = np.expand_dims(
                img_input, axis=0).astype(np.float32) / 255.0
            return img_input
        return frame

    def consume(self):
        start_exec_time = time.time()
        while True:
            try:
                frame = self.frame_queue.get()
            except queue.Empty:
                break

            if frame is None:
                break

            if self.out is None:
                original_height, original_width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(
                    'output_video.mp4', fourcc, TARGET_FPS, (original_width, original_height))

            input_data = self.preprocess_frame(frame.copy())
            inference_start = time.perf_counter()

            if BACKEND == "openvino":
                result = self.compiled_model([input_data])[self.output_layer]
            else:
                preds = self.model(input_data)
                preds = preds.pandas().xyxy[0]

            inference_end = time.perf_counter()
            inference_time = inference_end - inference_start
            fps = 1.0 / inference_time if inference_time > 0 else 0
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)

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
                    x1, y1, x2, y2 = map(
                        int, [row.xmin, row.ymin, row.xmax, row.ymax])
                    label = f"{row['name']} {row['confidence']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            cv2.putText(frame, f"FPS Medio: {avg_fps:.2f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (24, 56, 217), 3)
            cv2.putText(frame, f"Dispositivo: {DEVICE_OPENVINO}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (24, 56, 217), 3)

            self.out.write(frame)
            cv2.imshow('Detecção de Objetos', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(
            f"Tempo total de execução: {time.time() - start_exec_time:.2f} segundos")
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=1)
    producer = FrameProducer(
        "c:/Users/leand/Downloads/walking_video.mp4", frame_queue)
    consumer = FrameConsumer(frame_queue)

    producer.start()
    consumer.consume()
    producer.stop()
    producer.join()

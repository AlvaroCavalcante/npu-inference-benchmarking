import openvino as ov
import cv2
import numpy as np


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


def preprocess_image(image):
    img_input = cv2.resize(image, (640, 640))
    img_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
    img_input = img_input.transpose((2, 0, 1))
    img_input = np.expand_dims(
        img_input, axis=0).astype(np.float32) / 255.0
    return img_input


def run_npu_inference():
    core = ov.Core()
    compiled_model = core.compile_model(
        "c:/Users/leand/OneDrive/Área de Trabalho/camera-recall/yolov5l.xml", 'NPU')
    output_layer = compiled_model.output(0)

    image = cv2.imread(
        "c:/Users/leand/Downloads/will-colavito-LrxFcY6Ck5I-unsplash.jpg")
    original_height, original_width = image.shape[:2]

    input_data = preprocess_image(image.copy())

    result = compiled_model([input_data])[output_layer]

    pred = result.squeeze(0)
    boxes = non_max_suppression_opencv(
        [pred], 0.4, 0.45)

    scale_x = original_width / 640
    scale_y = original_height / 640

    for box in boxes:
        x1, y1, x2, y2, score, cls_id = box

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        label = f"{int(cls_id)} {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 5)

    # while True:
    #     cv2.imshow('Detecção de Objetos', image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    cv2.imwrite('output_image.jpg', image)

if __name__ == "__main__":
    run_npu_inference()

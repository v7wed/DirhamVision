# scripts/detect.py
# Usage:
#   python scripts/detect.py                                    (default - webcam 0)
#   python scripts/detect.py --source 0                        (webcam explicit , change 0 to 1,2,3... for other cameras)
#   python scripts/detect.py --source assets/demo.mp4          (video)
#   python scripts/detect.py --source assets/test_image.jpg    (image)
#By default it uses the augmented model if you want to try baseline nano on dataset then add --weights weights/dirhamvision_nano_baseline.pt argument.
#Add the --save argument to save the output to outputs/ folder instead of displaying it.

import argparse
import os
import cv2
from torch import save
from ultralytics import YOLO

DENOMINATION_VALUES = {
    '25_fils': 0.25,
    '50_fils': 0.50,
    '1_aed_coin': 1.00,
    '10_aed_bil': 10.00
}
CLASS_COLORS = {
    '25_fils':    (0, 215, 255),
    '50_fils':    (0, 165, 255),
    '1_aed_coin': (0, 255,   0),
    '10_aed_bil': (255, 0,  255)
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def draw_detections(frame, results, model):
    total_aed = 0.0
    detected = {}

    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        score    = float(box.conf[0])
        total_aed += DENOMINATION_VALUES.get(cls_name, 0)
        detected[cls_name] = detected.get(cls_name, 0) + 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(frame, label, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Counter panel
    panel_h = 60 + len(detected) * 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (340, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"Total: {total_aed:.2f} AED",
                (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 120), 2)
    y = 70
    for cls_name, count in detected.items():
        val   = DENOMINATION_VALUES.get(cls_name, 0)
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        cv2.putText(frame, f"{cls_name} x{count}  ({val*count:.2f} AED)",
                    (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 28

    return frame


def run_webcam_or_video(source, model, conf, save):
    cap = cv2.VideoCapture(
        int(source) if source.isdigit() else source,
        cv2.CAP_DSHOW if source.isdigit() else 0
    )
    if not cap.isOpened():
        print(f"Error: could not open source '{source}'")
        return

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save:
        os.makedirs('outputs', exist_ok=True)
        name     = 'webcam.mp4' if source.isdigit() else os.path.basename(source)
        out_path = os.path.join('outputs', name)
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (width, height))
        print(f"Saving to {out_path} — press Ctrl+C to stop")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=conf, verbose=False)[0]
                frame   = draw_detections(frame, results, model)
                writer.write(frame)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print("Running — press Q to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=conf, verbose=False)[0]
            frame   = draw_detections(frame, results, model)
            cv2.imshow('DirhamVision', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved → {out_path}")
    cv2.destroyAllWindows()


def run_image(source, model, conf, save):
    frame = cv2.imread(source)
    if frame is None:
        print(f"Error: could not read image '{source}'")
        print("Check the path is correct and the file exists.")
        return

    results = model(frame, conf=conf, verbose=False)[0]
    frame   = draw_detections(frame, results, model)

    if save:
        os.makedirs('outputs', exist_ok=True)
        out_path = os.path.join('outputs', os.path.basename(source))
        cv2.imwrite(out_path, frame)
        print(f"Saved → {out_path}")
    else:
        cv2.imshow('DirhamVision', frame)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run(source, weights, conf, save):
    model = YOLO(weights)
    ext   = os.path.splitext(source)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        run_image(source, model, conf, save)
    else:
        run_webcam_or_video(source, model, conf, save)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DirhamVision — UAE currency detector')
    parser.add_argument('--source',  default='0',
                        help='0=webcam | path/to/video.mp4 | path/to/image.jpg')
    parser.add_argument('--weights', default='weights/dirhamvision_nano.pt',
                        help='Path to model weights')
    parser.add_argument('--conf',    type=float, default=0.35,
                        help='Confidence threshold (default: 0.35)')
    parser.add_argument('--save', action='store_true',
                    help='Save output to outputs/ folder instead of displaying')
    args = parser.parse_args()
    run(args.source, args.weights, args.conf, args.save)
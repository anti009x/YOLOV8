# from curses import COLOR_BLACK
from turtle import color
import cv2
import argparse
import supervision as sv  
import torch
import numpy as np
from ultralytics import YOLO
from roboflow import Roboflow
ZONE_POLYGONE = np.array([
    [0,0],
    [1280,0 ],
    [1280,720 ],
    [0,720],
    [0,2000],
])

#inisiliasi Webcamp
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=10,
        type=int,
        help="Resolution of the webcam (width, height, FPS)",
    )
    args = parser.parse_args()
    return args

"--Main Program",
def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


    device = "0" if torch.cuda.is_available() else "cpu"
    # if device == 0:
    #     torch.cuda.set_device(0)
    #     print(f"Using device: {device}")
    #model = YOLO("yolov8l.pt")
    # if device == 0:
    # torch.cuda.set_device(0)
    #   model.to('cuda')


  # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO('yolov8n.pt')
    print("Model device:", model.device.type)# load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    #training #1

    # rf = Roboflow(api_key="QTnZMzKZ74qKU9HeppgT")
    # project = rf.workspace("suleman").project("hard-hat-sample-b0dxn")
    # version = project.version(1)
    # dataset = version.download("yolov8")
    
    # # Use the model
    # model.train(data="data.yaml", epochs=3)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format
    # #Best Paramteter
    # #results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
    # print(metrics)
    # print(results)
    # print(path)
    
    #traning 2


    box_annotator = sv.BoxAnnotator(

        # color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness = 2,
        # text_color: Color = Color.black(),
        text_scale = 0.5,
        text_thickness = 1,
        text_padding= 10,
    )
    
    zone = sv.PolygonZone(polygon=ZONE_POLYGONE, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_anotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.red(),
        thickness=2,
        text_scale=0.5,
        text_thickness=1,
        text_padding=10,

    )
    zone_text = f"Zone: Count: {{}}"
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame from webcam!")
            break

        print("Frame shape:", frame.shape)


        result = model(frame)[0] 


        detections = sv.Detections.from_yolov8(result)
        # object_count = len(zone.trigger(detections=detections))
        labels = [
            f"{model.model.names[class_id]} : {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
        ]


        object_count = len(zone.trigger(detections=detections))
        
        zone_text = f"Count: {object_count}"
 
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame = zone_anotator.annotate(scene=frame)
        
        


        cv2.imshow("yolov8", frame)


        if cv2.waitKey(30) == 27:
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

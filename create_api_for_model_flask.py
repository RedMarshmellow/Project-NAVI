from flask import Flask, request, jsonify, g as app_ctx
import io
import time
from position_calculator import calculate_position
import PIL
from ultralytics import YOLO


navi_app = Flask(__name__)

model = YOLO('yolov8n.yaml')

model = YOLO('yolov8n.pt')


@navi_app.before_request
def logging_before():
    app_ctx.start_time = time.perf_counter()


@navi_app.after_request
def logging_after(response):
    total_time = time.perf_counter() - app_ctx.start_time
    time_in_ms = int(total_time * 1000)
    print('Response time => ', time_in_ms, 'ms')
    return response


@navi_app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "Please send post request"

    elif request.method == "POST":
        frame = request.files.get('frame')  # get the frame sent by the API request
        im_bytes = frame.read()  # convert the file into byte stream
        image = PIL.Image.open(io.BytesIO(im_bytes))  # convert the byte stream into
        image_width, image_height = image.size

        prediction = model(image)
        obj_info = []
        for result in prediction:
            for idx, box in enumerate(result.boxes.xywh):
                obj = (result.boxes.cls[idx], box)
                obj_info.append(obj)
        objects_with_positions = calculate_position(obj_info, image_width,
                                                    image_height)
        data = {
            "objects_with_positions": objects_with_positions,
        }
        return jsonify(data)


navi_app.run(port=5000, host='0.0.0.0', debug=False)

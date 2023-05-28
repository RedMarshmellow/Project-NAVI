
# This dictionary will be utilized while finding the distance of objects from the user.
# We are using the width of an object instead of its height because while capturing the
# frame, the complete height of the object might not be captured. The width is more likely
# to be appearing in full length.
object_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                 'frisbee',
                 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard',
                 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                 'cell phone',
                 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                 'teddy bear',
                 'hair dryer', 'toothbrush']

objects_actual_width = {
    'person': 0.5,
    'bicycle': 1.0,
    'car': 1.8,
    'motorcycle': 0.9,
    'airplane': 30.0,
    'bus': 2.5,
    'train': 3.0,
    'truck': 2.5,
    'boat': 2.0,
    'traffic light': 0.3,
    'fire hydrant': 0.3,
    'stop sign': 0.9,
    'parking meter': 0.2,
    'bench': 1.5,
    'bird': 0.3,
    'cat': 0.4,
    'dog': 0.6,
    'horse': 1.2,
    'sheep': 1.0,
    'cow': 1.5,
    'elephant': 3.0,
    'bear': 1.5,
    'zebra': 2.0,
    'giraffe': 3.0,
    'backpack': 0.4,
    'umbrella': 1.0,
    'handbag': 0.4,
    'tie': 0.1,
    'suitcase': 0.5,
    'frisbee': 0.3,
    'skis': 1.5,
    'snowboard': 0.5,
    'sports ball': 0.22,
    'kite': 1.0,
    'baseball bat': 0.06,
    'baseball glove': 0.3,
    'skateboard': 0.2,
    'surfboard': 2.0,
    'tennis racket': 0.3,
    'bottle': 0.0635,
    'wine glass': 0.08,
    'cup': 0.1,
    'fork': 0.02,
    'knife': 0.02,
    'spoon': 0.03,
    'bowl': 0.2,
    'banana': 0.02,
    'apple': 0.08,
    'sandwich': 0.1,
    'orange': 0.08,
    'broccoli': 0.15,
    'carrot': 0.02,
    'hot dog': 0.1,
    'pizza': 0.3048,
    'donut': 0.1,
    'cake': 0.3048,
    'chair': 0.5,
    'couch': 1.8,
    'potted plant': 0.05,
    'bed': 1.3716,
    'dining table': 1.016,
    'toilet': 0.4,
    'tv': 1.10,
    'laptop': 0.4064,
    'mouse': 0.1,
    'remote': 0.05,
    'keyboard': 0.5,
    'cell phone': 0.15,
    'microwave': 0.5,
    'oven': 0.762,
    'toaster': 0.3,
    'sink': 0.762,
    'refrigerator': 1.016,
    'book': 0.1524,
    'clock': 0.81,
    'vase': 0.18,
    'scissors': 0.2,
    'teddy bear': 0.4572,
    'hair dryer': 0.2,
    'toothbrush': 0.02
}


# this function will calculate the distance of the object from the user
# by using the values of the focal_length, which will be given from the frontend,
# the actual_width of the object, which is already stored (the width of all the object of the same
# class are considered to be equal), and the perceived_width, which is returned from the ML model
# resource used: https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
def distance_calculation(focal_length, actual_width, perceived_width):
    distance = focal_length * (actual_width / perceived_width)
    return distance


# this function will take the prediction list from the ML model, which contains all the
# objects detected in a given frame. It also gets the actual_width of all the objects, which
# are already stored and the class_name dictionary, which specifies the name of the class
# given its index. E.g. class '0' represents a car.
def list_creation_objects_with_their_distances(predictions):
    objects_with_positions = [[], [], []]
    for cls, box in predictions:
        objects_with_positions[0].append(object_labels[int(cls.item())])
        actual_width = objects_actual_width[object_labels[int(cls.item())]]
        perceived_width = box[2]
        focal_length = 200
        distance = distance_calculation(focal_length, actual_width, perceived_width)
        objects_with_positions[1].append(distance.ceil().item())

    return objects_with_positions


# calculating the position of the detected object https://www.techscience.com/cmc/v71n1/45390/html
# The whole image is divided into three positions such as top, center and bottom
# as row-wise and left, center and right as column-wise.

# this function will take the prediction list from the ML model, which contains all the
# objects detected in a given frame. additionally, it will also take the dictionary returned by the
# list_creation_objects_with_their_distances and append the position information to the object name and object distance.
# the return value will be a list of dictionaries, each dictionary holding the object name, object distance, and object position
def list_creation_objects_with_their_positions(predictions, objects_with_positions, FRAME_WIDTH, FRAME_HEIGHT):

    # this for loop is used to find the distances of the objects that are found by the ml model
    for cls, box in predictions:
        x_center = box[0]
        if x_center <= FRAME_WIDTH // 3:
            position = "LEFT"
        elif x_center <= (FRAME_WIDTH // 3 * 2):
            position = "CENTER"
        else:
            position = "RIGHT"
        objects_with_positions[2].append(position)
    return objects_with_positions


# this function first performs distance calculation and then performs position calculation (top left, buttom right,
# etc). its return value is a list of objects with their names, distances from the users, and position with respect
# to the user. e.g. {'object': car, 'distance': 2.5 meters, 'position': top left}
def calculate_position(predictions, FRAME_WIDTH, FRAME_HEIGHT):
    objects_with_positions = list_creation_objects_with_their_distances(predictions)
    objects_with_positions = list_creation_objects_with_their_positions(predictions, objects_with_positions,
                                                                        FRAME_WIDTH, FRAME_HEIGHT)
    return objects_with_positions

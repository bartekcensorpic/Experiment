import tensorflow as tf
from typing import Tuple


def load_model(model_path:str, batch_size:int, image_size:Tuple, num_classes:int):
    tflite_model = tf.lite.Interpreter(model_path=model_path)
    # Learn about its input and output details
    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()

    # Resize input and output tensors to handle batch of 16 images
    tflite_model.resize_tensor_input(
        input_details[0]["index"], (batch_size, image_size[0], image_size[1], 3)
    )
    tflite_model.resize_tensor_input(
        output_details[0]["index"], (batch_size, num_classes)
    )
    tflite_model.allocate_tensors()

    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()
    return tflite_model, input_details, output_details



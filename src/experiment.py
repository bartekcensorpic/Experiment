import numpy as np

from typing import Tuple, List
from PIL import ImageFile


from src.model import load_model
from src.images import get_images_from_folder, get_batches

def prepare_experiment(model_path:str,
                       nudes_image_folder:str,
                       non_nudes_image_folder:str,
                       output_folder:str,
                       batch_size:int,
                       image_size:Tuple,
                       classes:List,
                       image_extention:str):

    num_classes = len(classes)
    tfmodel, input_details, output_detail =load_model(model_path,batch_size, image_size, num_classes)

    list_nude_images_paths = get_images_from_folder(nudes_image_folder,image_extention)
    list_non_nude_images_paths = get_images_from_folder(non_nudes_image_folder,image_extention)

    results = run_experiment(tfmodel, input_details, output_detail,list_nude_images_paths, list_non_nude_images_paths, image_size, batch_size)

def run_experiment(tf_model, input_details, output_details, nude_images_paths, non_nude_images_paths, image_size, batch_size):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    nude_batches = get_batches(nude_images_paths,batch_size, image_size)
    non_nude_batches = get_batches(non_nude_images_paths, batch_size, image_size)

    for batch in nude_batches:
        tf_model.set_tensor(input_details[0]["index"], batch)
        tf_model.invoke()
        tflite_q_model_predictions = tf_model.get_tensor(output_details[0]["index"])
        tflite_q_pred_idx = np.round(tflite_q_model_predictions)













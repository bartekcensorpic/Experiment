import numpy as np
import os
from typing import Tuple, List
from PIL import ImageFile
import pandas as pd

from src.model import load_model
from src.images import get_images_from_folder, get_batches
from src.utils import zip_prediction_with_classes, rounded_predictions


class Experiment:
    def __init__(self, model_path: str,
                 nudes_image_folder: str,
                 non_nudes_image_folder: str,
                 output_folder: str,
                 batch_size: int,
                 image_size: Tuple,
                 classes: List,
                 image_extention: str,
                ):
        self.model_path = model_path
        self.nudes_image_folder = nudes_image_folder
        self.non_nudes_image_folder = non_nudes_image_folder
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.image_size = image_size
        self.classes = classes
        self.image_extention = image_extention

    def __prepare(self):
        num_classes = len(self.classes)
        self.tf_model, self.input_details, self.output_details = load_model(self.model_path, self.batch_size, self.image_size, num_classes)

        self.list_nude_images_paths = get_images_from_folder(self.nudes_image_folder, self.image_extention)
        self.list_non_nude_images_paths = get_images_from_folder(self.non_nudes_image_folder, self.image_extention)

        self.df = pd.DataFrame(columns=['real_label']+self.classes)


    def __log_predictions(self, predictions, real_label, corrupted_idx):

        for i in range(predictions.shape[0]):
            if i in corrupted_idx:
                print('this image was corrupted, results skipped')
            else:
                pred = predictions[i]
                print(f"real label: {real_label}, "
                      f"predictions: {zip_prediction_with_classes(pred,self.classes)}, "
                      f"rounded_predictions: {rounded_predictions(pred, self.classes)}")
                loc = len(self.df)
                self.df.loc[loc] =  [real_label] + list(pred)

    def __evaluate_batches(self, batches_gen, real_label):

        for batch, corrupted_idx in batches_gen:
            self.tf_model.set_tensor(self.input_details[0]["index"], batch)
            self.tf_model.invoke()
            tflite_q_model_predictions = self.tf_model.get_tensor(self.output_details[0]["index"])
            tflite_q_pred_rounded = np.around(tflite_q_model_predictions, decimals=3)
            self.__log_predictions(tflite_q_pred_rounded, real_label, corrupted_idx)


    def run_experiment(self):
        self.__prepare()

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        nude_batches = get_batches(self.list_nude_images_paths, self.batch_size, self.image_size)
        non_nude_batches = get_batches(self.list_non_nude_images_paths, self.batch_size, self.image_size)

        self.__evaluate_batches(nude_batches, "nude")
        self.__evaluate_batches(non_nude_batches, "non_nude")

        save_file_path = os.path.join(self.output_details,'results.csv')
        self.df.to_csv(save_file_path, index=False, quotechar='"', encoding='ascii')




















# Censorpic experiment

This script assesses the quality of our nudity detection algorithm (TensorFlow lite model) in the hopes of being able to reduce the volume of indecent underage imagery. The output produces CSV file with the real class of image and our associated prediction.
This script does not save, display or send any images anywhere.

## Before running, the following arguments have to be specified:
--model_path "path/to/tflite/model
--nude_dataset_path "path/to/folder/with/nude/images"
--non_nude_dataset_path "path/to/folder/with/non_nude/images"
--output_path "path/where/output/is to be produces"

Example:

python main.py --model_path "C:\tf_lite_model.tflite" --nude_dataset_path "C:\images\nude" --non_nude_dataset_path "C:\images\non_nude" --output_path "C:\results"

## Requirements
- python 3.x
- TensorFlow 1.15/2.x
- Pillow 6.2
- Pandas 0.25.x

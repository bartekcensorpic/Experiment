import argparse
import os
from datetime import datetime


def check_if_folder_exists(path, create_if_not=False):
    if not os.path.exists(path):
        print(f"path/file '{path}' does not exist")
        if create_if_not:
            os.makedirs(path)


def check_args(args):


    check_if_folder_exists(args.nude_dataset_path)
    check_if_folder_exists(args.non_nude_dataset_path)
    check_if_folder_exists(args.output_path, create_if_not=True)
    check_if_folder_exists(args.model_path)


def process_arguments():

    parser = argparse.ArgumentParser(description="Testing tensorflow lite model in an experiment for the Home Office")

    ########################
    # These arguments have to be adjusted
    ########################
    parser.add_argument("--model_path", type=str, help='path to *.tflite model')
    parser.add_argument("--nude_dataset_path", type=str, help='path to folder with images that are classified as \'nude\'')
    parser.add_argument("--non_nude_dataset_path", type=str, help='path to folder with images that are classified as \'non_nude\'')
    parser.add_argument("--output_path", type=str, help='path to folder where the output will be produces')


    ########################
    # These arguments are ok to be left as they are
    ########################
    parser.add_argument("--pictures_extention", type=str, default='png', help='extentions of the images')
    parser.add_argument("--batch_size", type=int, default=8, help='batch size')
    parser.add_argument("--img_size", type=int,default=224, help='size of images')
    parser.add_argument('-c', '--classes', action='store', dest='classes',
                    type=str, nargs='*', default=['female', 'male', 'nipples','non_nude','penis'],
                    help="classes in correct order, example: -c female male nipples non_nude penis")

    args = parser.parse_args()

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.output_path = os.path.join(args.output_path,f"run_{now}")

    check_args(args)
    return args






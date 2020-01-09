import argparse
from src.args import process_arguments
from src.experiment import Experiment


if __name__ == '__main__':
    args = process_arguments()


    exp = Experiment(model_path=args.model_path,
                       nudes_image_folder= args.nude_dataset_path,
                       non_nudes_image_folder= args.non_nude_dataset_path,
                       output_folder=args.output_path,
                       batch_size= args.batch_size,
                       image_size= (args.img_size, args.img_size),
                       classes= args.classes,
                       image_extention = args.pictures_extention,
                       )

    exp.run_experiment()


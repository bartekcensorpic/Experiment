import argparse
from src.utils import parse_args
from src.experiment import prepare_experiment


if __name__ == '__main__':
    args = parse_args()




    prepare_experiment(model_path=args.model_path,
                       nudes_image_folder= args.nude_dataset_path,
                       non_nudes_image_folder= args.non_nude_dataset_path,
                       output_folder=args.output_path,
                       batch_size= args.batch_size,
                       image_size= (args.img_size, args.img_size),
                       classes= args.classes,
                       image_extention = args.pictures_extention
                       )


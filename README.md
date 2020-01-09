# HomeOfficeExperiment

Before running, the following arugments have to be specified:
--model_path "path/to/tflite/model
--nude_dataset_path "path/to/folder/with/nude/images"
--non_nude_dataset_path "path/to/folder/with/non_nude/images"
--output_path "path/where/output/is to be produces"

example:

python main.py --model_path "C:\tf_lite_model.tflite" --nude_dataset_path "C:\images\nude" --non_nude_dataset_path "C:\images\non_nude" --output_path" C:\results"



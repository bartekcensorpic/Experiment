from pathlib import Path
from PIL import Image
import numpy as np
import traceback

def get_images_from_folder(folder_path,image_extention):
    images_in_folder = list(Path(folder_path).glob(f"*.{image_extention}"))
    return images_in_folder

def get_batches(images_path_list,batch_size, image_size):
    n_batches = len(images_path_list) // batch_size
    for i in range(n_batches):
        start = i
        end = i+batch_size
        img_paths = images_path_list[start:end]
        normalised_imgs = []
        corrupted_idx = []
        for index, path in enumerate(img_paths):
            try:
                image = Image.open(path).convert('RGB').resize(image_size)
            except:
                traceback.print_exc()
                image = np.zeros(shape=(image_size+(3,)))
                corrupted_idx.append(index)
                print(f"[Info] image {path} can not be opened. Replacing with empty one")
                debug = 5

            pix = np.array(image)
            pix = pix.astype(np.float32)
            pix /= 127.5
            pix -= 1.

            normalised_imgs.append(pix)

        yield (np.array(normalised_imgs).reshape((batch_size,)+ image_size +(3,)),corrupted_idx)





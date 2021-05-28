import io
import os
import sys
from typing import List

import requests
from PIL import Image
from simple_image_download import simple_image_download as simp
from tqdm import tqdm

from app.face_embedder import FaceEmbedder


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


class ImageCollector:
    # LOGGER = LoggingHelper("Image").logger
    def __init__(self, number_images_to_save: int = 5,
                 number_of_images: int = 10,
                 extension: set = {'.jpg', '.jpeg'}):

        self.number_images_to_save = number_images_to_save
        self.number_of_images = number_of_images
        self.extension = extension
        self.response = simp.simple_image_download
        self.face_embedder = FaceEmbedder()

    def extract_and_save_image_from_url(self,
                                        name: str,
                                        list_urls_unique: List[str],
                                        path_to_current_actor: str,
                                        check_num_face: bool = True):

        label_image = 0
        counter_number_images_to_save = 0

        for image_url in list_urls_unique[:self.number_of_images]:
            try:
                label_image += 1
                img_data = requests.get(image_url).content

                if check_num_face:
                    img_data_picture = Image.open(io.BytesIO(img_data)).convert('RGB')

                    cropped_image = self.face_embedder.crop_image(img_data_picture)
                    number_of_persons = len(cropped_image)

                else:
                    number_of_persons = 1

                if number_of_persons == 1:
                    with open('%s/%s_%i.jpg' % (path_to_current_actor,
                                                name,
                                                label_image), 'wb') as writer:
                        writer.write(img_data)
                        counter_number_images_to_save += 1
                        if counter_number_images_to_save >= self.number_images_to_save:
                            break
            except:
                pass

    def collect_images(self,
                       list_names,
                       suffix,
                       folder_specification,
                       check_num_face: bool = True
                       ):

        name_global_folder = "data_" + folder_specification
        create_folder(name_global_folder)

        maximum_images_to_look_for_on_google = 2 * self.number_of_images + 1

        with tqdm(total=len(list_names), file=sys.stdout) as pbar:
            for name in list_names:
                pbar.update(1)
                path_to_current_actor = name_global_folder + "/" + name
                create_folder(path_to_current_actor)

                name_with_suffix = name + suffix
                list_urls_duplicates = self.response(). \
                    urls(name_with_suffix,
                         maximum_images_to_look_for_on_google,
                         extensions=self.extension)

                list_urls_unique = list(set(list_urls_duplicates))

                self.extract_and_save_image_from_url(name, list_urls_unique, path_to_current_actor, check_num_face)



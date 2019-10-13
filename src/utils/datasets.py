import os
import logging
from typing import Dict, List
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlickrDataset:
    # Adapted for working with the Flickr8k and Flickr30k dataset.

    def __init__(self, images_dir_path: str, texts_path: str):
        self.images_dir_path = images_dir_path
        self.img_path_caption = self.parse_captions_filenames(texts_path)
        logger.info("Object variables set...")
        self.all_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        print("adding special tokens")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    @staticmethod
    def parse_captions_filenames(texts_path: str) -> Dict[str, List[str]]:
        """Creates a dictionary that holds:

        Key: The full path to the image.
        Value: A list of lists where each token in the inner list is a word. The number
        of sublists is 5.

        Args:
            texts_path: Path where the text doc with the descriptions is.

        Returns:
            A dictionary that represents what is explained above.

        """
        img_path_caption: Dict[str, List[str]] = {}
        with open(texts_path, "r") as file:
            for line in file:
                line_parts = line.split("\t")
                image_tag = line_parts[0].partition("#")[0]
                caption = line_parts[1]
                if image_tag not in img_path_caption:
                    img_path_caption[image_tag] = []
                img_path_caption[image_tag].append(caption)

        return img_path_caption

    @staticmethod
    def get_data_wrapper(
        imgs_file_path: str,
        img_path_caption: Dict[str, List[str]],
        images_dir_path: str,
    ):
        """Returns the image paths, the captions and the lengths of the captions.

        Args:
            imgs_file_path: A path to a file where all the images belonging to the
            validation part of the dataset are listed.
            img_path_caption: Image name to list of captions dict.
            images_dir_path: A path where all the images are located.

        Returns:
            Image paths, captions and lengths.

        """
        image_paths = []
        captions = []
        with open(imgs_file_path, "r") as file:
            for image_name in file:
                # Remove the newline character at the end
                image_name = image_name[:-1]
                # If there is no specified codec in the name of the image append jpg
                if not image_name.endswith(".jpg"):
                    image_name += ".jpg"
                for i in range(5):
                    image_paths.append(os.path.join(images_dir_path, image_name))
                    captions.append(img_path_caption[image_name][i])

        assert len(image_paths) == len(captions)

        return image_paths, captions


class FlickrDatasetTrain(FlickrDataset, TorchDataset):
    def __init__(
        self, images_dir_path: str, texts_path: str, train_images_file_path: str
    ):
        super().__init__(images_dir_path, texts_path)
        self.image_paths, self.captions = self.get_data_wrapper(
            train_images_file_path, self.img_path_caption, self.images_dir_path
        )
        self.train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx])
        image_train_transformed = self.train_transform(image)
        image_all_transformed = self.all_transform(image_train_transformed)

        caption = torch.tensor(
            self.tokenizer.encode(self.captions[idx], add_special_tokens=True)
        )

        return image_all_transformed, caption


class FlickrDatasetVal(FlickrDataset, TorchDataset):
    def __init__(
        self, images_dir_path: str, texts_path: str, val_images_file_path: str
    ):
        super().__init__(images_dir_path, texts_path)
        self.image_paths, self.captions = self.get_data_wrapper(
            val_images_file_path, self.img_path_caption, self.images_dir_path
        )
        self.val_transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224)]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx])
        image_val_transformed = self.val_transform(image)
        image_all_transformed = self.all_transform(image_val_transformed)

        caption = torch.tensor(
            self.tokenizer.encode(self.captions[idx], add_special_tokens=True)
        )

        return image_all_transformed, caption


class FlickrDatasetTest(FlickrDataset, TorchDataset):
    def __init__(
        self, images_dir_path: str, texts_path: str, test_images_file_path: str
    ):
        super().__init__(images_dir_path, texts_path)
        self.image_paths, self.captions = self.get_data_wrapper(
            test_images_file_path, self.img_path_caption, self.images_dir_path
        )
        self.test_transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224)]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx])
        image_test_transformed = self.test_transform(image)
        image_all_transformed = self.all_transform(image_test_transformed)

        caption = torch.tensor(self.tokenizer.encode(self.captions[idx]))

        return image_all_transformed, caption

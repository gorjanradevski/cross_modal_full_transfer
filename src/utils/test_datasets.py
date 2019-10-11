import pytest
from utils.datasets import FlickrDataset


@pytest.fixture
def flickr_texts_path():
    return "data/testing_assets/flickr_tokens.txt"


@pytest.fixture
def flickr_images_path():
    return "data/testing_assets/flickr_images/"


@pytest.fixture
def flickr_train_path():
    return "data/testing_assets/flickr_train.txt"


@pytest.fixture
def flickr_val_path():
    return "data/testing_assets/flickr_val.txt"


def test_flickr_parse_captions_filenames(flickr_texts_path):
    img_path_caption = FlickrDataset.parse_captions_filenames(flickr_texts_path)
    unique_img_paths = set()
    for img_path in img_path_caption.keys():
        assert len(img_path_caption[img_path]) == 5
        unique_img_paths.add(img_path)
    assert len(unique_img_paths) == 5

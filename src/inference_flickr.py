import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from utils.datasets import FlickrDatasetTest
from utils.evaluator import Evaluator
from utils.data_loading_utils import collate_pad_batch
from models import ImageTextMatchingModel, TripletLoss


def inference(
    images_path: str,
    sentences_path: str,
    test_imgs_file_path: str,
    batch_size: int,
    load_model_path: str,
    joint_space: int,
):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_test = FlickrDatasetValTest(
        images_path, sentences_path, test_imgs_file_path
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_pad_batch,
        pin_memory=True,
    )
    # Create the model
    model = nn.DataParallel(ImageTextMatchingModel(joint_space)).to(device)
    # Load model
    model.load_state_dict(torch.load(load_model_path))
    # Set model in evaluation mode
    model.train(False)
    # Create evaluator
    evaluator = Evaluator(len(dataset_test), joint_space)
    with torch.no_grad():
        evaluator.reset_all_vars()
        for images, sentences in tqdm(test_loader):
            images, sentences = images.to(device), sentences.to(device)
            embedded_images, embedded_sentences = model(images, sentences)
            evaluator.update_embeddings(
                embedded_images.cpu().numpy().copy(),
                embedded_sentences.cpu().numpy().copy(),
            )

    print("=============================")
    print(
        f"Image-text recall at 1, 5, 10: "
        f"{evaluator.image2text_recall_at_k()} \n"
        f"Text-image recall at 1, 5, 10: "
        f"{evaluator.text2image_recall_at_k()}"
    )
    print("=============================")


def main():
    # Without the main sentinel, the code would be executed even if the script were
    # imported as a module.
    args = parse_args()
    inference(
        args.images_path,
        args.texts_path,
        args.test_imgs_file_path,
        args.batch_size,
        args.load_model_path,
        args.joint_space,
    )


def parse_args():
    """Parse command line arguments.
    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser(
        description="Trains a model on the Flickr8k or Flickr30k datasets."
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_Dataset",
        help="Path where all images are.",
    )
    parser.add_argument(
        "--texts_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr8k.token.txt",
        help="Path to the file where the image to caption mappings are.",
    )
    parser.add_argument(
        "--test_imgs_file_path",
        type=str,
        default="data/Flickr8k_dataset/Flickr8k_text/Flickr_8k.testImages.txt",
        help="Path to the file where the test images names are included.",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default="models/best.pt",
        help="From where to load the model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of the batch."
    )
    parser.add_argument(
        "--joint_space",
        type=int,
        default=512,
        help="The joint space where the encodings will be projected.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()

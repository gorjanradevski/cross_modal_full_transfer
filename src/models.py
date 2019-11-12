import torch
from torch import nn
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
from transformers import BertModel
import torch.nn.functional as F


class L2Normalize(nn.Module):
    def __init__(self):
        super(L2Normalize, self).__init__()

    def forward(self, x):
        norm = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
        normalized = torch.div(x, norm)

        return normalized


class ImageEncoder(nn.Module):
    def __init__(self, finetune: bool):
        super(ImageEncoder, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained("efficientnet-b5")

        for param in self.efficientnet.parameters():
            param.requires_grad = finetune

    def forward(self, images: torch.Tensor):
        embedded_images = torch.flatten(
            self.efficientnet.extract_features(images), start_dim=1
        )

        return embedded_images


class SentenceEncoder(nn.Module):
    def __init__(self, finetune: bool):
        super(SentenceEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        #  https://arxiv.org/abs/1801.06146

        for param in self.bert.parameters():
            param.requires_grad = finetune

    def forward(self, sentences: torch.Tensor):
        # TODO: Check about masking the padding
        # https://arxiv.org/abs/1801.06146
        hidden_states = self.bert(sentences)
        max_pooled = torch.max(hidden_states[0], dim=1)[0]
        mean_pooled = torch.mean(hidden_states[0], dim=1)
        # TODO: Check again about the CLS hidden state
        last_state = hidden_states[0][:, 0, :]
        embedded_sentences = torch.cat([last_state, max_pooled, mean_pooled], dim=1)

        return embedded_sentences


class Projector(nn.Module):
    def __init__(self, input_space, joint_space: int):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_space, joint_space)
        self.bn = nn.BatchNorm1d(joint_space)
        self.fc2 = nn.Linear(joint_space, joint_space)
        self.l2_normalize = L2Normalize()

    def forward(self, embeddings: torch.Tensor):
        projected_embeddings = self.fc2(self.bn(F.relu(self.fc1(embeddings))))

        return self.l2_normalize(projected_embeddings)


class ImageTextMatchingModel(nn.Module):
    def __init__(self, joint_space: int, finetune: bool = False):
        super(ImageTextMatchingModel, self).__init__()
        self.finetune = finetune
        # Image encoder
        self.image_encoder = ImageEncoder(finetune)
        self.image_encoder.eval()
        self.image_projector = Projector(1280 * 7 * 7, joint_space)
        # Sentence encoder
        self.sentence_encoder = SentenceEncoder(finetune)
        self.sentence_encoder.eval()
        self.sentence_projector = Projector(768 * 3, joint_space)

    def forward(self, images: torch.Tensor, sentences: torch.Tensor):
        embedded_images = self.image_encoder(images)
        embedded_sentences = self.sentence_encoder(sentences)

        return (
            self.image_projector(embedded_images),
            self.sentence_projector(embedded_sentences),
        )

    def train(self, mode: bool = True):
        if self.finetune and mode:
            self.image_encoder.train()
            self.sentence_encoder.train()
            self.image_projector.train(True)
            self.sentence_projector.train(True)
        elif mode:
            self.image_projector.train(True)
            self.sentence_projector.train(True)
        else:
            self.image_encoder.train(False)
            self.sentence_encoder.train(False)
            self.image_projector.train(False)
            self.sentence_projector.train(False)


class TripletLoss(nn.Module):
    # As per https://github.com/fartashf/vsepp/blob/master/model.py

    def __init__(self, margin: float):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = torch.matmul(im, s.t())
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        identity = Variable(mask)
        if torch.cuda.is_available():
            identity = identity.cuda()
        cost_s = cost_s.masked_fill_(identity, 0)
        cost_im = cost_im.masked_fill_(identity, 0)

        # keep the maximum violating negative for each query
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

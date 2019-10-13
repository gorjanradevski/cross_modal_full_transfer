import torch
from torch import nn
from torch.autograd import Variable
from torchvision.models import resnet152
from transformers import BertModel


class L2Normalize(nn.Module):
    def __init__(self):
        super(L2Normalize, self).__init__()

    def forward(self, x):
        norm = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
        X = torch.div(x, norm)

        return X


class ImageEncoder(nn.Module):
    def __init__(self, joint_space: int, finetune: bool):
        super(ImageEncoder, self).__init__()
        self.resnet = torch.nn.Sequential(
            *(list(resnet152(pretrained=True).children())[:-1])
        )
        self.fc = nn.Linear(2048, joint_space)
        self.l2_normalize = L2Normalize()

        for param in self.resnet.parameters():
            param.requires_grad = finetune

    def forward(self, images: torch.Tensor):
        embedded_images = torch.flatten(self.resnet(images), start_dim=1)
        embedded_images = self.fc(embedded_images)

        return self.l2_normalize(embedded_images)


class SentenceEncoder(nn.Module):

    # TODO: Option to freeze sentence encoder

    def __init__(self, joint_space: int, finetune: bool):
        super(SentenceEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, joint_space)
        self.l2_normalize = L2Normalize()

        for param in self.bert.parameters():
            param.requires_grad = finetune

    def forward(self, sentences: torch.Tensor):
        embedded_sentences = torch.mean(self.bert(sentences)[0], dim=1)
        embedded_sentences = self.fc(embedded_sentences)

        return self.l2_normalize(embedded_sentences)


class ImageTextMatchingModel(nn.Module):
    def __init__(
        self,
        joint_space: int,
        finetune_image_encoder: bool,
        finetune_sentence_encoder: bool,
    ):
        super(ImageTextMatchingModel, self).__init__()
        self.finetune_image_encoder = finetune_image_encoder
        self.finetune_sentence_encoder = finetune_sentence_encoder
        self.image_encoder = ImageEncoder(joint_space, finetune_image_encoder)
        self.image_encoder.eval()
        self.sentence_encoder = SentenceEncoder(joint_space, finetune_sentence_encoder)
        self.sentence_encoder.eval()

    def forward(self, images: torch.Tensor, sentences: torch.Tensor):
        embedded_images = self.image_encoder(images)
        embedded_sentences = self.sentence_encoder(sentences)

        return embedded_images, embedded_sentences

    def set_train(self):
        if self.finetune_image_encoder:
            self.image_encoder.train()
        if self.finetune_sentence_encoder:
            self.sentence_encoder.train()

    def set_eval(self):
        self.image_encoder.eval()
        self.sentence_encoder.eval()


class TripletLoss(nn.Module):
    # As per https://github.com/fartashf/vsepp/blob/master/model.py

    def __init__(self, margin: float = 0, batch_hard: bool = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.batch_hard = batch_hard

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
        if self.batch_hard:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

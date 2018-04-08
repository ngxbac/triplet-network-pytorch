import torch.nn as nn
from torchvision.models import *


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.base_model = resnet50(pretrained=True)
        # self.base_model.fc = nn.Linear(in_features=2048, out_features=128, bias=True)
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=128),
            nn.Linear(in_features=128, out_features=128)
        )

    def forward(self, x):
        return self.base_model(x)

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingL2(Embedding):
    def __init__(self):
        super(EmbeddingL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)
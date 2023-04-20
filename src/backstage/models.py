import torch
import torchvision
import torch.nn as nn


def convLayer(chin, chout, k, down=True, norm=True):
    layers = [
        nn.Conv2d(chin, chout, kernel_size=k, stride=2 if down else 1, padding=(k - 1) // 2, bias=not norm)
    ]
    if norm:
        layers += [nn.BatchNorm2d(chout)]
    layers += [nn.ReLU()]

    return nn.Sequential(*layers)


class Classifier(nn.Module):
    def __init__(self, args: dict, chin=3, nCategories=120):
        super().__init__()

        self.classifier_params = args["classifier"]
        model = torchvision.models.efficientnet_b1(pretrained=True)
        n_in = model.classifier[1].in_features # pocet kanalov

        self.features = model.features

        self.n_hidden = self.classifier_params["nHidden"]

        self.detector = nn.Sequential(
            nn.Conv2d(n_in, self.n_hidden, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.n_hidden),
            nn.ReLU(),
            nn.Conv2d(self.n_hidden, 5, kernel_size=1, stride=1, padding=0)
            # nn.Conv2d(n_in, 5, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        f = self.features(x)
        c = self.detector(f)
        return c

    def get_params(self):
        return self.detector.parameters()

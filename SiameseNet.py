import torch
import torch.nn as nn
import torchvision.models as models


class SiameseResNet(nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.dev = torch.device(dev)
        # Load pre-trained ResNet-18 models
        self.cnn1 = self.load_pretrained_model()
        self.cnn2 = self.load_pretrained_model()

        for param in list(self.cnn1.parameters()):
            param.requires_grad = False
        for param in list(self.cnn2.parameters()):
            param.requires_grad = False

        # Add a new layer on top
        self.classifier = nn.Sequential(
            nn.LayerNorm(4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.LayerNorm(4096),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.LayerNorm(4096),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.LayerNorm(4096),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )
        self.to(self.dev)

    def load_pretrained_model(self):
        # Download and load a pre-trained ResNet-18 model
        model = models.resnet50(weights=True)
        return nn.Sequential(*list(model.children())[:-1])

    def forward_once(self, x1, x2):
        # Forward pass through both of the CNN branches
        out1 = self.cnn1(x1)
        out1 = out1.view(out1.size()[0], -1)
        out2 = self.cnn2(x2)
        out2 = out2.view(out2.size()[0], -1)
        return out1, out2

    def forward(self, input1, input2):
        # Forward pass through both CNN branches
        output1, output2 = self.forward_once(input1.to(self.dev), input2.to(self.dev))
        # Concatenate the outputs of both branches
        combined = torch.cat((output1, output2), dim=1)
        # Forward pass through the new layer
        output = self.classifier(combined)
        return output  # Apply sigmoid activation for binary classification

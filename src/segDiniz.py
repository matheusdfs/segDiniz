import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from enum import Enum

# class syntax
class OperationMode(Enum):
    Train = 1
    Evaluate = 2
    Run = 3

class segDiniz():
    def __init__(self):
        pass

    def train(self):
        # First step: load dataset
        self.load_training_set()

        # Second step: loss function
        self.init_loss_function()

        #Third step: model
        self.init_model()

        #Forth step: optimizer
        self.init_optimizer()

    def execute(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def init_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def init_model(self):
        self.model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        )

    def init_loss_function(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def load_training_set(self):
        # Initialize training dataset
        self.training_set = datasets.Cityscapes(
            root="../data/cityscapes/", 
            split="train", 
            target_type="semantic"
        )

        # Initialize training dataloader
        self.training_loader = torch.utils.data.DataLoader(
            self.training_set, 
            batch_size=4, 
            shuffle=True
        )

        print('Trainig set initialized successfully: it has {} instances'.format(len(self.training_set)))

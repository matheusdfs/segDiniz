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

        self.load_validation_set()

        # Second step: loss function
        self.init_loss_function()

        # Third step: model
        self.init_model()

        # Forth step: optimizer
        self.init_optimizer()

        # Fifth step: execution of the epochs
        #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        #writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        EPOCHS = 5

        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number)


            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            #writer.add_scalars('Training vs. Validation Loss',
            #                { 'Training' : avg_loss, 'Validation' : avg_vloss },
            #                epoch_number + 1)
            #writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format("opa", epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

    def train_one_epoch(self, epoch_index):
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
                #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
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

    def load_validation_set(self):
        # Initialize validation dataset
        self.validation_set = datasets.Cityscapes(
            root="../data/cityscapes/", 
            split="val", 
            target_type="semantic"
        )

        # Initialize validation dataloader
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_set, 
            batch_size=4, 
            shuffle=True
        )

        print('Validation set initialized successfully: it has {} instances'.format(len(self.validation_set)))
        pass
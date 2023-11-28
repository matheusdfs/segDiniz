import torch
import logger
import pandas as pd
import configparser
import numpy as np
from matplotlib import pyplot as plt
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from datetime import datetime
#from torch.utils.tensorboard import    SummaryWriter

from enum import Enum

PATH = "../checkpoint/checkpoint.data"

# class syntax
class OperationMode(Enum):
    Train = 1
    Evaluate = 2
    Run = 3

class segDiniz():
    def __init__(self, args):

        logger.log("///// INITIALIZATION PARAMETERS /////")

        # Handle device
        self.device = torch.device(args.device)
        logger.log("// The device choosed is: " + str(self.device))

        # Check execution mode
        if args.train:
            logger.log("// The NN will be trained")

        # Check configuration file
        if args.checkpoint_file:
            logger.log("// The default.ini file will be used as configuration file")

        logger.log("/////")

    def predict(self):
        logger.log('Predict mode selected, start initialization!')

        # First step: load training configuration
        self.load_predict_config_file()

        # Second step: model
        self.init_model()



    def train(self):
        logger.log('Training mode selected, start initialization!')

        # First step: load training configuration
        self.load_train_config_file()

        # Second step: load training set
        self.load_training_set()

        # Third step: load validation set
        self.load_validation_set()

        #Forth step: loss function
        self.init_loss_function()

        #Fifth step: model
        self.init_model()

        #Sixth step: optimizer
        self.init_optimizer()

        logger.log("Training initialization done!")
        
        #Seventh step: start training
        self.start_training()

    def start_training(self):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

        for epoch in range(self.epoch):

            self.model.train(True)

            avg_loss = self.train_one_epoch(epoch)

            running_vloss = 0.0
            self.model.eval()

            if self.device == 'cuda':
                torch.cuda.synchronize()

            with torch.no_grad():
                for i, vdata in enumerate():
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            #writer.add_scalars(
            #    'Training vs. Validation Loss',
            #    { 'Training' : avg_loss, 'Validation' : avg_vloss },
            #    epoch_number + 1
            #)
            #writer.flush()

            if avg_loss < best_vloss:
                best_vloss = avg_vloss
                model_path = '../checkpoint/model_{}_{}'.format(timestamp, epoch)
                torch.save(self.model, model_path)
        
            

    def train_one_epoch(self, epoch_index):
        running_loss = 0.
        last_loss = 0.

        logger.log(f"Training epoch {epoch_index}")

        for i, data in enumerate(self.training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # inputs:
            #    it is a tensor, with the dimension [batch_size, 3, img_dim, img_dim]
            #    it is a RGB image

            # labels:
            #    it is a tensor, with the dimension [batch_size, 1, img_dim, img_dim]
            #    its a one dimension image, normalized between 0-1 and have the correct layers to the input image

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            prediction = outputs['out'][0].detach().numpy()
            prediction = np.argmax(prediction, axis=0)

            outputs['out'][0] = coco2cityscapes(prediction)

            prediction2 = outputs['out'][1].detach().numpy()
            prediction2 = np.argmax(prediction2, axis=0)

            print(prediction2.shape)

            outputs['out'][1] = coco2cityscapes(prediction2)

            # outputs['out’]
            #    it is a tensor, with the dimension [batch_size, num_categories, img_dim, img_dim]
            #    probability of a pixel be a certain class

            # Compute the loss and its gradients
            labels2 = torch.argmax(labels, dim=1)
            loss = self.loss_fn(outputs['out'], labels2)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            
            print('  iteration {} loss: {}'.format(i + 1, running_loss))
            #tb_x = epoch_index * len(self.training_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

            print_expected_reality("img{}".format(i), labels[0][0], prediction)

            if i % 7 == 0:
                torch.save(self.model, f'../checkpoint/model{i}')


        return last_loss

    def init_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def init_model(self):
        self.model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        )
        self.model.to(self.device)
       

    def init_loss_function(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def load_training_set(self):

        # Initialize training dataset
        self.training_set = datasets.Cityscapes(
            root="../data/cityscapes/", 
            split="train", 
            target_type="semantic",
            transform=transforms.Compose([
                #transforms.Resize((512, 512)),
                transforms.CenterCrop((1280, 720)),
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                #transforms.Resize((512, 512)),
                transforms.CenterCrop((1280, 720)),
                transforms.ToTensor()
            ])
        )

        # Initialize training dataloader
        self.training_loader = torch.utils.data.DataLoader(
            self.training_set, 
            batch_size=self.batch_size, 
            shuffle=True,
        )

        logger.info('Trainig set initialized successfully: it has {} instances'.format(len(self.training_set)))

    def load_validation_set(self):

        self.validation_set = datasets.Cityscapes(
            root="../data/cityscapes/", 
            split="val", 
            target_type="semantic",
            transform=transforms.Compose([
                #transforms.Resize((512, 512)),
                transforms.CenterCrop((1280, 720)),
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                #transforms.Resize((512, 512)),
                transforms.CenterCrop((1280, 720)),
                transforms.ToTensor()
            ])
        )

        # Initialize training dataloader
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_set, 
            batch_size=self.batch_size, 
            shuffle=False
        )

        logger.info('Validation set initialized successfully: it has {} instances'.format(len(self.validation_set)))

    def load_train_config_file(self):
        config = configparser.ConfigParser()
        config.read('../config/default.ini')

        self.epoch = int(config['TRAIN']['Epoch'])
        self.batch_size = int(config['TRAIN']['BatchSize'])

    def load_predict_config_file(self):
        pass
    
    def show_image(self, outputs):
        prediction = outputs['out'].squeeze(0).cpu().detach().numpy()
        prediction = np.argmax(prediction, axis=1)

        plt.imshow(prediction, interpolation='nearest')

def tensor_to_csv(tensor, filename):
    t_np = tensor.numpy() #convert to Numpy array
    df = pd.DataFrame(t_np) #convert to a dataframe
    df.to_csv(filename,index=False) #save to file

def print_expected_reality(name, expected, reality):
    f, axarr = plt.subplots(1,2) 

    axarr[0].imshow(expected)
    axarr[0].title.set_text("Expected")
    axarr[1].imshow(reality)
    axarr[1].title.set_text("Reality")

    plt.savefig("../plot/" + name)
    plt.close()

def coco2cityscapes(predict):
    output = torch.zeros(predict.shape, dtype=torch.int32)

    for i, row in enumerate(predict):
        for j, tensor in enumerate(row):
            if tensor == 2:
                output[i][j] = 18
            elif tensor == 6:
                output[i][j] = 15
            elif tensor == 7:
                output[i][j] = 13
            elif tensor == 15:
                output[i][j] = 11
            else:
                output[i][j] = 0

    return output
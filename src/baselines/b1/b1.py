import math

import torch
from sympy import ceiling
from torch import nn
import torchvision.models as models
import torch.optim as optim


class ImageLevelModel(nn.Module):
    def __init__(self, num_classes):
        super(ImageLevelModel, self).__init__()
        self.backbone_model = None
        self.classifier = None
        self.num_classes = num_classes

        self.optimizer = None
        self.criterion = None
        self.accuracy = None
        self.save_interval = None

        self.prepare_model()

    def prepare_model(self):
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*(list(model.children())[:-1]))
        fc2 = nn.Linear(2048, self.num_classes)

        fc_layers = nn.Sequential(
            fc2
        )
        self.backbone_model = model
        self.classifier = fc_layers

    def optimizers(self, optim):
        optims = dict(
            Adam=torch.optim.Adam([{'params': self.backbone_model.parameters()},
                                   {'params': self.classifier.parameters()}],
                                  lr=optim['lr'],
                                  weight_decay=optim['weight_decay']),
            # SGD=torch.optim.SGD([{'params': self.backbone_model.parameters()},
            #                      {'params': self.classifier.parameters()}],
            #                     lr=optim['lr'],
            #                     weight_decay=optim['weight_decay'])
        )
        return optims[optim['optimizer']]

    def set_metrics(self, optimizer, criterion, accuracy, save_interval=3):
        self.optimizer = self.optimizers(optimizer)
        self.optimizer = torch.optim.Adam([{'params': self.backbone_model.parameters()},
                                           {'params': self.classifier.parameters()}],
                                          lr=optimizer['lr'],
                                          weight_decay=optimizer['weight_decay'])
        self.criterion = criterion
        self.accuracy = accuracy
        self.save_interval = save_interval

    def train_model(self, trainLoader, backbone_model, classifier, optimizer, device):
        backbone_model.train()
        classifier.train()

        criterion = self.criterion
        train_loss_per_batch = 0
        total_correct_predictions = 0
        num_of_steps = len(trainLoader.dataset) / len(trainLoader)

        inter = math.floor(num_of_steps * 0.25)
        for batch_idx, (data, target) in enumerate(trainLoader):
            # if batch_idx > 5:
            #     break

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = backbone_model(data)
            output = output.view(output.size(0), -1)
            output = classifier(output)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss_per_batch += loss.item() * data.size(0)

            prediction = torch.argmax(output, dim=1)
            correct_predictions = sum(pred == tar for pred, tar in zip(prediction, target))

            total_correct_predictions += correct_predictions
            # if (num_of_steps/(batch_idx+1)) % inter == 0:
            #     print(f'steps: {batch_idx*num_of_steps}/{trainLoader.dataset}')


        total_loss = train_loss_per_batch / len(trainLoader.dataset)
        total_accuracy = total_correct_predictions / len(trainLoader.dataset)

        return backbone_model, classifier, optimizer, total_loss, total_accuracy

    def eval_model(self, valLoader, device):

        backbone_model = self.backbone_model
        classifier = self.classifier

        val_loss_per_batch = 0
        total_correct_predictions = 0

        with torch.no_grad():
            backbone_model.eval(), classifier.eval()

            for batch_idx, (data, target) in enumerate(valLoader):
                if batch_idx > 2:
                    break
                data, target = data.to(device), target.to(device)

                print(f'batch steps in VAL: {batch_idx}')
                output = backbone_model(data)
                output = output.view(output.size(0), -1)
                output = classifier(output)

                loss = criterion(output, target)
                val_loss_per_batch += loss.item() * data.size(0)

                prediction = torch.argmax(output, dim=1)
                correct_predictions = sum(pred == tar for pred, tar in zip(prediction, target))

                total_correct_predictions += correct_predictions

        total_loss = val_loss_per_batch / len(valLoader.dataset)
        total_acc = total_correct_predictions / len(valLoader.dataset)

        return total_loss, total_acc

    def forward(self, trainLoader, valLoader, epochs, output_path):
        print(self.backbone_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        backbone_model = self.backbone_model
        classifier = self.classifier
        save_interval = self.save_interval

        # optimizer = torch.optim.Adam([{'params': self.backbone_model.parameters()},
        #                                    {'params': self.classifier.parameters()}],
        #                                   lr=optimizer['lr'],
        #                                   weight_decay=optimizer['weight_decay'])
        optimizer = self.optimizer

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            print(f'epoch {epoch+1}/{epochs}, ')
            backbone_model, classifier, optimizer, train_loss, train_accuracy = self.train_model(trainLoader,
                                                                                                 backbone_model,
                                                                                                 classifier,
                                                                                                 optimizer,
                                                                                                 device)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            val_loss, val_accuracy = self.eval_model(valLoader, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            if (epoch + 1) % save_interval == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "backbone_model_state_dict": self.backbone_model.state_dict(),
                    "classifier_state_dict": self.classifier.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                torch.save(checkpoint, f'{output_path}/b1/volleyball_checkpoint{epoch + 1}.pth')

        torch.save(self.backbone_model.state_dict(), f'{output_path}/b1/backbone_model_state_dict.pth')
        torch.save(self.classifier.state_dict(), f'{output_path}/b1/classifier_state_dict.pth')

        loss_acc_epochs = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_acc": train_accuracies,
            "val_acc": val_accuracies,
        }

        with open(f'{output_path}/loss_acc.pickle', 'wb') as file:
            pickle.dump(loss_acc_epochs, file)

    def test_model(self, testLoader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        total_correct_predictions = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            print(f'batch steps in TEST: {batch_idx}')
            output = self.backbone_model(data)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)

            prediction = torch.argmax(output, dim=1)
            correct_predictions = sum(pred == tar for pred, tar in zip(prediction, target))

            total_correct_predictions += correct_predictions

        total_acc = total_correct_predictions / len(testLoader.dataset)
        return total_acc


import pickle
from src.volleyball_data_loader import VolleyBallDataSet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

if __name__ == '__main__':
    root_dataset = '/home/ma7moud-5aled/PycharmProjects/vollyball_GAR_project/volleyball_dataset/'
    train_path = root_dataset + 'videos/train'
    val_path = root_dataset + 'videos/val'

    annot_pkl_path = root_dataset + "volleyball-baseline-annotations/b1_annot.pickle"
    root_training_output_path = "/home/ma7moud-5aled/PycharmProjects/vollyball_GAR_project/training-outputs"

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(annot_pkl_path, 'rb') as file:
        data_annot = pickle.load(file)

    train_annot = data_annot["train"]
    val_annot = data_annot["val"]

    batch_size = 32
    train_loader = DataLoader(VolleyBallDataSet(train_path, train_annot, preprocess=preprocess), batch_size=batch_size)
    val_loader = DataLoader(VolleyBallDataSet(val_path, val_annot, preprocess=preprocess), batch_size=batch_size)

    print(f'len of dataset: {train_loader.__len__()}')
    print(f'len of dataset: {len(train_loader)}')
    print(f'len of dataset: {len(train_loader.dataset)}')

    num_classes = 8
    epochs = 10
    my_model = ImageLevelModel(num_classes)

    optimizer = "Adam"
    lr = 1e-2
    weight_decay = 1e-3
    optim_params = {
        "optimizer": optimizer,
        "lr": lr,
        "weight_decay": weight_decay
    }
    criterion = torch.nn.CrossEntropyLoss()
    acc = "accuracy"
    # my_model.set_metrics(optimizer=optim_params, criterion=criterion, accuracy=acc)
    #
    # my_model.forward(train_loader, val_loader, epochs, output_path=root_training_output_path)

    backbone_model_state_path = '/home/ma7moud-5aled/PycharmProjects/vollyball_GAR_project/training-outputs/b1/backbone_model_state_dict.pth'
    classifier_model_state_path = '/home/ma7moud-5aled/PycharmProjects/vollyball_GAR_project/training-outputs/b1/classifier_state_dict.pth'

    test_model = ImageLevelModel(num_classes)
    with open(backbone_model_state_path, 'rb') as backnone, open(classifier_model_state_path, 'rb') as classifier:
        test_model.backbone_model.load_state_dict(torch.load(backnone))
        test_model.classifier.load_state_dict(torch.load(classifier))

    test_path = root_dataset + '/videos/test'
    test_loader = DataLoader(VolleyBallDataSet(test_path, train_annot, preprocess=preprocess), batch_size=batch_size)

    test_acc = test_model.test_model(test_loader)
    print(test_acc)

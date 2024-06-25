import torch
import torch.nn as nn
import torch.optim as optim
import os
import requests
from PIL import Image
from io import BytesIO
from data_utils.dataloader import Load_data as Load_data
from data_utils.dataloader_cifar import Load_data as Load_data_cifar
from metric_utils.eval import evaluate, print_confusion_matrix
from model.custom_model import DAIGTModel, bertTF
from tqdm import tqdm


class Classify:
    def __init__(self, config):
        self.num_epochs = config['n_epochs']
        self.patience = config['patience']
        self.learning_rate = config['learning_rate']
        self.save_path = os.path.join(config['save_path'], config['model'])
        self.early_stop = config['early_stop']
        # self.dataloader = Load_data(config)
        if config['dataset'] == 'MNIST':
            self.dataloader = Load_data(config)
        elif config['dataset'] == 'CIFAR10':
            self.dataloader = Load_data_cifar(config)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # self.base_model = LeNet(config).to(self.device)
        self.base_model_name = config['model']
        if self.base_model_name == 'BERT':
            self.base_model = bertTF().to(self.device)
        elif self.base_model_name == 'daigtModel':
            self.base_model = DAIGTModel(config).to(self.device)
        self.optimizer = optim.SGD(
            lr=self.learning_rate, params=self.base_model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        train_loader, dev_loader = self.dataloader.load_train_dev()
        best_acc = 0
        patience = 0
        print(f'Number of epochs: {self.num_epochs}')
        print(f'Patience: {self.patience}')
        print(f'Learning rate: {self.learning_rate}')
        print(f'Early stop: {self.early_stop}')
        print(f'Base model: {self.base_model_name}')

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            self.base_model.train()
            # for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            # change tqdm progress bar
            for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", bar_format="{l_bar}%s{bar}%s{r_bar}",):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.base_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                accuracy = output.argmax(dim=1).eq(
                    target).sum().item() / len(data)
            print(
                f'Epoch: {epoch+1}/{self.num_epochs},Loss: {loss.item():.6f}, Accuracy: {accuracy*100:.2f}%')
            # validation
            self.base_model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in dev_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.base_model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            acc = correct / len(dev_loader.dataset)

            if acc > best_acc:
                best_acc = acc
                patience = 0
                torch.save(self.base_model.state_dict(),
                           os.path.join(self.save_path, 'best_model.pt'))
            else:
                patience += 1
                if patience > self.patience and self.early_stop and epoch > 50:
                    print("Early stopping")
                    break

        print(f'Best validation accuracy: {best_acc*100:.2f}%')

    def test(self):
        test_loader = self.dataloader.load_test()
        self.base_model.load_state_dict(torch.load(
            os.path.join(self.save_path, 'best_model.pt')))
        self.base_model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.base_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                acc, precision, rec, f1 = evaluate(output, target)
                print(
                    f'Accuracy: {acc}, Recall: {rec}, F1: {f1}')
                print_confusion_matrix(output, target)
                break
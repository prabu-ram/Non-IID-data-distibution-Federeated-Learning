# importing all neccesssary libraries.
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import numpy as np
import random
from Federated_base import Federated_base
from models import SimpleCNN, CNNCifar
from plot import display_class_distribution_heatmap, count_class_samples, plot_accuracies

DEVICE = torch.device("mps")  
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
class FederatedTraining:
    def __init__(self, model, device, trainset, testset, num_clients, iid=True, alpha=0.5):
        self.model = model
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.num_clients = num_clients
        self.iid = iid
        self.alpha = alpha

        self.federated = Federated_base(num_clients, trainset.data, trainset.targets, testset.data, testset.targets)
        self.clients = self.federated.create_clients_iid() if iid else self.federated.create_clients_dirichlet(alpha)
        self.client_datasets = {client: DatasetSplit(trainset, indices) for client, indices in self.clients.items()}
        self.test_loader = DataLoader(testset, batch_size=32, shuffle=False)  # Fixed batch size for testing

    def train(self, epochs, lr, batch_size):
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()

        self.client_loaders = {client: DataLoader(dataset, batch_size=batch_size, shuffle=True) 
                               for client, dataset in self.client_datasets.items()}
        
        accuracy_list = []
        
        for epoch in range(epochs):
            client_models = []
            for client, loader in self.client_loaders.items():
                client_model = SimpleCNN().to(self.device)
                client_model.load_state_dict(self.model.state_dict())
                optimizer = optim.SGD(client_model.parameters(), lr=lr)

                client_model.train()
                for images, labels in loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    output = client_model(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()

                client_models.append(client_model.state_dict())
            
            # Federated averaging
            new_state_dict = self.model.state_dict()
            for key in new_state_dict.keys():
                new_state_dict[key] = torch.stack([client_model[key].float() for client_model in client_models], 0).mean(0)
            self.model.load_state_dict(new_state_dict)

            # Evaluate the model
            test_loss, accuracy = self.evaluate()
            accuracy_list.append(accuracy)
            print(f'Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return accuracy_list

    def evaluate(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                test_loss += criterion(output, labels).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        return test_loss, accuracy
    
# Dataset Preparation
    
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

num_clients = 10
X_train = trainset.data
y_train = np.array(trainset.targets)
X_test = testset.data
y_test = np.array(testset.targets)

federated = Federated_base(num_clients, X_train, y_train, X_test, y_test)

# Create IID client datasets
clients_iid = federated.create_clients_iid()

# Optionally, create non-IID client datasets
clients_non_iid = federated.create_clients_dirichlet(alpha=0.5)

client_datasets_iid = {client: DatasetSplit(trainset, indices)
                       for client, indices in clients_iid.items()}

# Wrap non-IID client datasets using DatasetSplit
client_datasets_non_iid = {client: DatasetSplit(trainset, indices)
                           for client, indices in clients_non_iid.items()}

# Create DataLoaders for each client
client_loaders_iid = {client: DataLoader(dataset, batch_size=32, shuffle=True)
                      for client, dataset in client_datasets_iid.items()}

client_loaders_non_iid = {client: DataLoader(dataset, batch_size=32, shuffle=True)
                          for client, dataset in client_datasets_non_iid.items()}

num_classes = 10
client_class_counts_iid = count_class_samples(client_datasets_iid, num_classes)

# Count samples per class for non-IID datasets
client_class_counts_non_iid = count_class_samples(client_datasets_non_iid, num_classes)

# Display IID distribution
display_class_distribution_heatmap(client_class_counts_iid, num_classes, "IID Client Data Distribution")

# Display non-IID distribution
display_class_distribution_heatmap(client_class_counts_non_iid, num_classes, "Non-IID Client Data Distribution")

# Training for graphs 1
num_clients = 10
epochs = 100
lr = 0.01
batch_sizes = [16, 32, 64, 128]

accuracies = []
for batch_size in batch_sizes:
    model = SimpleCNN()
    
    print("For Batch size: ", batch_size)
    federated_training = FederatedTraining(model, DEVICE, trainset, testset, num_clients, iid=True, alpha=0.5)
    accuracy = federated_training.train(epochs, lr, batch_size)
    accuracies.append(accuracy)

plot_accuracies(batch_sizes, accuracies)

# Performance with reporting fraction and alpha

# num_clients = 50
# epochs = 200
# lr = 0.01
# batch_size = 32
# reporting_fraction = 0.5
# alphas = [100, 10, 1, 0.5, 0]

# all_accuracies = {}
# for alpha in alphas:
#     model = SimpleCNN()
#     federated_training = FederatedTraining(model, DEVICE, trainset, testset, num_clients, iid=True, alpha=alphas)
#     print(f"Training with alpha: {alpha}")
#     accuracies = federated_training.train(epochs, lr, batch_size, reporting_fraction)
#     all_accuracies[alpha] = accuracies

# plot_accuracies(alphas, all_accuracies)
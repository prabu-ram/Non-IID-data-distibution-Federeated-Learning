import random
import numpy as np
from dirchlet import dirichlet_distribution
from collections import defaultdict


class Federated_base():

    def __init__(self, num_clients, X_train, y_train, X_test, y_test):
        self.num_clients = num_clients
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def create_clients_iid(self, initial = "entity"):
        ''' 
        Creates an IID data distribution for clients.

        Returns:
            Dictionary with keys as client's name and values as tuples of image and label lists.
        '''


        # creates a list of all client names using the "initial"
        client_names = ['{}_{}'.format(initial, i+1) for i in range(self.num_clients)]

        data = list(zip(self.X_train, self.y_train))
        random.shuffle(data)

        size = len(data)//self.num_clients
        shards = [data[i:i + size] for i in range(0, size*self.num_clients, size)]

        #number of clients must equal number of shards
        assert(len(shards) == len(client_names))

        return {client_names[i] : shards[i] for i in range(len(client_names))} 


    def create_clients_dirichlet(self, alpha=0.5):
        ''' 
        Creates a non-IID data distribution using Dirichlet distribution for clients.

        Returns:
            Dictionary with keys as client's name and values as tuples of image and label lists.
        '''
        
        # Create the dictionary to store clients' data
        non_iid_clients = defaultdict(list)
        label_array = np.array(self.y_train)
        
        # Get unique labels and their indices
        unique_labels, label_indices = np.unique(label_array, return_inverse=True)
        
        # Number of classes
        num_classes = len(unique_labels)
        
        # Generate Dirichlet distribution for each class
        class_distribution = np.random.dirichlet([alpha] * self.num_clients, num_classes)
        
        # Assign data points to clients based on the Dirichlet distribution
        for class_idx, class_dist in enumerate(class_distribution):
            class_data_indices = np.where(label_indices == class_idx)[0]
            random.shuffle(class_data_indices)
            
            # Distribute data points to clients
            class_split = np.array_split(class_data_indices, np.cumsum(class_dist)[:-1])
            
            for client_idx, indices in enumerate(class_split):
                for idx in indices:
                    non_iid_clients[f'client_{client_idx}'].append((self.X_train[idx], self.y_train[idx]))
        
        for client in non_iid_clients:
            images, labels = zip(*non_iid_clients[client])
            non_iid_clients[client] = (list(images), list(labels))
        
        return non_iid_clients
    

# class LocalUpdate():
#     def __init__(self, Dataset) -> None:




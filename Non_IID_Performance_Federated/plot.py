import matplotlib.pyplot as plt
import seaborn as sns


def count_class_samples(client_datasets, num_classes):
    client_class_counts = {client: np.zeros(num_classes, dtype=int) for client in client_datasets.keys()}

    for client, dataset in client_datasets.items():
        for _, label in dataset:
            client_class_counts[client][label] += 1

    return client_class_counts

import pandas as pd

def display_class_distribution_heatmap(client_class_counts, num_classes, title):
    # Create a DataFrame from the client class counts
    df = pd.DataFrame(client_class_counts).T
    df.columns = [f'Class_{i}' for i in range(num_classes)]
    
    # Add a total column
    df['Total'] = df.sum(axis=1)
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt='d')
    plt.title(title)
    plt.show()

def plot_accuracies(batch_sizes, accuracies):
    for batch_size, accuracy in zip(batch_sizes, accuracies):
        plt.plot(range(1, len(accuracy)+1), accuracy, label=f'alpha = {batch_size}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Federated Learning Accuracy for Different alpha (Dirichlit)')
    plt.legend()
    plt.show()

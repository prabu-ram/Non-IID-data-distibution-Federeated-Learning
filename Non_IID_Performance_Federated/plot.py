import numpy as np
import matplotlib.pyplot as plt


def count_class_samples(client_datasets, num_classes):
    client_class_counts = {client: np.zeros(num_classes, dtype=int) for client in client_datasets.keys()}

    for client, dataset in client_datasets.items():
        for _, label in dataset:
            client_class_counts[client][label] += 1

    return client_class_counts

def plot_class_distribution(client_class_counts, num_classes, title):
    num_clients = len(client_class_counts)
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.35
    index = np.arange(num_classes)
    
    for i, (client, counts) in enumerate(client_class_counts.items()):
        ax.bar(index + i * bar_width, counts, bar_width, label=client)

    ax.set_xlabel('Class')
    ax.set_ylabel('Number of samples')
    ax.set_title(title)
    ax.set_xticks(index + bar_width * (num_clients - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(num_classes)])
    ax.legend()

    plt.show()

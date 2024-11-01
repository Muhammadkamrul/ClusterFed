import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, Subset
import client as client_obj
from server import Server, download_mnist, download_cifar10, download_fmnist
import csv
import matplotlib.pyplot as plt

random_seed=42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

#torch.set_default_dtype(torch.float64)  # For usign 64 bit floating-point tensors data type

# Configuration
NUM_CLUSTERS = 2
CLIENTS_PER_CLUSTER = [5,5]
SERVER_CAPACITY = 6
ROUNDS = 3000
learning_rate = 0.001
samples_per_client = 600
weight = 0.5

LOCAL_EPOCHS = 5
local_learning_rate = 0.01
K = 2
G = 5  # Example threshold for L2 norm
D = 10000 # Example deadline for transmission time (in seconds)
#QUANTIZATION_BITS = [31, 15, 7, 4, 2]  # Possible quantization levels
QUANTIZATION_BIT = 64
threshold = None
top_n = True
client_select_type = 'custom'
#client_select_type = 'random'

num_classes=10
global_total_samples = 0

def check_data_distribution(clusters):
    """
    Prints the data distribution (class-wise sample count) for each client.

    Args:
        clusters: A list of cluster objects, each containing a list of clients.

    Returns:
        None. Prints class-wise sample distribution for each client.
    """
    global global_total_samples

    for cluster in clusters:
        for client in cluster.clients:
            client_data = client.get_data()  # Assuming this returns a Dataset (like Subset or DataLoader)
            class_counts = {}
            client_total_samples = 0  # To track total samples for this client

            # Iterate over client data and count the occurrences of each class
            for _, target in client_data:
                class_counts[target] = class_counts.get(target, 0) + 1
                client_total_samples += 1

            # Print the class-wise distribution for this client
            print(f"Client {client.client_id}:")
            for class_id in sorted(class_counts.keys()):
                print(f"  Class {class_id}: {class_counts[class_id]} samples")

            print(f"Total samples for Client {client.client_id}: {client_total_samples} samples")
            print("-" * 20)

            global_total_samples += client_total_samples

    # Print the total number of samples across all clusters
    print(f"\nTotal samples across all clusters = {global_total_samples}")

def distribute_custom_skewed_data(clusters, train_dataset):
    """
    Distributes MNIST data to 10 clients. Each client gets 1000 samples, with a specific percentage 
    from a designated "main" class, and the rest randomly sampled from other classes excluding the main class.
    """

    # Flatten the list of clients into a single list
    client_list = []
    for cluster in clusters:
        client_list.extend(cluster.clients)

    assert len(client_list) == 10, "This function is designed for exactly 10 clients."

    # Get class-wise indices
    class_indices = [[] for _ in range(num_classes)]
    for idx, (data, target) in enumerate(train_dataset):
        class_indices[target].append(idx)

    # Shuffle class indices to randomize sample order for each class
    for indices in class_indices:
        np.random.shuffle(indices)

    # Distribute data to clients
    for client_id, client in enumerate(client_list):
        main_class_id = client_id  # Each client has a different main class

        # Calculate main class percentage and number of samples
        if client_id == 0:
            main_class_percentage = 100  # Client 0 gets 100% from main class
        else:
            main_class_percentage = client_id * 10  # 10%, 20%, ..., 90%

        main_class_samples = int((main_class_percentage / 100) * samples_per_client)
        other_class_samples = samples_per_client - main_class_samples

        # Assign samples from the main class
        if len(class_indices[main_class_id]) >= main_class_samples:
            client_samples = class_indices[main_class_id][:main_class_samples]
            class_indices[main_class_id] = class_indices[main_class_id][main_class_samples:]  # Remove assigned samples
        else:
            raise ValueError(f"Not enough samples in class {main_class_id} for client {client_id}")

        # Distribute remaining samples evenly across the other classes
        other_classes = [i for i in range(num_classes) if i != main_class_id]
        np.random.shuffle(other_classes)  # Shuffle other classes independently for each client

        remaining_samples_needed = other_class_samples
        samples_per_other_class = remaining_samples_needed // (num_classes - 1)  # Evenly split among other classes
        extra_samples = remaining_samples_needed % (num_classes - 1)  # Handle the remainder

        client_other_samples = []

        # Assign samples from other classes
        for other_class_id in other_classes:
            if len(class_indices[other_class_id]) > 0:
                num_samples = samples_per_other_class
                if extra_samples > 0:
                    num_samples += 1  # Distribute extra samples
                    extra_samples -= 1
                
                num_samples = min(num_samples, len(class_indices[other_class_id]))  # Limit to available samples
                client_other_samples.extend(class_indices[other_class_id][:num_samples])
                class_indices[other_class_id] = class_indices[other_class_id][num_samples:]  # Remove assigned samples

        # Combine main class samples and other class samples
        client_samples.extend(client_other_samples)
        np.random.shuffle(client_samples)  # Shuffle the combined samples

        # Assign the collected samples to the client
        client.set_data(Subset(train_dataset, client_samples))

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')
    return test_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clusters = client_obj.create_clusters(NUM_CLUSTERS, CLIENTS_PER_CLUSTER, K, device)
    server = Server(clusters, SERVER_CAPACITY, device, learning_rate, threshold = None, top_n = True)
    
    # Download and distribute the dataset
    train_dataset, test_dataset = download_mnist()
    #train_dataset, test_dataset = download_fmnist()
    #train_dataset, test_dataset = download_cifar10() 
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    distribute_custom_skewed_data(clusters, train_dataset)
    
    check_data_distribution(clusters)     
    
    # Initialize logging
    log_file = "loss_acc_rnd.csv"
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Number of Selected Clients", "Global Loss", "Training Loss", "Accuracy"])
    
    detail_log_file = "clnt_clstr_norm_enrgy_bits_time.csv"
    with open(detail_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Client ID", "Cluster ID", "Gradient Norm",  "local_loss", "hybrid_score", "Transmitted Energy", "Transmitted Bits", "Transmission Time"])
    
    metric_log_file = "all client_metric.csv"
    with open(metric_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Client ID", "Cluster ID", "metric", "Transmitted Energy", "Transmitted Bits", "Transmission Time"])

    loss_list = []
    training_loss_list = []
    accuracy_list = []
    total_transmitted_bits_list = []
    total_energy_list = []
    old_list = []
    
    num_clients = sum(len(cluster.clients) for cluster in clusters) 
    client_selection_tracker = {client_id: 0 for client_id in range(num_clients)}
    
    for round_num in range(ROUNDS):
        print(f"\n\nFederated Learning Round {round_num + 1} *********************")
        
        server.distribute_model()
        
        server.train_and_setMinMax()
        
        #server.compute_local_loss()
        
        selected_clients = server.select_clients(K, G, D, QUANTIZATION_BIT, client_select_type, round_num + 1, weight, global_total_samples, num_classes)

        if not selected_clients:
            print("No valid clusters were selected, restart training round.")         
            total_transmitted_bits_list.append(None)
            total_energy_list.append(None)          
            loss_list.append(None)
            training_loss_list.append(None)
            accuracy_list.append(None)           
            continue
        
        for client, _, _ in selected_clients:
            # Increment the count for the selected client
            client_selection_tracker[client.client_id] += 1
        
        current_list=[]
        print("\nSelected clients at round: ", round_num + 1)
        for client,_,_ in selected_clients:
            print(f"Client {client.client_id}")
            current_list.append(client.client_id)
        
        if (sorted(current_list) == sorted(old_list)):
            print("\nSelected the same set of clients as last round\n")
        
        old_list = current_list
        
        
        
        #client_obj.verify_clusters(clusters, selected_clients)
        #server.update_learning_rate(round_num)
        server.aggregate_quantized_grads()
        loss, accuracy = evaluate(server.global_model, device, test_loader)
        
        #if (round_num  == 0):
        #    continue
        
        total_training_loss = 0
        
        print("\nAll clients metric for round:", round_num + 1)
        for cluster in clusters:
            for client in cluster.clients:
                total_training_loss += client.local_loss
                print(f"Client {client.client_id}: Local Loss {client.local_loss}, L2 Norm {client.local_l2_norm}")
                
        
        total_clients = sum(len(cluster.clients) for cluster in clusters)        
        average_training_loss = total_training_loss / total_clients
        
        # Log the results
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round_num + 1, len(selected_clients), loss, average_training_loss, accuracy])

        # Detailed logging
        with open(detail_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for client, hybrid_score, _ in selected_clients:
                writer.writerow([round_num + 1, client.client_id, client.cluster_id, client.local_l2_norm, client.local_loss, hybrid_score, client.energy, client.transmitted_bits, client.transmission_time])
        
        #Metric logging
        with open(metric_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for cluster in clusters:
                for client in cluster.clients:
                    writer.writerow([round_num + 1, client.client_id, client.cluster_id, client.metric, client.energy, client.transmitted_bits, client.transmission_time])
        
        
        # Calculate total transmitted bits and energy for the round
        total_transmitted_bits = sum(client.transmitted_bits for client, _, _ in selected_clients)
        total_energy = sum(client.energy for client, _, _ in selected_clients)
        
        total_transmitted_bits_list.append(total_transmitted_bits)
        total_energy_list.append(round(total_energy,5))
        
        loss_list.append(round(loss,4))
        training_loss_list.append(round(average_training_loss,4))
        accuracy_list.append(round(accuracy,2))
    
    client_ids = list(client_selection_tracker.keys())
    selection_counts = list(client_selection_tracker.values())

    with open('output_'+client_select_type+'_sel_s4.txt', 'w') as file:
        # Write each list in the desired format
        file.write(f'loss_{client_select_type} = {loss_list}\n')
        file.write(f'accuracy_{client_select_type} = {accuracy_list}\n')
        file.write(f'bits_{client_select_type} = {total_transmitted_bits_list}\n')
        file.write(f'energy_{client_select_type} = {total_energy_list}\n')
        file.write(f'train_loss_{client_select_type} = {training_loss_list}\n')
        file.write(f'selected_client_id_{client_select_type} = {client_ids}\n')
        file.write(f'selected_client_count_{client_select_type} = {selection_counts}\n')



if __name__ == "__main__":
    main()



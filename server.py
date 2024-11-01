import torch
from torch import nn
from torch.utils.data import DataLoader
from client import Client, Cluster
import torch.nn.functional as F
import random
import collections

random_seed=42
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

"""
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # Increase the number of neurons in the first hidden layer
        self.fc2 = nn.Linear(256, 128)      # Add a second hidden layer
        self.fc3 = nn.Linear(128, 10)       # Output layer for 10 classes
        self.dropout = nn.Dropout(0.3)      # 30% dropout to prevent overfitting

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)          # Apply dropout after the first hidden layer
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNN_Cifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 84)
        self.fc4 = nn.Linear(84, 50)
        self.fc5 = nn.Linear(50, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return self.logsoftmax(x)
"""
class MLP_FMNIST(nn.Module):
    def __init__(self, dim_in=784, dim_hidden1=64, dim_hidden2=30, dim_out=10):
        super(MLP_FMNIST, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(dim_hidden1, dim_hidden2)
        self.layer_hidden2 = nn.Linear(dim_hidden2, dim_out)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)

        return self.logsoftmax(x)

class Server:
    def __init__(self, clusters, max_clients, device, learning_rate, threshold, top_n):
        self.max_clients = max_clients
        self.device = device
        self.global_model = MLP_FMNIST().to(self.device)
        self.clusters = clusters
        self.selected_clients = []
        self.threshold = threshold
        self.top_n = top_n
        self.learning_rate = learning_rate
        
    
    def distribute_model(self):
        
        print("\n\nInside Model Distribution")
        
        global_state_dict = self.global_model.state_dict()
        for cluster in self.clusters:
            for client in cluster.clients:
                client.set_model(global_state_dict)
                    
    def compute_local_loss(self):       
        print("\nComputing local loss...")
        for cluster in self.clusters:
            for client in cluster.clients:
                client.compute_local_loss()   
    
    def train_and_setMinMax(self):
        #min_val_list = []
        #max_val_list = []
        self.l2_norm_avg = []
        print("\nTraining ongoing...")
        for cluster in self.clusters:
            for client in cluster.clients:
                local_l2_norm = client.train()
                #total_loss, l2_norm = client.calculate_gradients() #The purpose of total_loss is tracking the loss value during training for each client.
                self.l2_norm_avg.append(local_l2_norm)
                #min_val, max_val = client.find_min_max_grads()
                #min_val_list.append(min_val)
                #max_val_list.append(max_val)
                
        #self.global_min_val = min(min_val_list)
        #self.global_max_val = max(max_val_list)
        print("Training Done!\n")
    
    def select_clients(self, K, G, D, quantization_bit, client_select_type, FL_round, weight, global_total_samples, num_classes):
        
        print("\n\nInside Client selection")
        
        self.quantization_bit = quantization_bit
        valid_clusters = []
        deadline_avg = []
        original_payload_sizes = []
        quantized_payload_sizes = []
        packed_payload_sizes = []
        tranmission_time_avg = []
        
        self.selected_clients.clear()     
        
        if (client_select_type == 'random'):          

            valid_clusters = []  # To store clusters with valid clients, for random, assume all clients valid

            for cluster in self.clusters:
                valid_clients = []
                for client in cluster.clients:                                               
                    orig_payload_size_bits = client.calculate_payload_size()
                    original_payload_sizes.append(orig_payload_size_bits/8)
                    transmission_time = client.calculate_transmission_time(orig_payload_size_bits)                                                        
                    tranmission_time_avg.append(transmission_time)
                    client.calculate_energy_consumption(transmission_time)
                    valid_clients.append((client, None, self.quantization_bit))
                valid_clusters.append((cluster, valid_clients))
                    
            print("\nRandom Client Selection, using Proportional Allocation")
            #print("Average l2_norm: ",  sum(self.l2_norm_avg)/len(self.l2_norm_avg))                     
            print("Average transmission time: ", sum(tranmission_time_avg)/len(tranmission_time_avg))   
            print("Max payload size(original): ", max(original_payload_sizes), " bytes", " Min payload size(original): ", min(original_payload_sizes), " bytes") 
            
            # Step 2: Randomly select a larger subset of clients from the valid clusters       
            X = K  # Example: Select top K clients from each cluster

            # Step 3: Iterate through each cluster and select the top X clients based on their scores
            for cluster, valid_clients in valid_clusters:
                # Step 3.1: Sort clients within the cluster by their hybrid score (descending order)
                #valid_clients.sort(key=lambda client_tuple: client_tuple[1], reverse=True)
                
                # Step 3.2: Select the top X clients from this cluster
                rand_clients = random.sample(valid_clients, X)
                
                # Step 3.3: Add these clients to the final selection list
                self.selected_clients.extend(rand_clients)

            # Step 4: At this point, self.selected_clients will have the top X clients from each cluster
            # No need to apply privacy constraints as stated in the requirement

            # Step 5: Print the number of clients selected from each cluster for verification (optional)
            selected_clients_by_cluster = collections.defaultdict(list)
            for client, hybrid_score, quantization_bit in self.selected_clients:
                selected_clients_by_cluster[client.cluster_id].append((client, hybrid_score, quantization_bit))

            # Output the result
            for cluster_id, selected_clients in selected_clients_by_cluster.items():
                print(f"Cluster {cluster_id}: Selected {len(selected_clients)} clients")                                            

            return self.selected_clients                                
            
        #else                         

        valid_clusters = []  # To store clusters with valid clients

        # Step 1: Collect valid clients per cluster
        for cluster in self.clusters:
            valid_clients = []
            for client in cluster.clients:                    
                orig_payload_size_bits = client.calculate_payload_size()
                original_payload_sizes.append(orig_payload_size_bits / 8)

                if self.quantization_bit < 64:
                    quantized_gradients = client.quantize_gradients(self.global_min_val, self.global_max_val, self.quantization_bit)
                    quantized_payload_size_bits = sum(g.numel() * g.element_size() * 8 for g in quantized_gradients)
                    quantized_payload_sizes.append(quantized_payload_size_bits / 8)
                    
                    packed_data = client.pack_quantized_grads(self.quantization_bit)
                    packed_payload_sizes.append(len(packed_data))
                    transmission_time = client.calculate_transmission_time(len(packed_data) * 8)
                else:
                    transmission_time = client.calculate_transmission_time(orig_payload_size_bits)

                deadline_avg.append(transmission_time)
                client.calculate_energy_consumption(transmission_time)

                if transmission_time <= D:
                    hybrid_score = (weight * client.local_loss) + ((1 - weight) * client.local_l2_norm * (client.get_num_samples()/global_total_samples) * (client.get_num_classes()/num_classes))
                    client.metric = hybrid_score
                    valid_clients.append((client, hybrid_score, self.quantization_bit))

            # Only consider clusters that have at least K valid clients
            if len(valid_clients) >= K:
                valid_clusters.append((cluster, valid_clients))

        # If no valid clusters were found
        if len(valid_clusters) == 0:
            return None  # Restart round if no valid clusters

        # Step 2: Randomly select a larger subset of clients from the valid clusters       
        X = K  # Example: Select top K clients from each cluster

        # Step 3: Iterate through each cluster and select the top X clients based on their scores
        for cluster, valid_clients in valid_clusters:
            # Step 3.1: Sort clients within the cluster by their hybrid score (descending order)
            valid_clients.sort(key=lambda client_tuple: client_tuple[1], reverse=True)
            
            # Step 3.2: Select the top X clients from this cluster
            top_clients = valid_clients[:X]
            
            # Step 3.3: Add these clients to the final selection list
            self.selected_clients.extend(top_clients)

        # Step 4: At this point, self.selected_clients will have the top X clients from each cluster
        # No need to apply privacy constraints as stated in the requirement

        # Step 5: Print the number of clients selected from each cluster for verification (optional)
        selected_clients_by_cluster = collections.defaultdict(list)
        for client, hybrid_score, quantization_bit in self.selected_clients:
            selected_clients_by_cluster[client.cluster_id].append((client, hybrid_score, quantization_bit))

        # Output the result
        for cluster_id, selected_clients in selected_clients_by_cluster.items():
            print(f"Cluster {cluster_id}: Selected {len(selected_clients)} clients")        
        
        # Final logging and statistics
        print("\nQuantization using: ", self.quantization_bit, " bits")
        print("Average l2_norm: ", sum(self.l2_norm_avg) / len(self.l2_norm_avg), "Required G: ", G)
        print("Max payload size(original): ", max(original_payload_sizes), " bytes", "Min payload size(original): ", min(original_payload_sizes), " bytes")

        if self.quantization_bit < 64:
            print("Max payload size (quantized): ", max(quantized_payload_sizes), " bytes,", "Min payload size (quantized): ", min(quantized_payload_sizes), " bytes")
            print("Max payload size (packed): ", max(packed_payload_sizes), " bytes", "Min payload size (packed): ", min(packed_payload_sizes), " bytes")

        print("Average transmission time: ", sum(deadline_avg) / len(deadline_avg), "Required D: ", D)
        
        return self.selected_clients        


    def aggregate_quantized_grads(self):                
        print("\n\nInside aggregate_quantized_grads")      
        
        total_gradients = None
        total_samples = 0  # Total number of samples across all selected clients
        
        # Iterate over each selected client
        for client, _, _ in self.selected_clients: 

            if(self.quantization_bit <64):                            
                                                 
                    unpacked_quantized_grads = self.unpack_quantized_grads(client.packed_bytes, client.num_elements_in_Qgrads, client.Qgrads_shapes, self.quantization_bit, client.dtype)
                    
                    #self.verify_unpacking(client.quantized_gradients, unpacked_quantized_grads)
                    #self.verify_decoding(client.quantized_gradients, , global_min_val, global_max_val, self.quantization_bit)
                                        
                    #verification_result = self.verify_quantized_gradients(client.quantized_gradients, unpacked_quantized_grads, self.quantization_bit)
                    #if(verification_result==False):
                        #input("\nverify_quantized_gradients returned false")
                    
                    client_gradients = unpacked_quantized_grads
                    #client_gradients = client.quantized_gradients
                    
            else:                            
                    client_gradients = client.gradients
            
            num_samples = client.get_num_samples()
            total_samples += num_samples
            
            if total_gradients is None:
                total_gradients = [torch.zeros_like(grad) for grad in client_gradients]
            for i, grad in enumerate(client_gradients):
                total_gradients[i] += grad * num_samples               
        
        # Average gradients
        for i in range(len(total_gradients)):
            total_gradients[i] = total_gradients[i] / total_samples
        
        self.global_gradients = total_gradients
        
        if(self.quantization_bit <64):
            self.global_gradients = self.decode_quantized_gradients(self.global_gradients, self.quantization_bit)
        
        with torch.no_grad():
            for param, grad in zip(self.global_model.parameters(), self.global_gradients):            
                param -= self.learning_rate * grad  # Update model parameters with learning rate


    def unpack_quantized_grads(self, packed_bytes, num_elements, grad_shapes, bits, dtype):
        print("\n\nInside Unpacking")
        
        num_bits_per_value = bits + 1  # Including sign bit
        total_bits = len(packed_bytes) * 8
        packed_bits = []
        for byte in packed_bytes:
            packed_bits.extend([int(bit) for bit in bin(byte)[2:].zfill(8)])

        unpacked_values = []
        bit_index = 0

        for _ in range(num_elements):
            if bit_index + num_bits_per_value > total_bits:
                raise ValueError(f"Not enough bits left to unpack {num_bits_per_value}-bit value. Bit index: {bit_index}, total bits: {total_bits}")
            
            bits_for_value = packed_bits[bit_index:bit_index + num_bits_per_value]
            combined_value = int(''.join(map(str, bits_for_value)), 2)
            sign_bit = combined_value >> bits
            magnitude = combined_value & ((1 << bits) - 1)
            value = -magnitude if sign_bit else magnitude
            unpacked_values.append(value)
            bit_index += num_bits_per_value

        #Reshape the unpacked values back to their original tensor shapes
        unpacked_grads = []
        value_index = 0
        for shape in grad_shapes:
            num_elements = torch.tensor([], dtype=dtype).new_zeros(shape).numel()
            tensor_values = unpacked_values[value_index:value_index + num_elements]
            unpacked_grads.append(torch.tensor(tensor_values, dtype=dtype).reshape(shape))
            value_index += num_elements

        return unpacked_grads

    
    def decode_quantized_gradients(self, quantized_grads, bits=2):           
        print("\n\nInside decode_quantized_gradients")
        
        decoded_gradients = []
        num_levels = 2 ** bits
        step_size = (self.global_max_val - self.global_min_val) / (num_levels - 1)
        
        for grad in quantized_grads:
            grad = grad.to(self.device)  # Move data to GPU
            decoded_grad = (grad * step_size).to(self.device) + self.global_min_val
            decoded_gradients.append(decoded_grad)
        
        return decoded_gradients

    def update_learning_rate(self, current_round):
            if current_round > 0 and current_round % self.decay_rounds == 0:
                self.learning_rate *= self.decay_factor
                #print(f"Learning rate adjusted to: {self.learning_rate}")

    

    def verify_quantized_gradients2(self, quantized_grads, unpacked_quantized_grads, bits):
        
        print("\n\nInside verify_quantized_gradients")
        #print("Verifying quantized parameters...")
        
        if len(quantized_grads) != len(unpacked_quantized_grads):
            print(f"Mismatch in number of tensors: original({len(quantized_grads)}), unpacked({len(unpacked_quantized_grads)})")
            return False

        for i, (quant_grad, unpacked_grad) in enumerate(zip(quantized_grads, unpacked_quantized_grads)):
            #print(f"Verifying tensor {i+1}/{len(quantized_grads)}:")
            
            # Move tensors to the same device before comparison
            device = quant_grad.device
            unpacked_grad = unpacked_grad.to(device)
            
            # Check shape
            shape_match = quant_grad.shape == unpacked_grad.shape
            print(f"  Shape matches: {quant_grad.shape}" if shape_match else f"  Shape mismatch: original({quant_grad.shape}), unpacked({unpacked_grad.shape})")

            # Check dtype
            dtype_match = quant_grad.dtype == unpacked_grad.dtype
            print(f"  Dtype matches: {quant_grad.dtype}" if dtype_match else f"  Dtype mismatch: original({quant_grad.dtype}), unpacked({unpacked_grad.dtype})")

            if not shape_match or not dtype_match:
                return False

            if bits < 32:  # Only mask if less than 32 bits
                mask = (1 << (bits + 1)) - 1
                #mask = (1 << bits) - 1
                orig_grad_int_masked = quant_grad.int() & mask
                unpacked_grad_int = unpacked_grad.int()
            else:
                orig_grad_int_masked = quant_grad.int()
                unpacked_grad_int = unpacked_grad.int()

            # Masking to ensure that only the relevant bits are compared
            # mask = (1 << (bits + 1)) - 1
            # orig_param_int_masked = orig_param.int() & mask
            # unpacked_param_int = unpacked_param.int()

            # Check value equality
            value_match = torch.equal(orig_grad_int_masked, unpacked_grad_int)
            print(f"  Values match: {value_match}")
            
            if not value_match:
                print(f"  Value mismatch at tensor {i+1}")
                return False

        #print("Verification complete. All parameters match.")
        return True
    
    def verify_quantized_gradients(self, quantized_grads, unpacked_grads, bits):
        print("\n\nInside verify_quantized_gradients")
        
        if len(quantized_grads) != len(unpacked_grads):
            print(f"Mismatch in number of tensors: original({len(quantized_grads)}), unpacked({len(unpacked_grads)})")
            return False

        for i, (quant_grad, unpacked_grad) in enumerate(zip(quantized_grads, unpacked_grads)):
            # Ensure gradients are on the same device
            device = quant_grad.device
            unpacked_grad = unpacked_grad.to(device)
            
            # Check shape and dtype
            shape_match = quant_grad.shape == unpacked_grad.shape
            dtype_match = quant_grad.dtype == unpacked_grad.dtype
            if not shape_match or not dtype_match:
                print(f"Shape or dtype mismatch: original({quant_grad.shape}, {quant_grad.dtype}), unpacked({unpacked_grad.shape}, {unpacked_grad.dtype})")
                return False

            # Compare values directly (consider the step size)
            orig_grad_rounded = torch.round((quant_grad - self.global_min_val) / ((self.global_max_val - self.global_min_val) / ((2 ** bits) - 1)))
            value_match = torch.equal(orig_grad_rounded, unpacked_grad)
            
            if not value_match:
                print(f"Value mismatch at tensor {i+1}")
                print(f"Original grad rounded: {orig_grad_rounded}")
                print(f"Unpacked grad: {unpacked_grad}")
                return False
        
        print("Verification complete. All parameters match.")
        return True
    

    def verify_unpacking(self, original_quantized_grads, unpacked_grads):
        print("\n\nInside verify_unpacking")
        
        # Check if the number of tensors matches
        if len(original_quantized_grads) != len(unpacked_grads):
            print(f"Mismatch in number of tensors: original({len(original_quantized_grads)}), unpacked({len(unpacked_grads)})")
            return False

        for i, (original, unpacked) in enumerate(zip(original_quantized_grads, unpacked_grads)):
            # Check shape
            
            if original.shape != unpacked.shape:
                print(f"Shape mismatch at tensor {i}: original({original.shape}), unpacked({unpacked.shape})")
                return False
            original = original.to(self.device)
            unpacked = unpacked.to(self.device)
            # Check value equality
            if not torch.equal(original, unpacked):
                print(f"Value mismatch at tensor {i}")
                print(f"Original: {original.flatten()[:10]}")
                print(f"Unpacked: {unpacked.flatten()[:10]}")
                return False
        
        print("Unpacking verification complete. All values match.")
        return True
        
        
    def verify_decoding(self, original_grads, decoded_grads, global_min_val, global_max_val, bits=2):
        print("\n\nInside verify_decoding")
        
        num_levels = 2 ** bits
        step_size = (global_max_val - global_min_val) / (num_levels - 1)

        # Check if the number of tensors matches
        if len(original_grads) != len(decoded_grads):
            print(f"Mismatch in number of tensors: original({len(original_grads)}), decoded({len(decoded_grads)})")
            return False
        
        for i, (original, decoded) in enumerate(zip(original_grads, decoded_grads)):
            # Reconstruct the quantized values
            expected_decoded = (original * step_size).to(self.device) + global_min_val
            
            # Check shape
            if expected_decoded.shape != decoded.shape:
                print(f"Shape mismatch at tensor {i}: expected({expected_decoded.shape}), decoded({decoded.shape})")
                return False

            # Check value equality
            if not torch.allclose(expected_decoded, decoded, atol=1e-5):
                print(f"Value mismatch at tensor {i}")
                print(f"Expected: {expected_decoded.flatten()[:10]}")
                print(f"Decoded: {decoded.flatten()[:10]}")
                return False
        
        print("Decoding verification complete. All values match.")
        return True
        


def download_mnist():
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    return train_dataset, test_dataset
"""    
def download_cifar10():
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('../data', train=False, transform=transform)
    return train_dataset, test_dataset
"""    
def download_cifar10():
    from torchvision import datasets, transforms
    transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
    
    transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    test_dataset = datasets.CIFAR10('../data', train=False, transform=transform_test)
    return train_dataset, test_dataset
    

"""
def download_fmnist():
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])   
    # Download FMNIST dataset
    train_dataset = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('../data', train=False, transform=transform)   
    return train_dataset, test_dataset
"""

def download_fmnist():
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  
    # Download FMNIST dataset
    train_dataset = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('../data', train=False, transform=transform)   
    return train_dataset, test_dataset
    
                
                
def aggregate_quantized_grads_chat(self):                  
        
        total_gradients = None
        total_samples = 0  # Total number of samples across all selected clients
        
        # Iterate over each selected client
        for client, _, _ in self.selected_clients: 
                          
            client_gradients = client.gradients
            num_samples = client.get_num_samples()
            total_samples += num_samples
            
            if total_gradients is None:
                total_gradients = [torch.zeros_like(grad) for grad in client_gradients]
            for i, grad in enumerate(client_gradients):
                total_gradients[i] += grad * num_samples               
        
        # Average gradients
        for i in range(len(total_gradients)):
            total_gradients[i] = total_gradients[i] / total_samples
        
        self.global_gradients = total_gradients
 
        with torch.no_grad():
            for param, grad in zip(self.global_model.parameters(), self.global_gradients):            
                param -= self.learning_rate * grad             

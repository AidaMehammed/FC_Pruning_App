import torch

import torch.nn as nn
import torch.optim as optim

import fc_pruning.Compress.utils as pf


def set_weights(model, weights):
  for param, weight in zip(model.parameters(), weights):
        param.data = weight


def finetune(model, train_loader,epochs,learning_rate):
    print('Fine-tuning model...')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / total

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


def train(model, train_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0


        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = out.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {100 * train_accuracy:.2f}%, '
              )
    print('Finished Training')


def update_mask(model, data_with_mask):
    flattened_params_ref = [tensor.flatten() for tensor in pf.get_weights(model)]
    model_flat = torch.cat(flattened_params_ref, dim=0)

    data_with_mask = data_with_mask.coalesce()
    mask_indices_prep = data_with_mask.indices().numpy()
    mask_indices_list = mask_indices_prep.tolist()

    max_index = max(mask_indices_list[0]) + 1


    updated_mask_values = []
    for idx, val in enumerate(model_flat):
        updated_mask_values.append(val.item())

    updated_mask_values = torch.tensor(updated_mask_values, dtype=torch.float32)

    updated_mask_sparse = torch.sparse_coo_tensor(mask_indices_list, updated_mask_values, size=(max_index,))

    return updated_mask_sparse


def average_weights(reconstructed_clients):
    num_clients = len(reconstructed_clients)
    num_weights = len(reconstructed_clients[0])

    summed_weights = []
    for i in range(num_weights):
        total_weight = sum(weights[i] for weights in reconstructed_clients)
        summed_weights.append(total_weight)

    # Calculate the average for each position
    averaged_weights = []
    for weight_sum in summed_weights:
        averaged_weight = weight_sum / num_clients
        averaged_weights.append(averaged_weight)

    print('Data has been averaged')

    return averaged_weights

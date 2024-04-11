import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
import time
import pickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from SiameseNet import SiameseResNet

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


# train_max_index should be defined by counting files in the directory, but for now it should be fine
def train_model(model, num_epochs, train_max_index, path_train, weight_n, lr, path_test, test_max_index=37):
    weight = tensor([weight_n]).to(device)
    criterion = nn.BCELoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_start = time.time()
        loading_time_start = time.time()
        cum_loss = 0.

        for i in range(train_max_index + 1):
            with open(path_train + str(i) + ".pkl", 'rb') as inp:
                data = pickle.load(inp)
            if i % 10 == 9:
                print("loading time : " + str(time.time() - loading_time_start))
            mini_epoch_start = time.time()
            img1, img2, labels = data
            reshaped_labels = torch.reshape(labels, (labels.shape[0], 1)).to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            value_before = float(list(model.parameters())[-1][0])
            # Forward pass
            outputs = model(img1, img2)
            # Calculate loss
            loss = criterion(outputs, reshaped_labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            value_after = float(list(model.parameters())[-1][0])
            if abs(value_before - value_after) < 0.000000000001:
                print("VALUES ARE THE SAME")

            # Print statistics
            running_loss += loss.item()
            cum_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print('%f per batch : [%d, %5d] loss: %.3f' %
                      (time.time() - mini_epoch_start, epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
            loading_time_start = time.time()
        model.eval()
        loss_train, loss_test, acc_train, acc_test = evaluate_model_prep(model, criterion, True, train_max_index,
                                                                         test_max_index, path_train, path_test)
        print('\n loss train: %.3f; acc train %.3f \n loss test %.3f acc test %.3f' %
              (loss_train, acc_train * 100, loss_test, acc_test * 100))
        model.train()
        epoch_end = time.time()
        print("EPOCH TIME : " + str(epoch_end - epoch_start))
        print("Cumulative loss: %.3f" % cum_loss)
    print('Finished Training')
    return model


def get_predictions_and_metrics(model, threshold=0.5):
    correct_labels = []
    predict_labels = []
    model.eval()
    path = "processed_eval/"
    max_index = 37
    with torch.no_grad():
        for i in range(max_index + 1):
            with open(path + str(i) + ".pkl", 'rb') as inp:
                data = pickle.load(inp)
            img1, img2, labels = data
            correct_labels = correct_labels + labels.tolist()
            torch.reshape(labels, (labels.shape[0], 1)).to(device)

            # Forward pass
            outputs = model(img1, img2)
            # Calculate loss
            predictions = (outputs > threshold).float()
            predict_labels = predict_labels + predictions.tolist()
    print(confusion_matrix(correct_labels, predict_labels))
    print('precision %.3f, recall %.3f, f1 %.3f' %
          (precision_score(correct_labels, predict_labels), recall_score(correct_labels, predict_labels),
           f1_score(correct_labels, predict_labels)))
    return predict_labels, correct_labels


def evaluate_model_prep(model, criterion=nn.BCELoss(weight=tensor([13])), disable_train=False, train_max_index=1,
                        test_max_index=1, path_train="output/processed_dataset/", path_test="output/processed_eval/"):

    criterion_upd = criterion.to(device)
    loss1 = 0.
    acc1 = 0.
    if not disable_train:
        loss1, acc1 = model_eval_prep(model, path_train, train_max_index, criterion_upd)
    loss2, acc2 = model_eval_prep(model, path_test, test_max_index, criterion_upd)
    return loss1, loss2, acc1, acc2


def model_eval_prep(model, path, max_index, criterion):
    total_correct = 0.
    total_samples = 0.
    total_loss = 0.
    with torch.no_grad():
        for i in range(max_index + 1):
            with open(path + str(i) + ".pkl", 'rb') as inp:
                data = pickle.load(inp)
            img1, img2, labels = data
            reshaped_labels = torch.reshape(labels, (labels.shape[0], 1)).to(device)
            # Forward pass
            outputs = model(img1, img2)
            # Calculate loss
            loss = criterion(outputs, reshaped_labels)

            predictions = (outputs > 0.5).float()
            correct = (predictions == reshaped_labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            total_loss += loss
    return total_loss / total_samples, float(total_correct) / total_samples


if __name__ == "__main__":
    model = SiameseResNet()
    model = train_model(model, 2, 1359, "output/processed_dataset_augmented/", 0.1, 0.00001, "output/processed_eval/", 37)
    model = train_model(model, 2, 1359, "output/processed_dataset_augmented/", 0.1, 0.0000001, "output/processed_eval/", 37)
    torch.save(model.state_dict(), "output/siamese_model.pt")

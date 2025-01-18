from datetime import datetime

import torch
from torch import nn
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict

def train_model(model, train_loader, val_loader, epochs=25, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    train_accs = np.zeros(epochs)
    val_accs = np.zeros(epochs)

    for epoch in range(epochs) :
        train_loss = []
        val_loss = []
        n_correct = 0
        n_total = 0

        model.train()
        t0 = datetime.now()
        for images, labels in train_loader :
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            y_pred = model(images)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
            _, prediction = torch.max(y_pred, 1)
            n_correct += (prediction==labels).sum().item()
            n_total += labels.shape[0]

        train_loss = np.mean(train_loss)
        train_losses[epoch] = train_loss
        train_accs[epoch] = n_correct / n_total
        
        model.eval()
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            y_pred = model(images)
            loss = criterion(y_pred, labels)

            val_loss.append(loss.item())
            _, prediction = torch.max(y_pred, 1)
            n_correct += (prediction==labels).sum().item()
            n_total += labels.shape[0]

        val_loss = np.mean(val_loss)
        val_losses[epoch] = val_loss
        val_accs[epoch] = n_correct / n_total
        dt = datetime.now() - t0
        print(f'Epoch [{epoch+1}/{epochs}] -> Train Loss:{train_loss:.4f}, Train Acc:{train_accs[epoch]:.4f}| Val Loss:{val_loss:.4f}, Val Acc:{val_accs[epoch]:.4f}| Duration : {dt}')
    metrics = {"train_accs": train_accs,
               "train_losses": train_losses,
               "val_accs": val_accs,
               "val_losses": val_losses}
    return model, metrics

def plot_training_results(train_accs, train_losses, val_accs, val_losses, title="Training results"):

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # Left plots
    ax[0].plot(train_accs, label='Train Accuracy')
    ax[0].plot(val_accs, label='Validation Accuracy')
    ax[0].set_title('Accuracy')
    ax[0].legend()
    ax[0].grid()

    # Right plots
    ax[1].plot(train_losses, label='Train Loss')
    ax[1].plot(val_losses, label='Validation Loss')
    ax[1].set_title('Losses')
    ax[1].legend()
    ax[1].grid()
    plt.show()

def get_test_predicts(model, test_loader, device="cpu"):
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            _, preds = torch.max(preds, 1)
            y_pred.extend(preds.detach().cpu())
            y_true.extend(labels.detach().cpu())
            
    return y_true, y_pred

def get_test_metrics(model, model_name, test_loader, device="cpu", idx_to_class=None):
    y_true, y_pred = get_test_predicts(model, test_loader, device=device)
    
    if idx_to_class == None:
        display_labels = list(idx_to_class.values())
    else:
        display_labels = None
        
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=display_labels)
    disp.plot()
    plt.title(f"{model_name} confusion matrix")
    print(f"Accuracy {accuracy_score(y_true, y_pred)}")
    
def get_top5_similarity(emb, embeddings):
    emb = emb.reshape(1, -1)
    similarities = cosine_similarity(emb, embeddings)
    results = OrderedDict()
    for i in np.argsort(similarities)[0,:-6:-1]:
        results[i] = similarities[0,i]
    return results
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_metrics(dir_path, train_loss, train_acc, val_loss, val_acc, name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0][0].plot(train_loss, label='Train Loss')
    axes[0][0].set_xlabel('Epochs')
    axes[0][0].set_title('Train Loss')
    axes[0][0].grid(True)
    axes[0][0].legend()

    axes[0][1].plot(train_acc, label='Train Accuracy', color='orange')
    axes[0][1].set_xlabel('Epochs')
    axes[0][1].set_title('Train Accuracy')
    axes[0][1].grid(True)
    axes[0][1].legend()

    axes[1][0].plot(val_loss, label='Validation Loss')
    axes[1][0].set_xlabel('Epochs')
    axes[1][0].set_title('Validation Loss')
    axes[1][0].grid(True)
    axes[1][0].legend()

    axes[1][1].plot(val_acc, label='Validation Accuracy', color='orange')
    axes[1][1].set_xlabel('Epochs')
    axes[1][1].set_title('Validation Accuracy')
    axes[1][1].grid(True)
    axes[1][1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(dir_path, f'{name}.png'))
    plt.close(fig)

def plot_folds(dir_path, fold_loss, fold_acc):
    plt.figure()
    plt.plot(fold_loss, label='Validation Loss')
    plt.plot(fold_acc, label='Validation Accuracy')
    plt.xlabel('Folds')
    plt.legend()
    plt.title('Folds Performance')
    plt.grid()
    plt.savefig(os.path.join(dir_path, 'folds.png'))
    plt.close()

def plot_confusion(dir_path, cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(dir_path, 'confusion.png'))
    plt.close()

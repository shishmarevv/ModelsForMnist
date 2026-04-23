import os
import sys
import torch
import time
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
    
from dataset import get_loaders, get_cross_validation_loaders
from model import SimpleMLP, SimpleCNN, Architecture
from training import train, evaluate, get_metrics
from predict import predict_samples
import argparse
import draw

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP with Cross-Validation")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[32], help="List of hidden layer sizes")
    parser.add_argument("--dropout", type=float, nargs='+', default=[0.0], help="List of dropout rates for each hidden layer")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("-o", "--output_path", type=str, default="results", help="Path to save results")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--model", type=str, choices=["mlp", "cnn"], default="mlp", help="Model architecture to use")
    parser.add_argument("-m", "--mode", type=str, choices=['cross-validation', 'retries'], default='cross-validation', help="Mode of operation: 'cross-validation' or 'retries'")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.mode == 'cross-validation':
        folds(args, device)
    elif args.mode == 'retries':
        retries(args, device)

def folds(args, device):
    output_path = os.path.join(os.path.dirname(__file__),'output', args.output_path)
    os.makedirs(output_path, exist_ok=True)
    
    val_loss = []
    val_acc = []

    best_val_acc = 0.0

    best_train_losses = []
    best_train_accuracies = []
    best_validation_losses = []
    best_validation_accuracies = []

    all_train_losses = []
    all_train_accuracies =[]
    all_validation_losses = []
    all_validation_accuracies = []

    arch = None
    model = None
    best_arch = None
    best_model = None

    test_loader = None
    best_speed = float('inf')


    
    for fold, (train_loader, val_loader, test_loader) in enumerate(get_cross_validation_loaders( batch_size=args.batch_size)):
        print(f"\n=== Fold {fold+1} ===")
        if args.model == "mlp":
            arch = Architecture(
            input_dim=784, 
            hidden_dims=args.hidden_dims, 
            output_dim=10,
            dropout=args.dropout
            )
            model = SimpleMLP(arch).to(device)
        elif args.model == "cnn":
            arch = Architecture(
                input_dim=1, 
                hidden_dims=args.hidden_dims, 
                output_dim=10,
                dropout=args.dropout
                )
            model = SimpleCNN(arch).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        train_losses = []
        val_losses = []

        train_accuracies = []
        val_accuracies = []

        start = time.time()
        for t_loss, t_acc, v_loss, v_acc in train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=args.epochs):
            train_losses.append(t_loss)
            val_losses.append(v_loss)

            train_accuracies.append(t_acc)
            val_accuracies.append(v_acc)
        best_speed = min(best_speed, time.time() - start)
        
        val_loss.append(val_losses[-1])
        val_acc.append(val_accuracies[-1])

        all_train_losses = all_train_losses + train_losses
        all_train_accuracies = all_train_accuracies + train_accuracies
        all_validation_losses = all_validation_losses + val_losses
        all_validation_accuracies = all_validation_accuracies + val_accuracies

        if val_accuracies[-1] > best_val_acc:
            best_val_acc = val_accuracies[-1]
            best_train_losses = train_losses
            best_train_accuracies = train_accuracies
            best_validation_losses = val_losses
            best_validation_accuracies = val_accuracies

            best_arch = arch
            best_model = model

    val_loss_min = min(val_loss)
    val_acc_min = min(val_acc)
    val_loss_max = max(val_loss)
    val_acc_max = max(val_acc)
    val_loss_avg = sum(val_loss) / len(val_loss)
    val_acc_avg = sum(val_acc) / len(val_acc)

    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device)

    test_acc, conf_matrix, sensitivity, specificity = get_metrics(best_model, test_loader, device)

    print(f"Validation Loss - Min: {val_loss_min}, Max: {val_loss_max}, Avg: {val_loss_avg}")
    print(f"Validation Accuracy - Min: {val_acc_min}, Max: {val_acc_max}, Avg: {val_acc_avg}")
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Sensitivity: {sensitivity}, Specificity: {specificity}")

    output_file = os.path.join(output_path, "result.txt")
    with open(output_file, "w") as file:
        file.write("=== Training Configuration ===\n")
        headers = ["Device", "Epochs", "Learning Rate", "Batch Size"]
        rows = [
            [str(device), args.epochs, args.lr, args.batch_size]
        ]
        file.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")


        file.write("\n=== Model Architecture ===\n")
        headers = ["Input Dim"] + [f"Hidden Dim {i}/Dropout {i}" for i in range(len(args.hidden_dims))] + ["Output Dim"]
        rows = [
            [str(arch.input_dim)] + [f"{dim}/{dropout}" for dim, dropout in zip(arch.hidden_dims, arch.dropout)] + [str(arch.output_dim)]
        ]
        file.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")


        file.write("\n=== Results ===\n")
        headers = ["Validation Loss Min", "Validation Accuracy Min", "Validation Loss Avg", "Validation Accuracy Avg", "Validation Loss Max", "Validation Accuracy Max"]
        rows = [
            [val_loss_min, val_acc_min, val_loss_avg, val_acc_avg, val_loss_max, val_acc_max]
        ]
        file.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")
        
        headers = ["Test Loss", "Test Accuracy"]
        rows = [
            [test_loss, test_acc]
        ]
        file.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")

        headers = ["Sensitivity", "Specificity"]
        rows = [
            [sensitivity[i], specificity[i]] for i in range(len(sensitivity))
        ]

        file.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")

        headers = ["Best Time (s)"]
        rows = [
            [best_speed]
        ]
        file.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")
    
    draw.plot_metrics(output_path, 
                      all_train_losses, all_train_accuracies,
                       all_validation_losses, all_validation_accuracies, "metrics")
    draw.plot_metrics(output_path,
                      best_train_losses, best_train_accuracies,
                      best_validation_losses, best_validation_accuracies, "metrics_best")
    draw.plot_confusion(output_path, conf_matrix)

    predict_samples(best_model, device, output_path)

def retries(args, device):

    output_path = os.path.join(os.path.dirname(__file__), 'output', args.output_path)
    os.makedirs(output_path, exist_ok=True)

    val_loss = []
    val_acc = []

    all_train_losses = []
    all_train_accuracies = []
    all_validation_losses = []
    all_validation_accuracies = []

    best_val_acc = 0.0

    best_train_losses = []
    best_train_accuracies = []
    best_validation_losses = []
    best_validation_accuracies = []

    arch = None
    model = None
    best_arch = None
    best_model = None

    test_loader = None
    best_speed = float('inf')

    for i in range(5):
        print(f"\n=== RUN {i+1} ===")

        if args.model == "mlp":
            arch = Architecture(
                input_dim=784,
                hidden_dims=args.hidden_dims,
                output_dim=10,
                dropout=args.dropout
            )
            model = SimpleMLP(arch).to(device)
        elif args.model == "cnn":
            arch = Architecture(
                input_dim=1,
                hidden_dims=args.hidden_dims,
                output_dim=10,
                dropout=args.dropout
            )
            model = SimpleCNN(arch).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_loader, val_loader, test_loader = get_loaders(batch_size=args.batch_size)

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        start = time.time()
        for t_loss, t_acc, v_loss, v_acc in train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=args.epochs):
            train_losses.append(t_loss)
            val_losses.append(v_loss)
            train_accuracies.append(t_acc)
            val_accuracies.append(v_acc)
        best_speed = min(best_speed, time.time() - start)

        print(f"Run {i+1} finished | Val Acc: {val_accuracies[-1]:.4f}")

        val_loss.append(val_losses[-1])
        val_acc.append(val_accuracies[-1])

        all_train_losses += train_losses
        all_train_accuracies += train_accuracies
        all_validation_losses += val_losses
        all_validation_accuracies += val_accuracies

        if val_accuracies[-1] > best_val_acc:
            best_val_acc = val_accuracies[-1]
            best_train_losses = train_losses
            best_train_accuracies = train_accuracies
            best_validation_losses = val_losses
            best_validation_accuracies = val_accuracies

            best_arch = arch
            best_model = model

    val_loss_min = min(val_loss)
    val_acc_min = min(val_acc)
    val_loss_max = max(val_loss)
    val_acc_max = max(val_acc)
    val_loss_avg = sum(val_loss) / len(val_loss)
    val_acc_avg = sum(val_acc) / len(val_acc)

    test_acc, conf_matrix, sensitivity, specificity = get_metrics(best_model, test_loader, device)

    print(f"Validation Loss - Min: {val_loss_min}, Max: {val_loss_max}, Avg: {val_loss_avg}")
    print(f"Validation Accuracy - Min: {val_acc_min}, Max: {val_acc_max}, Avg: {val_acc_avg}")
    print(f"Test Accuracy: {test_acc}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Sensitivity: {sensitivity}, Specificity: {specificity}")

    output_file = os.path.join(output_path, "result.txt")
    with open(output_file, "w") as file:
        file.write("=== Training Configuration ===\n")
        headers = ["Device", "Epochs", "Learning Rate", "Batch Size"]
        rows = [[str(device), args.epochs, args.lr, args.batch_size]]
        file.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")

        file.write("\n=== Model Architecture ===\n")
        headers = ["Input Dim"] + [f"Hidden Dim {i}/Dropout {i}" for i in range(len(args.hidden_dims))] + ["Output Dim"]
        rows = [[str(arch.input_dim)] + [f"{dim}/{dropout}" for dim, dropout in zip(arch.hidden_dims, arch.dropout)] + [str(arch.output_dim)]]
        file.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")

        file.write("\n=== Results (5 runs) ===\n")
        headers = ["Val Loss Min", "Val Acc Min", "Val Loss Avg", "Val Acc Avg", "Val Loss Max", "Val Acc Max"]
        rows = [[val_loss_min, val_acc_min, val_loss_avg, val_acc_avg, val_loss_max, val_acc_max]]
        file.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")

        headers = ["Test Accuracy"]
        rows = [[test_acc]]
        file.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")

        headers = ["Sensitivity", "Specificity"]
        rows = [[sensitivity[i], specificity[i]] for i in range(len(sensitivity))]
        file.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")

        headers = ["Best Time (s)"]
        rows = [[best_speed]]
        file.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")

    draw.plot_metrics(output_path,
                      all_train_losses, all_train_accuracies,
                      all_validation_losses, all_validation_accuracies,
                      name="metrics_all")
    draw.plot_metrics(output_path,
                      best_train_losses, best_train_accuracies,
                      best_validation_losses, best_validation_accuracies,
                      name="metrics_best")
    draw.plot_confusion(output_path, conf_matrix)

    predict_samples(best_model, device, output_path)

if __name__ == "__main__":
    main()
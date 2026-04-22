import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def predict_samples(model, device, output_path, seed=42):
    rng = np.random.default_rng(seed)

    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    targets = np.array(test_dataset.targets)

    selected_indices = []
    for cls in range(10):
        cls_indices = np.where(targets == cls)[0]
        selected_indices.append(int(rng.choice(cls_indices)))

    model.eval()

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))

    with torch.no_grad():
        for col, idx in enumerate(selected_indices):
            image, true_label = test_dataset[idx]

            input_tensor = image.unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_label = int(probs.argmax())

            axes[0][col].imshow(image.squeeze(), cmap='gray')
            axes[0][col].set_title(f'True: {true_label}\nPred: {pred_label}',
                                   color='green' if pred_label == true_label else 'red',
                                   fontsize=9)
            axes[0][col].axis('off')

            axes[1][col].bar(range(10), probs, color='steelblue')
            axes[1][col].set_xticks(range(10))
            axes[1][col].set_ylim(0, 1)
            axes[1][col].set_xlabel('Class')
            axes[1][col].tick_params(labelsize=7)

    axes[1][0].set_ylabel('Probability')

    fig.suptitle('Sample Predictions (green = correct, red = wrong)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, 'predictions.png'))
    plt.close(fig)

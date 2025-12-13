
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from scLDL.label_enhancer import DiffLEVI

def get_mnist_subset(n_train=2000, n_test=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Download if needed
    os.makedirs('./data', exist_ok=True)
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Subset
    np.random.seed(42)
    train_idx = np.random.choice(len(train_data), n_train, replace=False)
    test_idx = np.random.choice(len(test_data), n_test, replace=False)
    
    # Extract tensors
    X_train = train_data.data[train_idx].float().unsqueeze(1) / 255.0
    y_train = train_data.targets[train_idx]
    
    X_test = test_data.data[test_idx].float().unsqueeze(1) / 255.0
    y_test = test_data.targets[test_idx]
    
    # Convert labels to one-hot for DiffLEVI (clean labels)
    L_train = F.one_hot(y_train, num_classes=10).float()
    
    return X_train, L_train, y_train, X_test, y_test

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback if tqdm is missing
    def tqdm(iterable, desc=""):
        return iterable

def train_and_eval():
    print("Loading data...")
    # Use full MNIST dataset
    X_train, L_train, y_train, X_test, y_test = get_mnist_subset(n_train=60000, n_test=10000)
    
    print(f"Training DiffLEVI on {len(X_train)} clean samples (Full MNIST)...")
    
    # Increase epochs to 100 as requested
    model = DiffLEVI(
        n_features=784,
        n_outputs=10,
        encoder_type='resnet',
        input_shape=(1, 28, 28),
        epochs=10, 
        batch_size=64,
        timesteps=1000,
        lr=2e-4
    )
    print(f"Using device: {model.device}")
    
    # Use subset for validation speed? Or full test set? 
    # User said "add test accuracy monitory". X_test has 10k samples.
    # Diffusion predict is slow. 10k samples might take ~constant * 1000 * 10000 ops.
    # Actually, let's use a smaller subset (e.g. 500) for intermediate validation to keep speed reasonable,
    # unless user insists on full test set accuracy. 
    # "test accuracy monitory" implies the test set. 
    # But 10k samples might take 10 mins per epoch just to validate.
    # I'll use a subset of 500 for validation feedback loop, to be safe.
    # Or just pass X_test[:500].
    
    val_subset_size = 500
    X_val = X_test[:val_subset_size]
    y_val = y_test[:val_subset_size].numpy()
    
    model.fit(X_train, L_train, X_val=X_val, y_val=y_val)
    
    # Plot training loss and validation accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(model.history['loss'])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(model.history['val_acc'])
    plt.title("Validation Accuracy (Subset)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    
    plt.tight_layout()
    plt.savefig("difflevi_training_metrics.png")
    print("Saved metrics plot to difflevi_training_metrics.png")
    
    print("Evaluating on full test set...")
    # Reduce n_samples if too slow, but n_samples=5 is good for stability
    # Predict in batches to show progress/avoid OOM? 
    # For 10k samples with diffusion, simple predict might take a while.
    # We'll rely on user patience or implement batched prediction if needed.
    probs = model.predict(X_test, n_samples=5) 
    y_pred = np.argmax(probs, axis=1)
    acc = (y_pred == y_test.numpy()).mean()
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    return model, X_test, y_test, probs, y_pred

def plot_confusion_matrix(y_true, y_pred, filename="difflevi_confusion_matrix.png"):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix to {filename}")

def plot_class_distributions(X, probs, y_true, y_pred, filename="difflevi_class_distributions.png"):
    # Plot one example for each digit class (0-9)
    plt.figure(figsize=(15, 6))
    
    seen_digits = set()
    indices = []
    
    # Find one index for each true digit
    for idx, label in enumerate(y_true):
         label = int(label)
         if label not in seen_digits:
             indices.append(idx)
             seen_digits.add(label)
         if len(seen_digits) == 10:
             break
             
    # Sort indices by label associated
    indices.sort(key=lambda i: int(y_true[i]))
    
    for i, idx in enumerate(indices):
        # 2 rows: Top row images, Bottom row distributions
        
        # Image
        ax_img = plt.subplot2grid((2, 10), (0, i))
        ax_img.imshow(X[idx].squeeze(), cmap='gray')
        ax_img.set_title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}")
        ax_img.axis('off')
        
        # Bar chart
        ax_bar = plt.subplot2grid((2, 10), (1, i))
        ax_bar.bar(range(10), probs[idx], color='skyblue')
        # Highlight true class
        ax_bar.get_children()[int(y_true[idx])].set_color('green')
        # Highlight pred class if diff
        if y_pred[idx] != y_true[idx]:
             ax_bar.get_children()[y_pred[idx]].set_color('red')
             
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks([]) # Hide x ticks for cleanliness in small plot
        if i == 0:
            ax_bar.set_ylabel("Probability")
            
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved class distributions to {filename}")


def plot_distributions(probs, y_test, title="Prediction Distributions", filename="difflevi_dist.png"):
    # Plot average predicted probability for each class across all test samples (aggregated)
    avg_probs = np.mean(probs, axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(10), avg_probs)
    plt.title(title)
    plt.xlabel("Digit Class")
    plt.ylabel("Average Predicted Probability")
    plt.xticks(range(10))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved distribution plot to {filename}")

def plot_grid(X, y_true, probs, y_pred, title, filename):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    
    indices = np.random.choice(len(X), 9, replace=False)
    
    for i, idx in enumerate(indices):
        row = i // 3
        col = i % 3
        ax = axs[row, col]
        
        # Image
        # Resize canvas to fit image on left and bar plot on right? 
        # Or just show image with bar plot below? Let's do bar plot below or beside?
        # Let's do simple: Main plot is image, inset or overlay is not easy.
        # Let's split 3x3 cells into sub-subplots? No.
        
        # Better: 3x3 grid where each cell contains the image on left and probabilities on right
        
        # Create a sub-gridspec for this cell? Too complex.
        # Let's just plot the image with the predicted/true label text, 
        # and maybe just plotting the distribution separately is better?
        # The prompt asked "Plot a 3x3 grid with the digits whos for current and incorrect labels."
        # And "Plot the distribution."
        
        # I'll interpret this as: 
        # 3x3 grid of IMAGES. 
        # AND separate plot for "distribution".
        # But wait "digits *whos* for correct and incorrect labels". Maybe "whose"?
        # And "Plot the distribution" came before.
        
        # Let's try to show the bar chart next to the image for each of the 9 examples.
        # Implemented by making a 3x6 grid (allocating 2 columns per item).
        
        pass

    # Re-impl with subplots
    plt.close(fig)
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # We want 9 items.
    
    for i, idx in enumerate(indices):
        # 3 rows, each row has 3 items. Each item needs space for image + bar.
        # layout:
        # Img1 Bar1 | Img2 Bar2 | Img3 Bar3
        
        # So 3 rows, 6 columns.
        row = i // 3
        col_img = (i % 3) * 2
        col_bar = col_img + 1
        
        # Image
        ax_img = plt.subplot2grid((3, 6), (row, col_img))
        ax_img.imshow(X[idx].squeeze(), cmap='gray')
        ax_img.set_title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}")
        ax_img.axis('off')
        
        # Bar chart
        ax_bar = plt.subplot2grid((3, 6), (row, col_bar))
        ax_bar.bar(range(10), probs[idx], color='skyblue')
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks(range(10))
        # Highlight true class
        ax_bar.get_children()[y_true[idx]].set_color('green')
        # Highlight pred class if diff
        if y_pred[idx] != y_true[idx]:
             ax_bar.get_children()[y_pred[idx]].set_color('red')
             
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename)
    plt.close()
    print(f"Saved grid to {filename}")

def main():
    model, X_test, y_test, probs, y_pred = train_and_eval()
    
    # 1. Plot overall distribution
    plot_distributions(probs, y_test)
    
    # 2. Correct examples
    correct_mask = (y_pred == y_test.numpy())
    if np.sum(correct_mask) >= 9:
        X_correct = X_test[correct_mask]
        y_true_correct = y_test[correct_mask]
        probs_correct = probs[correct_mask]
        y_pred_correct = y_pred[correct_mask]
        
        plot_grid(X_correct, y_true_correct, probs_correct, y_pred_correct, 
                  "Correct Predictions (Green=True)", "difflevi_correct.png")
    else:
        print("Not enough correct predictions for 3x3 grid.")
        
    # 3. Incorrect examples
    incorrect_mask = ~correct_mask
    if np.sum(incorrect_mask) >= 9:
        X_wrong = X_test[incorrect_mask]
        y_true_wrong = y_test[incorrect_mask]
        probs_wrong = probs[incorrect_mask]
        y_pred_wrong = y_pred[incorrect_mask]
        
        plot_grid(X_wrong, y_true_wrong, probs_wrong, y_pred_wrong, 
                  "Incorrect Predictions (Green=True, Red=Pred)", "difflevi_incorrect.png")
    else:
        print(f"Not enough incorrect predictions for 3x3 grid (Found {np.sum(incorrect_mask)}).")

    # 4. Confusion Matrix
    plot_confusion_matrix(y_test.numpy(), y_pred)
    
    # 5. Class Distributions
    plot_class_distributions(X_test, probs, y_test.numpy(), y_pred)

if __name__ == "__main__":
    main()

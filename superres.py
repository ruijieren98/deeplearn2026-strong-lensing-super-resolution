import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import torch.utils.data as data
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import os
import random
# %matplotlib inline

"""#### 1.2 Preview the Data"""

# Define the input paths for high-resolution (HR) and low-resolution (LR) training images
train_hr_path = '/scratch/ruijieren/dataset_superres/train/HR'
train_hr_files = [os.path.join(train_hr_path, f) for f in os.listdir(train_hr_path) if f.endswith(".npy")]
train_lr_path = '/scratch/ruijieren/dataset_superres/train/LR'
train_lr_files = [os.path.join(train_lr_path, f) for f in os.listdir(train_lr_path) if f.endswith(".npy")]


"""#### 1.3 Import Training and Validation Data"""

# Set Batch Size
batch_size = 100

# Define a custom Dataset class for loading Super Resolution data
class SuperResolutionDataset(data.Dataset):
    def __init__(self, lr_path, hr_path, augment=True):
        # Initialize the dataset with lists of low-resolution and high-resolution image file paths
        self.lr_files = [os.path.join(lr_path, f) for f in os.listdir(lr_path) if f.endswith(".npy")]
        self.hr_files = [os.path.join(hr_path, f) for f in os.listdir(hr_path) if f.endswith(".npy")]
        self.augment = augment

    def __len__(self):
        # Return the total number of low-resolution images (The number of HR and LR images is the same)
        return len(self.lr_files)

    def __getitem__(self, idx):
        # Load the low-resolution and high-resolution images from the file paths
        lr_image = np.load(self.lr_files[idx])
        hr_image = np.load(self.hr_files[idx])

        if self.augment:
            # Horizontal flip (left-right)
            if random.random() < 0.5:
                lr_image = np.flip(lr_image, axis=-1)
                hr_image = np.flip(hr_image, axis=-1)

            # Vertical flip (up-down)
            if random.random() < 0.5:
                lr_image = np.flip(lr_image, axis=-2)
                hr_image = np.flip(hr_image, axis=-2)

        lr_image = lr_image.copy()
        hr_image = hr_image.copy()
        # Convert numpy arrays to PyTorch tensors and return them
        return torch.from_numpy(lr_image).float(), torch.from_numpy(hr_image).float()

# Create the training data loader
train_data = SuperResolutionDataset('/scratch/ruijieren/dataset_superres/train/LR', '/scratch/ruijieren/dataset_superres/train/HR')
train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

# Create the validation data loader
val_data = SuperResolutionDataset('/scratch/ruijieren/dataset_superres/val/LR', '/scratch/ruijieren/dataset_superres/val/HR')
val_data_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)

"""### 2. Training
"""

# Define the Super-Resolution Convolutional Neural Network (SRCNN) model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # First convolutional layer: 1 input channel, 64 output channels, 9x9 kernel, 4 pixels padding
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=4)
        # Second convolutional layer: 64 input channels, 32 output channels, 5x5 kernel, 2 pixels padding
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        # Third convolutional layer: 32 input channels, 1 output channel, 5x5 kernel, 2 pixels padding
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, padding=1)
        # ReLU activation function
        self.relu = nn.ReLU()
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        # Apply the first convolutional layer followed by ReLU activation
        x = self.relu(self.conv1(x))
        # Apply the second convolutional layer followed by ReLU activation
        x = self.relu(self.conv2(x))
        # Apply the third convolutional layer
        x = self.conv3(x)
        # Learnable upsample
        x = self.shuffle(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels, 
                channels, 
                kernel_size=3, 
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels, 
                channels, 
                kernel_size=3, 
                padding=1
            )
        )

    def forward(self, x):
        residual = x
        out = self.block(x)

        # local skip connection
        out = out + residual

        return out


class SRCNN_ResNet(nn.Module):
    def __init__(self, num_blocks=16):
        super(SRCNN_ResNet, self).__init__()

        # Initial feature extraction
        self.conv_input = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=9,
            padding=4
        )

        self.relu = nn.ReLU(inplace=True)

        # Residual encoder
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(64)
                for _ in range(num_blocks)
            ]
        )

        # Feature reconstruction after residual blocks
        self.conv_mid = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Global skip connection output layer
        self.conv_output = nn.Conv2d(
            in_channels=64,
            out_channels=4,
            kernel_size=3,
            padding=1
        )

        # Upsampling
        self.pixel_shuffle = nn.PixelShuffle(2)


    def forward(self, x):

        # First convolution
        x = self.relu(self.conv_input(x))

        # Save global residual
        global_skip = x

        # Residual blocks
        x = self.res_blocks(x)

        # Reconstruction convolution
        x = self.conv_mid(x)

        # Global skip connection
        x = x + global_skip

        # Final convolution
        x = self.conv_output(x)

        # Upsampling x2
        x = self.pixel_shuffle(x)

        return x


# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the SRCNN model and move it to the appropriate device
model = SRCNN_ResNet().to(device)

def evaluate_model(model, val_data_loader, device, criteria, criteria2):
    model.eval()  # Set the model to evaluation mode
    out_eval = []  # List to store model predictions

    # Prepare ground truth for comparison. Need to ensure val_hr is always fresh for this function.
    # It's better to pass it in or load it within the function if not already in memory.
    # For now, let's assume val_hr will be available from previous cells or re-created.

    # Temporarily load val_hr within the function for self-containment if not passed.
    # If val_hr is already globally defined and correctly structured, this part can be omitted.
    local_val_hr = []
    with torch.no_grad():
        for lr, hr in val_data_loader:
            lr = lr.to(device)
            recon = model(lr)  # Get model predictions
            out_eval.append(recon.cpu().detach().numpy())
            local_val_hr.append(hr.cpu().numpy())
            del lr, hr, recon
            torch.cuda.empty_cache()

    dataSR_eval = np.concatenate(out_eval, axis=0)
    local_val_hr = np.concatenate(local_val_hr, axis=0)

    losses_eval = []
    losses2_eval = []
    Ssim_eval = []
    Psnr_eval = []

    for i in range(dataSR_eval.shape[0]):
        losses_eval.append(criteria(torch.from_numpy(dataSR_eval[i]), torch.from_numpy(local_val_hr[i])))
        losses2_eval.append(criteria2(torch.from_numpy(dataSR_eval[i]), torch.from_numpy(local_val_hr[i])))
        Ssim_eval.append(ssim(local_val_hr[i][0], dataSR_eval[i][0], data_range=dataSR_eval[i][0].max() - dataSR_eval[i][0].min()))
        Psnr_eval.append(psnr(local_val_hr[i][0], dataSR_eval[i][0], data_range=dataSR_eval[i][0].max() - dataSR_eval[i][0].min()))

    avg_mse = np.average(losses_eval)
    avg_l1 = np.average(losses2_eval)
    avg_ssim = np.average(Ssim_eval)
    avg_psnr = np.average(Psnr_eval)

    model.train() # Set the model back to training mode
    return avg_mse, avg_l1, avg_ssim, avg_psnr

"""#### 2.2 Training the Super-Resolution CNN Model"""

criteria = nn.MSELoss()  # Mean Squared Error Loss
criteria2 = nn.L1Loss() # L1 Loss for evaluation
# Optimizer (Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

n_epochs = 200  # Number of Training Epochs

# Lists to store metrics for plotting
train_losses_per_epoch = []
eval_mse_history = []
eval_l1_history = []
eval_ssim_history = []
eval_psnr_history = []
epochs_for_eval = []

best_psnr = -1.0 # Initialize with a very low PSNR score
best_model_path = 'super_resolution_model_best.pt'


for epoch in tqdm(range(1, n_epochs + 1)):
    model.train() # Set model to training mode at the start of each epoch
    train_loss = 0.0  # Initialize training loss for the epoch

    # Iterate over the training data loader
    for step, (lr, hr) in enumerate(train_data_loader):
        lr = Variable(lr).type(torch.FloatTensor).to(device)  # Move low-resolution images to the device
        hr = Variable(hr).type(torch.FloatTensor).to(device)  # Move high-resolution images to the device
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(lr)  # Forward pass through the model
        loss = (outputs-hr) ** 2  # Calculate the loss
        loss *= (1.-hr)*2.
        loss = torch.mean(loss)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the model parameters

        train_loss += loss.item()  # Accumulate the training loss
        del lr, hr, outputs, loss # Free memory
        torch.cuda.empty_cache()

    avg_train_loss = train_loss / len(train_data_loader)  # Compute average training loss for the epoch
    train_losses_per_epoch.append(avg_train_loss)  # Store average training loss

    # Evaluate every 10 epochs
    if epoch % 10 == 0:
        print(f"\nEvaluating model after Epoch {epoch}...")
        avg_mse, avg_l1, avg_ssim, avg_psnr = evaluate_model(model, val_data_loader, device, criteria, criteria2)
        eval_mse_history.append(avg_mse)
        eval_l1_history.append(avg_l1)
        eval_ssim_history.append(avg_ssim)
        eval_psnr_history.append(avg_psnr)
        epochs_for_eval.append(epoch)
        print(f"Epoch {epoch} - Eval MSE: {avg_mse:.7f}, L1: {avg_l1:.7f}, SSIM: {avg_ssim:.5f}, PSNR: {avg_psnr:.5f}")

        # Save the best model based on PSNR
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with PSNR: {best_psnr:.5f}")

print("\nTraining complete!")
"""### 5. Visualization of Training Loss and Evaluation Metrics"""

plt.figure(figsize=(12, 6))
plt.plot(range(1, n_epochs + 1), train_losses_per_epoch, marker='o')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Training Loss (MSE)')
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(epochs_for_eval, eval_mse_history, marker='o', color='blue')
plt.title('Evaluation MSE over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average MSE')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(epochs_for_eval, eval_l1_history, marker='o', color='green')
plt.title('Evaluation L1 Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average L1 Loss')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(epochs_for_eval, eval_ssim_history, marker='o', color='red')
plt.title('Evaluation SSIM over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average SSIM')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(epochs_for_eval, eval_psnr_history, marker='o', color='purple')
plt.title('Evaluation PSNR over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average PSNR')
plt.grid(True)

plt.tight_layout()
#plt.show()
plt.savefig('eval.png')

"""### 3. Testing

#### 3.1 Testing the Super-Resolution CNN Model on Validation Data - Calculate Quantitative Metrics

- **MSE (Mean Squared Error):** A measure of the average squared difference between the estimated values and the actual value. Lower values indicate better performance.

- **SSIM (Structural Similarity Index):** A method for measuring the similarity between two images. It is used to measure the quality of the super-resolved images compared to the original high-resolution images.

- **PSNR (Peak Signal-to-Noise Ratio):** The ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher values indicate better image quality.

You may refer to this [article](https://medium.com/@datamonsters/a-quick-overview-of-methods-to-measure-the-similarity-between-images-f907166694ee) to learn more about these metrics

*Note: Metrics need to be calculated on a sample-by-sample basis, not on a batch basis. This is because metrics like SSIM and PSNR are used for assessing the quality of individual images and the scikit-image functions do not average over the first axis when we pass batches of images to them.*
"""

# Calculate Metrics

# Set the model to evaluation mode
model.eval()
out = []  # List to store model predictions
val_hr = []
with torch.no_grad():  # Disable gradient calculation for validation
    for lr, hr in val_data_loader:
        lr = lr.to(device)  # Move low-resolution images to the device
        hr = hr.to(device)  # Move high-resolution images to the device
        recon = model(lr)  # Get model predictions
        out.append(recon.cpu().detach().numpy())  # Append predictions to the list and move to CPU
        val_hr.append(hr.cpu().numpy())
        del lr, hr, recon  # Free memory
        torch.cuda.empty_cache()  # Clear cached memory
dataSR = np.concatenate(out, axis=0)  # Concatenate predictions along the batch axis
val_hr = np.concatenate(val_hr, axis=0)  # Concatenate ground truth images along the batch axis

# Calculate metrics
print("Metrics:")
criteria = nn.MSELoss()  # Mean Squared Error Loss
criteria2 = nn.L1Loss()  # L1 Loss

losses = []  # List to store MSE losses
losses2 = []  # List to store L1 losses
Ssim = []  # List to store SSIM scores
Psnr = []  # List to store PSNR scores

for i in range(dataSR.shape[0]):
    # Calculate MSE loss between predicted and ground truth images
    losses.append(criteria(torch.from_numpy(dataSR[i]), torch.from_numpy(val_hr[i])))
    # Calculate L1 loss between predicted and ground truth images
    losses2.append(criteria2(torch.from_numpy(dataSR[i]), torch.from_numpy(val_hr[i])))
    # Calculate SSIM score between predicted and ground truth images
    Ssim.append(ssim(val_hr[i][0], dataSR[i][0], data_range=dataSR[i][0].max() - dataSR[i][0].min()))
    # Calculate PSNR score between predicted and ground truth images
    Psnr.append(psnr(val_hr[i][0], dataSR[i][0], data_range=dataSR[i][0].max() - dataSR[i][0].min()))

# Print average metrics
print("Average MSE super resolution samples: " + str('%.7f' % np.average(losses)))
print("Average L1 super resolution samples: " + str('%.7f' % np.average(losses2)))
print("Average SSIM super resolution samples: " + str('%.5f' % np.average(Ssim)))
print("Average PSNR super resolution samples: " + str('%.5f' % np.average(Psnr)))

"""#### 3.2 Visualize Outputs for Qualitative Analysis"""

# Visualize Outputs
with torch.no_grad():  # Disable gradient calculation
    for lr, hr in val_data_loader:
        lr = lr.to(device)  # Move low-resolution images to the device
        hr = hr.to(device)  # Move high-resolution images to the device
        output = model(lr)  # Get model predictions

        lr = lr.cpu().numpy()  # Move low-resolution images to CPU and convert to numpy array
        output = output.cpu().numpy()  # Move predicted images to CPU and convert to numpy array
        hr = hr.cpu().numpy()  # Move high-resolution images to CPU and convert to numpy array

        # Display the results
        plt.figure(figsize=(12, 8))  # Set figure size
        for i in range(5):  # Display first 5 images
            plt.subplot(3, 5, i + 1)  # Create subplot for low-resolution image
            plt.imshow(lr[i].reshape(64, 64), cmap='gray')  # Display low-resolution image in grayscale
            plt.title('Low Res')  # Set title for low-resolution image
            plt.axis('off')  # Hide axis
            plt.subplot(3, 5, i + 6)  # Create subplot for high-resolution image
            plt.imshow(hr[i].reshape(128, 128), cmap='gray')  # Display high-resolution image in grayscale
            plt.title('High Res')  # Set title for high-resolution image
            plt.axis('off')  # Hide axis
            plt.subplot(3, 5, i + 11)  # Create subplot for output image
            plt.imshow(output[i].reshape(128, 128), cmap='gray')  # Display predicted image in grayscale
            plt.title('Output')  # Set title for predicted image
            plt.axis('off')  # Hide axis
        plt.savefig('vis.png')  # Display the figure
        break  # Break after first batch to visualize

"""## Submission Guidelines

* Fill out the pre- and post- hackathon surveys.
* You are required to submit a Google Colab Jupyter Notebook (.ipynb and pdf) clearly showing your implementation along with the evaluation metrics (MSE, SSIM, and PSNR) for the validation data.
* You must also submit the final trained model, including the model architecture and the trained weights ( For example: HDF5 file, .pb file, .pt file, etc. )
* You can use this example notebook as a template for your work.

> **_NOTE:_**  You are free to use any ML framework such as PyTorch, Keras, TensorFlow, etc.
"""

# torch.save(model.state_dict(), 'super_resolution_model.pt')
# print("Model weights saved to super_resolution_model.pt")

"""### 4. Load and Evaluate Model"""

# Initialize a new model instance
loaded_model = SRCNN_ResNet().to(device)

# Load the saved state dictionary
loaded_model.load_state_dict(torch.load('super_resolution_model_best.pt'))
loaded_model.eval() # Set the loaded model to evaluation mode

print("Model weights loaded successfully.")

# Evaluate the loaded model
# Set the model to evaluation mode
loaded_model.eval()
out_loaded = []  # List to store model predictions
val_hr = []
with torch.no_grad():  # Disable gradient calculation for validation
    for lr, hr in val_data_loader:
        lr = lr.to(device)  # Move low-resolution images to the device
        hr = hr.to(device)  # Move high-resolution images to the device
        recon = loaded_model(lr)  # Get model predictions
        out_loaded.append(recon.cpu().detach().numpy())  # Append predictions to the list and move to CPU
        val_hr.append(hr.cpu().numpy())
        del lr, hr, recon  # Free memory
        torch.cuda.empty_cache()  # Clear cached memory
dataSR_loaded = np.concatenate(out_loaded, axis=0)  # Concatenate predictions along the batch axis
val_hr = np.concatenate(val_hr, axis=0)  # Concatenate ground truth images along the batch axis
# Prepare ground truth for comparison (val_hr is already available from previous execution)

# Calculate metrics for the loaded model
print("\nMetrics for Loaded Model:")
losses_loaded = []  # List to store MSE losses
losses2_loaded = []  # List to store L1 losses
Ssim_loaded = []  # List to store SSIM scores
Psnr_loaded = []  # List to store PSNR scores

for i in range(dataSR_loaded.shape[0]):
    # Calculate MSE loss between predicted and ground truth images
    losses_loaded.append(criteria(torch.from_numpy(dataSR_loaded[i]), torch.from_numpy(val_hr[i])))
    # Calculate L1 loss between predicted and ground truth images
    losses2_loaded.append(criteria2(torch.from_numpy(dataSR_loaded[i]), torch.from_numpy(val_hr[i])))
    # Calculate SSIM score between predicted and ground truth images
    Ssim_loaded.append(ssim(val_hr[i][0], dataSR_loaded[i][0], data_range=dataSR_loaded[i][0].max() - dataSR_loaded[i][0].min()))
    # Calculate PSNR score between predicted and ground truth images
    Psnr_loaded.append(psnr(val_hr[i][0], dataSR_loaded[i][0], data_range=dataSR_loaded[i][0].max() - dataSR_loaded[i][0].min()))

# Print average metrics for the loaded model
print("Average MSE super resolution samples (Loaded Model): " + str('%.7f' % np.average(losses_loaded)))
print("Average L1 super resolution samples (Loaded Model): " + str('%.7f' % np.average(losses2_loaded)))
print("Average SSIM super resolution samples (Loaded Model): " + str('%.5f' % np.average(Ssim_loaded)))
print("Average PSNR super resolution samples (Loaded Model): " + str('%.5f' % np.average(Psnr_loaded)))


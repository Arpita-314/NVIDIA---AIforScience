import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# =================================================================
# PHASE 1: PHYSICS-INSIGHT DATA GENERATION
# This mimics the behavior of light in a Silicon-on-Insulator (SOI) 
# chip, where Silicon (eps=12) traps light through total internal reflection.
# =================================================================

def generate_photonic_dataset(samples=400, grid_size=64):
    print(f"--- Generating {samples} Photonic Data Samples ---")
    geometries = np.zeros((samples, grid_size, grid_size))
    fields = np.zeros((samples, grid_size, grid_size))
    
    for i in range(samples):
        # Randomize Waveguide Geometry
        width = np.random.randint(8, 25)
        offset = np.random.randint(-10, 10)
        
        # Draw the Silicon Waveguide (1.0 = Silicon, 0.0 = Air)
        y_start, y_end = 32 + offset - width//2, 32 + offset + width//2
        geometries[i, y_start:y_end, :] = 1.0
        
        # Simulate Mode Propagation (Simplified Maxwell Solution)
        x = np.linspace(-5, 5, grid_size)
        y = np.linspace(-5, 5, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # The light field decays exponentially outside the silicon
        sigma = width / 15.0
        mode = np.exp(-(Y - (offset/6.4))**2 / (2 * sigma**2))
        phase = np.sin(X * 4) # Represents the oscillating wave
        fields[i] = mode * phase * geometries[i]
        
    return (torch.FloatTensor(geometries).unsqueeze(1), 
            torch.FloatTensor(fields).unsqueeze(1))

# =================================================================
# PHASE 2: NEURAL SURROGATE ARCHITECTURE
# A Convolutional Encoder-Decoder (UNet-Lite) designed to act as a 
# "Digital Twin" of Maxwell's Equations.
# =================================================================

class OptiCoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(), # Downsample
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),  # Upsample
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x): return self.network(x)

# =================================================================
# PHASE 3: TRAINING AND PERFORMANCE PROFILING
# =================================================================

# Prepare Data
X, Y = generate_photonic_dataset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OptiCoreNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.MSELoss()

# Training Loop
start_time = time.time()
for epoch in range(101):
    model.train()
    optimizer.zero_grad()
    out = model(X.to(device))
    loss = criterion(out, Y.to(device))
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Physics-Loss: {loss.item():.7f}")

train_time = time.time() - start_time
print(f"Training Complete in {train_time:.2f}s")

# =================================================================
# PHASE 4: PORTFOLIO VISUALIZATION
# =================================================================

model.eval()
with torch.no_grad():
    test_idx = 0
    pred = model(X[test_idx:test_idx+1].to(device)).cpu().numpy()[0,0]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(X[test_idx, 0], cmap='gray'); axes[0].set_title("Silicon Design")
axes[1].imshow(Y[test_idx, 0], cmap='RdBu'); axes[1].set_title("Maxwell Ground Truth")
axes[2].imshow(pred, cmap='RdBu'); axes[2].set_title("AI Surrogate Prediction")
plt.show()

```
  /$$$$$$   /$$$$$$   /$$$$$$  /$$    /$$  /$$$$$$  /$$$$$$$$
 /$$__  $$ /$$__  $$ /$$__  $$| $$   | $$ /$$__  $$| $$_____/
| $$  \__/| $$  \__/| $$  \__/| $$   | $$| $$  \ $$| $$      
| $$      | $$ /$$$$| $$      |  $$ / $$/| $$$$$$$$| $$$$$   
| $$      | $$|_  $$| $$       \  $$ $$/ | $$__  $$| $$__/   
| $$    $$| $$  \ $$| $$    $$  \  $$$/  | $$  | $$| $$      
|  $$$$$$/|  $$$$$$/|  $$$$$$/   \  $/   | $$  | $$| $$$$$$$$
 \______/  \______/  \______/     \_/    |__/  |__/|________/
```                                                          
                                                             
# CGCVAE

**CGCVAE** (Crystal Graph Conditional Variational Autoencoder) is a deep generative model for conditional generation of crystal structures based on graph neural networks (GNNs). This project aims to generate crystal structures conditioned on properties such as bandgap or formation energy.

## 🚀 Overview

- Encode crystal graphs using a pretrained `UniCrystalFormer` model
- Learn a latent representation using Conditional VAE
- Generate new crystal-like structures conditioned on target properties

## 📦 Features

- Graph-based representation of materials (PyTorch Geometric)
- Conditional generation with latent space control
- Modular architecture for easy experimentation

## 📁 Project Structure
```
project_root/
├── main.py                   # Entry point
├── train.py                  # Training loop
├── generate.py               # Generation script
├── src/
│   └── models/
│       └── unicrystalformer/ # Encoder backbone (UniCrystalFormer)
│       └── encoder.py        # Encoder wrapper
│       └── decoder.py        # Graph decoder
│       └── cgcvae.py         # ConditionalGraphVAE
```

## 🔧 Requirements

- Python 3.10.10 (recommended)
- PyTorch
- PyTorch Geometric

## 🛠️ Setup

```bash
setup.bat
```
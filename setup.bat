@echo off

echo Creating virtual environment...
python -m venv venv
call .\venv\Scripts\activate

python -m pip install --upgrade pip
pip install typing-extensions==4.12.2 --index-url https://pypi.org/simple

echo Installing PyTorch...
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

echo Installing torch_geometric...
pip install torch_geometric

echo Installing PyG dependencies...
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

echo Installing from requirements.txt...
pip install -r requirements.txt
1. Create a venv : 
```
python -m venv venv
```
2. Activate the venv : 
```
.\venv\Scripts\activate
```
3. Install torch with CUDA : 
```
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```
4. Install the requirements : 
```
pip install -r requirements.txt
```
6. Create .env file with :
```
MODEL_DIR="./models/"
HUGGINGFACE_TOKEN=".."
```
7. Run the app : 
```
python app.py
```
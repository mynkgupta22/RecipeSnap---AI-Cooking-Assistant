# Create and activate virtual environment
python -m venv backend/venv
./backend/venv/Scripts/Activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Create necessary directories
mkdir -p backend/models

Write-Host "Backend setup complete!"
Write-Host "Please create the following .env files:"
Write-Host "1. frontend/.env.local with:"
Write-Host "NEXT_PUBLIC_API_URL=http://localhost:8000"
Write-Host ""
Write-Host "2. backend/.env with:"
Write-Host "MODEL_PATH=./models"
Write-Host "CUDA_VISIBLE_DEVICES=0" 
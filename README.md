# RecipeSnap - AI Cooking Assistant

RecipeSnap is an innovative AI-powered cooking assistant that helps you create delicious recipes from the ingredients in your fridge. Simply take a photo of your ingredients, and let our AI suggest personalized recipes!

## Features

- 📸 Upload or take photos of ingredients
- 🔍 Advanced ingredient detection using state-of-the-art computer vision
- 🤖 Smart recipe recommendations using AI
- 📱 Responsive modern interface
- ⚡ Real-time processing

## Tech Stack

### Frontend
- Next.js 14 with TypeScript
- Tailwind CSS for styling
- Shadcn/ui for component library

### Backend
- FastAPI (Python)
- ML Models:
  - Image Captioning: nlpconnect/vit-gpt2-image-captioning
  - Object Detection: facebook/detr-resnet-50
  - Recipe Generation: mistralai/Mistral-7B-Instruct

## Setup Instructions

### Prerequisites
- Node.js 18+
- Python 3.10+
- CUDA-capable GPU (recommended)

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Environment Variables

Create a `.env` file in both frontend and backend directories:

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend (.env)
```
MODEL_PATH=./models
CUDA_VISIBLE_DEVICES=0  # If using GPU
```

## API Endpoints

- `POST /api/analyze-image`: Upload and analyze fridge image
- `GET /api/recipes`: Get recipe recommendations

## License

MIT License - see LICENSE file for details 
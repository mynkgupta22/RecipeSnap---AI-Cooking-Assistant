from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Set
import torch
from transformers import pipeline, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import io
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = FastAPI(title="RecipeSnap API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
image_captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
object_detector_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
object_detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Define food categories and common ingredients
FOOD_CATEGORIES = {
    'fruits': {'apple', 'orange', 'banana', 'pear', 'grape', 'lemon', 'strawberry', 'blueberry', 'raspberry'},
    'vegetables': {'carrot', 'broccoli', 'tomato', 'potato', 'onion', 'lettuce', 'cucumber', 'pepper', 'mushroom'},
    'proteins': {'chicken', 'beef', 'fish', 'egg', 'tofu', 'pork', 'shrimp', 'salmon'},
    'dairy': {'cheese', 'milk', 'yogurt', 'butter', 'cream'},
    'pantry': {'bread', 'rice', 'pasta', 'flour', 'sugar', 'oil'}
}

# Flatten food items into a set for easy lookup
FOOD_ITEMS: Set[str] = set().union(*FOOD_CATEGORIES.values())

# Common kitchen items to exclude
NON_FOOD_ITEMS = {
    'chair', 'bottle', 'vase', 'cup', 'bowl', 'plate', 'fork', 'knife', 'spoon',
    'refrigerator', 'oven', 'microwave', 'sink', 'table', 'counter', 'cabinet',
    'potted plant', 'person', 'clock', 'book'
}

class IngredientResponse(BaseModel):
    ingredients: List[str]
    confidence_scores: List[float]
    caption: str

class RecipeResponse(BaseModel):
    title: str
    ingredients: List[str]
    instructions: List[str]
    cooking_time: str
    difficulty: str

def is_food_item(item: str) -> bool:
    """Check if an item is a food item."""
    # Convert to lowercase for comparison
    item = item.lower()
    # Check if item is in our food items set
    if item in FOOD_ITEMS:
        return True
    # Check if item is NOT in our non-food items set
    if item in NON_FOOD_ITEMS:
        return False
    # For items we're not sure about, we'll include them
    # as they might be valid food items not in our list
    return True

def generate_recipe(ingredients: List[str]) -> RecipeResponse:
    """Generate a recipe based on the available ingredients."""
    # Filter and deduplicate ingredients
    valid_ingredients = list(set([ing.lower() for ing in ingredients if is_food_item(ing)]))
    
    if not valid_ingredients:
        return RecipeResponse(
            title="No Valid Ingredients Detected",
            ingredients=[],
            instructions=["Please provide a photo with visible food ingredients."],
            cooking_time="N/A",
            difficulty="N/A"
        )

    # For now, we'll return a simple recipe template
    # TODO: Replace with actual Mistral-7B-Instruct integration
    return RecipeResponse(
        title=f"Recipe with {', '.join(valid_ingredients[:3])}",
        ingredients=valid_ingredients,
        instructions=[
            f"Prepare {', '.join(valid_ingredients)}",
            "Combine ingredients in a suitable way",
            "Cook until done",
            "Serve and enjoy!"
        ],
        cooking_time="30 minutes",
        difficulty="Medium"
    )

@app.get("/")
async def root():
    return {"message": "Welcome to RecipeSnap API"}

@app.post("/api/analyze-image", response_model=IngredientResponse)
async def analyze_image(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File uploaded is not an image")
    
    try:
        # Read and process the image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Get image caption
        try:
            caption_output = image_captioner(image)
            caption = caption_output[0]['generated_text']
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating image caption: {str(e)}")
        
        # Detect objects (ingredients)
        try:
            inputs = object_detector_processor(images=image, return_tensors="pt")
            outputs = object_detector(**inputs)
            
            # Process results
            target_sizes = torch.tensor([image.size[::-1]])
            results = object_detector_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.7
            )[0]
            
            ingredients = []
            confidence_scores = []
            
            # Filter and deduplicate detected objects
            seen_ingredients = set()
            for score, label in zip(results["scores"], results["labels"]):
                item = object_detector.config.id2label[label.item()]
                if is_food_item(item) and item.lower() not in seen_ingredients:
                    ingredients.append(item)
                    confidence_scores.append(float(score))
                    seen_ingredients.add(item.lower())
            
            if not ingredients:
                return IngredientResponse(
                    ingredients=["No food ingredients detected"],
                    confidence_scores=[1.0],
                    caption=caption
                )
            
            return IngredientResponse(
                ingredients=ingredients,
                confidence_scores=confidence_scores,
                caption=caption
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error detecting objects: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        await file.close()

@app.post("/api/recipes", response_model=RecipeResponse)
async def get_recipes(ingredients: Dict[str, List[str]]):
    try:
        ingredient_list = ingredients.get('ingredients', [])
        if not ingredient_list:
            raise HTTPException(status_code=400, detail="No ingredients provided")
        
        return generate_recipe(ingredient_list)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
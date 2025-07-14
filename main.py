# FastAPI Fake News Classifier with Frontend
# Run with: uvicorn main:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import re
import uvicorn
import os
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(title="Fake News Classifier API", version="1.0.0")

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Request/Response models
class TextRequest(BaseModel):
    text: str

class URLRequest(BaseModel):
    url: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probability_fake: float
    probability_real: float
    text_analyzed: str

# Load model and tokenizer
def load_model():
    global model, tokenizer
    
    print("🤖 Loading model and tokenizer...")
    
    # Load tokenizer
    MODEL_NAME = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Load trained weights if available
    if os.path.exists('best_news_classifier.pth'):
        print("📦 Loading trained weights...")
        model.load_state_dict(torch.load('best_news_classifier.pth', map_location=device))
        print("✅ Trained weights loaded successfully!")
    else:
        print("⚠️  No trained weights found. Using base model.")
    
    model = model.to(device)
    model.eval()
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully!")

# Extract text from URL
def extract_text_from_url(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from common news article elements
        text_elements = []
        
        # Try to find title
        title = soup.find('title')
        if title:
            text_elements.append(title.get_text())
        
        # Try to find article content
        for selector in ['article', '.article-content', '.content', '.post-content', 'main', '.main-content']:
            content = soup.select_one(selector)
            if content:
                text_elements.append(content.get_text())
                break
        
        # If no specific content found, get all paragraphs
        if not text_elements:
            paragraphs = soup.find_all('p')
            text_elements.extend([p.get_text() for p in paragraphs])
        
        # Clean and join text
        text = ' '.join(text_elements)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text[:2000]  # Limit to first 2000 characters
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text from URL: {str(e)}")

# Predict function
def predict_text(text: str) -> PredictionResponse:
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(outputs.logits, dim=-1)
            confidence = probabilities.max().item()
            
            # Debug prints
            print("Logits:", outputs.logits)
            print("Probabilities:", probabilities)
            print("Prediction index:", prediction.item())
        
        # Get probabilities for each class
        prob_fake = probabilities[0][0].item()
        prob_real = probabilities[0][1].item()
        print("prob_fake:", prob_fake, "prob_real:", prob_real)
        
        # Determine prediction label
        prediction_label = "FAKE" if prediction.item() == 0 else "REAL"
        
        return PredictionResponse(
            prediction=prediction_label,
            confidence=confidence,
            probability_fake=prob_fake,
            probability_real=prob_real,
            text_analyzed=text[:200] + "..." if len(text) > 200 else text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

# API Routes
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fake News Classifier</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }
            
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .content {
                padding: 40px;
            }
            
            .tabs {
                display: flex;
                margin-bottom: 30px;
                border-bottom: 2px solid #eee;
            }
            
            .tab-button {
                background: none;
                border: none;
                padding: 15px 30px;
                font-size: 1.1em;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                transition: all 0.3s ease;
            }
            
            .tab-button.active {
                color: #667eea;
                border-bottom-color: #667eea;
            }
            
            .tab-content {
                display: none;
            }
            
            .tab-content.active {
                display: block;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
            }
            
            textarea, input[type="url"] {
                width: 100%;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 10px;
                font-size: 1em;
                resize: vertical;
                transition: border-color 0.3s ease;
            }
            
            textarea:focus, input[type="url"]:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 1.1em;
                border-radius: 10px;
                cursor: pointer;
                transition: transform 0.3s ease;
                width: 100%;
            }
            
            .btn:hover {
                transform: translateY(-2px);
            }
            
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 10px;
                display: none;
            }
            
            .result.fake {
                background: #ffebee;
                border-left: 5px solid #f44336;
            }
            
            .result.real {
                background: #e8f5e8;
                border-left: 5px solid #4caf50;
            }
            
            .result h3 {
                margin-bottom: 15px;
                font-size: 1.5em;
            }
            
            .result.fake h3 {
                color: #f44336;
            }
            
            .result.real h3 {
                color: #4caf50;
            }
            
            .confidence-bar {
                background: #eee;
                border-radius: 10px;
                height: 20px;
                margin: 10px 0;
                overflow: hidden;
            }
            
            .confidence-fill {
                height: 100%;
                border-radius: 10px;
                transition: width 0.5s ease;
            }
            
            .confidence-fill.fake {
                background: #f44336;
            }
            
            .confidence-fill.real {
                background: #4caf50;
            }
            
            .loading {
                text-align: center;
                padding: 20px;
                display: none;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error {
                background: #ffebee;
                border: 1px solid #f44336;
                color: #f44336;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                display: none;
            }
            
            .probabilities {
                display: flex;
                justify-content: space-between;
                margin-top: 15px;
            }
            
            .prob-item {
                text-align: center;
                flex: 1;
                padding: 10px;
                border-radius: 8px;
                margin: 0 5px;
            }
            
            .prob-item.fake {
                background: rgba(244, 67, 54, 0.1);
            }
            
            .prob-item.real {
                background: rgba(76, 175, 80, 0.1);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Fake News Classifier</h1>
                <p>AI-powered detection of fake and real news</p>
            </div>
            
            <div class="content">
                <div class="tabs">
                    <button class="tab-button active" onclick="switchTab('text')">Analyze Text</button>
                    <button class="tab-button" onclick="switchTab('url')">Analyze URL</button>
                </div>
                
                <div id="text-tab" class="tab-content active">
                    <div class="form-group">
                        <label for="text-input">Enter news text to analyze:</label>
                        <textarea id="text-input" rows="6" placeholder="Paste your news article text here..."></textarea>
                    </div>
                    <button class="btn" onclick="analyzeText()">Analyze Text</button>
                </div>
                
                <div id="url-tab" class="tab-content">
                    <div class="form-group">
                        <label for="url-input">Enter news article URL:</label>
                        <input type="url" id="url-input" placeholder="https://example.com/news-article">
                    </div>
                    <button class="btn" onclick="analyzeURL()">Analyze URL</button>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Analyzing content...</p>
                </div>
                
                <div class="error" id="error"></div>
                
                <div class="result" id="result">
                    <h3 id="result-title"></h3>
                    <p><strong>Confidence:</strong> <span id="confidence"></span></p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="confidence-fill"></div>
                    </div>
                    <div class="probabilities">
                        <div class="prob-item fake">
                            <strong>Fake</strong><br>
                            <span id="prob-fake"></span>
                        </div>
                        <div class="prob-item real">
                            <strong>Real</strong><br>
                            <span id="prob-real"></span>
                        </div>
                    </div>
                    <p><strong>Text analyzed:</strong></p>
                    <p style="font-style: italic; margin-top: 10px;" id="analyzed-text"></p>
                </div>
            </div>
        </div>
        
        <script>
            function switchTab(tabName) {
                // Hide all tab contents
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Remove active class from all buttons
                document.querySelectorAll('.tab-button').forEach(btn => {
                    btn.classList.remove('active');
                });
                
                // Show selected tab and mark button as active
                document.getElementById(tabName + '-tab').classList.add('active');
                event.target.classList.add('active');
                
                // Hide result and error
                hideResult();
                hideError();
            }
            
            function showLoading() {
                document.getElementById('loading').style.display = 'block';
                document.querySelectorAll('.btn').forEach(btn => btn.disabled = true);
            }
            
            function hideLoading() {
                document.getElementById('loading').style.display = 'none';
                document.querySelectorAll('.btn').forEach(btn => btn.disabled = false);
            }
            
            function showResult(data) {
                const result = document.getElementById('result');
                const resultTitle = document.getElementById('result-title');
                const confidence = document.getElementById('confidence');
                const confidenceFill = document.getElementById('confidence-fill');
                const probFake = document.getElementById('prob-fake');
                const probReal = document.getElementById('prob-real');
                const analyzedText = document.getElementById('analyzed-text');
                
                const isFake = data.prediction === 'FAKE';
                
                result.className = 'result ' + (isFake ? 'fake' : 'real');
                resultTitle.textContent = isFake ? '🚨 FAKE NEWS DETECTED' : '✅ REAL NEWS DETECTED';
                confidence.textContent = Math.round(data.confidence * 100) + '%';
                
                confidenceFill.className = 'confidence-fill ' + (isFake ? 'fake' : 'real');
                confidenceFill.style.width = Math.round(data.confidence * 100) + '%';
                
                probFake.textContent = Math.round(data.probability_fake * 100) + '%';
                probReal.textContent = Math.round(data.probability_real * 100) + '%';
                analyzedText.textContent = data.text_analyzed;
                
                result.style.display = 'block';
            }
            
            function hideResult() {
                document.getElementById('result').style.display = 'none';
            }
            
            function showError(message) {
                const error = document.getElementById('error');
                error.textContent = message;
                error.style.display = 'block';
            }
            
            function hideError() {
                document.getElementById('error').style.display = 'none';
            }
            
            async function analyzeText() {
                const text = document.getElementById('text-input').value.trim();
                
                if (!text) {
                    showError('Please enter some text to analyze.');
                    return;
                }
                
                hideError();
                hideResult();
                showLoading();
                
                try {
                    const response = await fetch('/predict/text', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: text })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        showResult(data);
                    } else {
                        showError(data.detail || 'An error occurred while analyzing the text.');
                    }
                } catch (error) {
                    showError('Network error: ' + error.message);
                } finally {
                    hideLoading();
                }
            }
            
            async function analyzeURL() {
                const url = document.getElementById('url-input').value.trim();
                
                if (!url) {
                    showError('Please enter a URL to analyze.');
                    return;
                }
                
                hideError();
                hideResult();
                showLoading();
                
                try {
                    const response = await fetch('/predict/url', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ url: url })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        showResult(data);
                    } else {
                        showError(data.detail || 'An error occurred while analyzing the URL.');
                    }
                } catch (error) {
                    showError('Network error: ' + error.message);
                } finally {
                    hideLoading();
                }
            }
            
            // Allow Enter key to submit in text area
            document.getElementById('text-input').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) {
                    analyzeText();
                }
            });
            
            // Allow Enter key to submit in URL input
            document.getElementById('url-input').addEventListener('keydown', function(e) {
                if (e.key === 'Enter') {
                    analyzeURL();
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text_endpoint(request: TextRequest):
    """Predict if text is fake or real news"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    return predict_text(request.text)

@app.post("/predict/url", response_model=PredictionResponse)
async def predict_url_endpoint(request: URLRequest):
    """Extract text from URL and predict if it's fake or real news"""
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    
    # Extract text from URL
    text = extract_text_from_url(request.url)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the URL")
    
    return predict_text(text)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
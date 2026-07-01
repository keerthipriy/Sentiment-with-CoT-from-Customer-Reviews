# Gen_ai_task1
Sentiment‑with‑CoT from Customer Reviews
# Customer Review Sentiment Analyzer & Hyperparameter Analysis

This repository contains two AI-powered applications built using Streamlit and Groq LLM.

## Project 1: Customer Review Sentiment Analyzer

This application analyzes customer reviews and predicts sentiment.

### Features
- Takes customer review as input
- Detects sentiment (Positive, Negative, Neutral)
- Gives reason for sentiment
- Uses Groq API

### Technologies Used
- Python
- Streamlit
- Groq
- Dotenv
### Project Structure
```bash
reviews/
├── app1.py
├── reviews.csv
├── .env
├── requirements.txt
```
### Installation
```bash
pip install -r requirements.txt
```
### Run Project
```bash
python -m streamlit run app1.py
```
## Project 2: Tree-of-Thought Hyperparameter Analysis

This application analyzes machine learning hyperparameter tuning results.

### Features
- Upload CSV file
- Preview dataset
- Analyze top configurations
- Compare performance
- Detect overfitting
- Recommend best configuration

### Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- Groq

### Project Structure
```bash
reviews/
├── app2.py
├── ml.py
├── hyperparameter_results.csv
├── .env
├── requirements.txt
```

### Run Project
```bash
python -m streamlit run app2.py
```
## Environment Setup

Create a `.env` file:

```env
groq_api_key=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
```
## Requirements

Install all dependencies:

```bash
pip install streamlit groq python-dotenv pandas scikit-learn
```

## Output

### Sentiment Analyzer
- Sentiment Label
- Reason

### Hyperparameter Analysis
- Best Configuration
- Bias-Variance Analysis
- Overfitting Detection
- Final Recommendation

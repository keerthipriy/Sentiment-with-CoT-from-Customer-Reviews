import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

client = Groq(api_key=api_key)

def analyze_review(review):
    prompt = f"""
    Analyze the following customer review.

    Review:
    {review}

    Return:
    1. Sentiment (Positive, Negative, Neutral)
    2. Reason
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

st.title("Review Sentiment Analyzer using Groq")

review = st.text_area("Enter Review")

if st.button("Analyze"):
    if review:
        result = analyze_review(review)
        st.write(result)
    else:
        st.warning("Please enter a review") 
import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

st.title("Tree-of-Thought Hyperparameter Analysis")

uploaded_file = st.file_uploader(
    "Upload hyperparameter_results.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write("Shape:", df.shape)
    st.dataframe(df)

    if st.button("Analyze Hyperparameters"):

        if "mean_test_score" in df.columns:
            top_configs = df.sort_values(
                by="mean_test_score",
                ascending=False
            ).head(5)
        else:
            top_configs = df.head(5)

        data_text = top_configs.to_string(index=False)

        prompt = f"""
You are an expert Machine Learning Engineer.

Analyze the following hyperparameter tuning results.

Tasks:
1. Group configurations into:
   - Highly Promising
   - Moderately Promising
   - Weak
2. Evaluate bias-variance tradeoffs.
3. Compare validation performance, training time, and overfitting gap.
4. Identify which configuration generalizes best.
5. Select ONE best configuration.
6. Explain why it is the best choice.

Hyperparameter Results:

{data_text}

Provide detailed reasoning.
"""

        try:
            with st.spinner("Analyzing..."):

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )

                result = response.choices[0].message.content

                st.subheader("Tree-of-Thought Analysis")
                st.write(result)

        except Exception as e:
            st.error(f"Error: {e}")
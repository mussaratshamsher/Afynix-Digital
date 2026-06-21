import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def generate_insights(dataset_summary, model_results, top_features):
    """
    Generates business insights using Groq LLM.
    """
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        return "Groq API Key not found. Please set GROQ_API_KEY in .env file."

    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are a Real Estate Strategy Consultant for a startup in Karachi, Pakistan.
    Based on the following data analysis, provide concise business insights.
    
    Dataset Summary:
    {dataset_summary}
    
    Model Performance (R2, MAE, RMSE):
    {model_results}
    
    Top Influencing Features:
    {top_features}
    
    Provide:
    1. A summary of the Karachi real estate market based on this data.
    2. Top 3 pricing drivers in Karachi.
    3. Risks identified in the current pricing model.
    4. Two strategic business recommendations for the startup.
    5. A concise conclusion.
    
    Keep the tone professional and the insights actionable.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"

if __name__ == "__main__":
    # Example usage
    print(generate_insights("1200 records across 8 locations in Karachi", "RF R2: 0.85", "Area, Location_DHA, Bedrooms"))

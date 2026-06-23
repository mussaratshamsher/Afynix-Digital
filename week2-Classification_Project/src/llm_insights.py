from __future__ import annotations

import os


def generate_insights(summary_text: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not set; skipping LLM narrative generation."

    from groq import Groq

    client = Groq(api_key=api_key)

    prompt = (
        "You are a data science communicator. Convert the following model evaluation summary into a concise business-ready narrative. "
        "Include: what it predicts, why it matters, key strengths (precision/recall/PR-AUC), and recommended next actions.\n\n"
        f"SUMMARY:\n{summary_text}\n"
    )

    resp = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    return resp.choices[0].message.content


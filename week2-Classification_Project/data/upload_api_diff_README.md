# Upload vs API comparison CSVs (Week 2)

This project lets you test the **same input data** two ways:
1) **Streamlit upload path**: your CSV is read by Streamlit and then converted into JSON payload (`{"data": [...]}`) and sent to the backend.
2) **API request path**: you manually send JSON to `POST /predict_batch`.

## What file to upload in the Streamlit UI?
Use this CSV:
- `week2-Classification_Project/data/upload_api_diff_demo.csv`

Notes:
- It is designed to match the backend’s expected input columns used by `build_features`.
- Include the `Churn` column to observe that the frontend removes it before calling the API.

## JSON payload used by the API
The frontend uses the CSV content like this:
- Reads CSV -> `df`
- Drops `Churn` column if present
- Converts to records:
  - `records = df_to_predict.to_dict(orient="records")`
  - Sends:
    - `POST {API_URL}/predict_batch`
    - body: `{ "data": records }`

You can generate a ready-to-send payload using:
- `week2-Classification_Project/data/api_upload_demo.py`

That script outputs:
- `demo_upload.csv`
- `demo_payload.json` (ready for `POST /predict_batch`)

## Expected behavior differences you should see
- If you upload a CSV that matches expected columns:
  - **Streamlit (Remote API mode)** should return predictions and show them in the table/bar chart.
- If your CSV is missing required columns or has wrong column names:
  - API call should fail (and Streamlit will show an error).

## Quick manual test (API only)
1) Open `demo_payload.json` in the browser/editor.
2) `POST` it to:
   - `POST https://mussarat123shamsher-classification-project.hf.space/predict_batch`
   - Content-Type: `application/json`
3) Compare returned JSON:
   - `churn_probabilities`
   - `predicted_churns`
   - `risk_tiers`


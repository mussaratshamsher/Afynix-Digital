rea# TODO - Week 2 Classification Project

## Step 1: Fix API model loading crash
- [ ] Confirm `week2-Classification_Project/models/churn_model.joblib` exists (currently missing).
- [ ] Train the model to generate the artifact (preferred).

## Step 2: Make API robust to missing model (optional but recommended)
- [ ] Update `app/api.py` to avoid crashing on import when the model file is missing.
- [ ] Return a clear error response from `/predict` if the model isn’t loaded.

## Step 3: Verify end-to-end flow
- [ ] Run `python week2-Classification_Project/src/train.py --data_path ... --output_dir week2-Classification_Project`
- [ ] Start API with `uvicorn week2-Classification_Project.app.api:app --reload --port 8000`
- [ ] Call `POST /predict` and confirm response.

## Step 4: Remote API testing with Streamlit
- [ ] Set default backend URL in `app/dashboard.py` to `https://mussarat123shamsher-classification-project.hf.space`.
- [ ] Ensure Streamlit uses that URL for `/` and `/predict_batch`.

## Step 5: Upload behavior comparison helper
- [x] Added `data/api_upload_demo.py` which generates:
  - `demo_upload.csv`
  - `demo_payload.json` (same data formatted for POST /predict_batch)

https://mussarat123shamsher-classification-project.hf.space


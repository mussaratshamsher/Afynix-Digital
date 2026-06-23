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


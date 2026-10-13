# ChurnPredict — AI-Based Customer Churn Prediction

ChurnPredict is an end-to-end machine-learning web application for exploring customer churn risk from labelled customer datasets. Users can upload a dataset, run a dataset-specific analysis, review aggregate risk segments, and inspect a limited set of customer-level predictions in an interactive dashboard.

**Live Demo:** https://churnpredict-public.onrender.com  
**GitHub Repository:** https://github.com/Mohammad-Ali-SK/churnpredict_public_launch

## Features

- Dataset-specific churn analysis without replacing the shared saved model.
- Chunked CSV processing designed for up to **200,000 rows and 100 columns**, subject to upload limits.
- Training sample bounded to 5,000 labelled rows for large CSVs.
- Full-dataset scoring for aggregate risk counts, with at most 1,000 detailed predictions returned by the large-CSV path.
- Numeric preprocessing with median imputation and standard scaling.
- Categorical preprocessing with most-frequent imputation and one-hot encoding.
- Logistic Regression for large-dataset evaluation; smaller datasets can compare Logistic Regression and Random Forest.
- Interactive dashboard with evaluation metrics, feature importance, and churn-risk summaries.

## Technology Stack

**Frontend:** React, TypeScript, Tailwind CSS, Recharts  
**Backend:** Python, FastAPI, Uvicorn  
**Machine Learning:** Pandas, NumPy, scikit-learn  
**Deployment:** Render

## How It Works

1. Upload a labelled customer dataset.
2. Validate and process the uploaded data.
3. Build a bounded training sample for large CSV files.
4. Train and evaluate the selected machine-learning pipeline.
5. Score labelled rows in chunks and calculate aggregate churn-risk counts.
6. Display the analysis results in the interactive dashboard.

The dataset-specific analysis does not save or replace the shared production model.

## Large-Dataset Validation

The 200,000-row × 100-column synthetic dataset test completed successfully in Codespaces. The test verified the full-dataset scan, a training sample of at most 5,000 rows, and a maximum of 1,000 returned detailed predictions.

**Important:** This validates the processing path, not real-world predictive accuracy. Model quality should be evaluated using representative, unseen customer data.

## Local Setup

```bash
git clone https://github.com/Mohammad-Ali-SK/churnpredict_public_launch.git
cd churnpredict_public_launch
python -m venv .venv
```

Activate the environment, then install and run:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 in your browser.

## Future Improvements

- Add automated CI tests.
- Evaluate models on real-world datasets.
- Add model monitoring and drift detection.
- Improve production monitoring and large-workload handling.

**Author:** Mohammad Ali Sk

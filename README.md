# Sales Recommendation & Next-Best-Product Engine

A senior-level full-stack machine learning project that recommends which products a business should offer to each customer to improve conversion, sales, and average order value. The project includes synthetic retail data generation, feature engineering, supervised ML model training, model artifact persistence, a FastAPI prediction service, automated tests, and a React TypeScript analytics dashboard.

## Features

- Synthetic customer, transaction, browsing, and product catalog generation
- Supervised ML target: whether a customer buys a candidate product in the next 45 days
- Clear feature matrix `X` and target vector `y`
- Baseline model: `RandomForestClassifier`
- Stronger model: `GradientBoostingClassifier`
- Recommendation score that includes supervised ML purchase probability
- Supporting similarity and business-value ranking signals
- Precision@K, Recall@K, NDCG@K, ROC AUC, Average Precision, Precision, Recall, and F1 evaluation
- Saved model artifacts with `joblib`
- FastAPI runtime loading of the trained model
- React TypeScript dashboard showing recommended products, ML scores, and expected business impact
- Backend tests for feature engineering and API predictions

## Tech Stack

### Machine Learning

- Python
- pandas
- NumPy
- scikit-learn
- joblib

### Backend

- FastAPI
- Uvicorn
- pytest

### Frontend

- React
- TypeScript
- Vite
- Recharts
- Lucide React icons

### Database Readiness

- The demo uses generated retail data saved as local ML artifacts for reproducible training.
- The service boundaries are ready to replace the artifact-backed data source with PostgreSQL tables for customers, products, events, and transactions.

## Project Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── ml/
│   │       ├── artifacts/
│   │       ├── features.py
│   │       └── outcome_model.py
│   ├── scripts/
│   │   └── train_model.py
│   ├── tests/
│   │   ├── test_api.py
│   │   └── test_features.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   ├── main.tsx
│   │   └── styles.css
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── .gitignore
└── README.md
```

## ML Design

### Dataset

The training script generates a deterministic retail dataset with:

- Customers: segment, region, loyalty tier, age, discount sensitivity, email engagement
- Products: category, brand, price, margin percentage, rating
- Transactions: customer-product purchases with quantity, discount, date, and revenue
- Browsing events: views, wishlists, add-to-cart events, dwell time, and dates

### Features `X`

The model uses customer, product, purchase history, browsing, similarity, and business-value features:

- Customer signals: segment, region, loyalty tier, age, LTV, order count, average order value, discount sensitivity, email engagement
- Product signals: category, brand, price, margin, rating, popularity, repeat rate, revenue share
- Purchase history: prior product purchases, quantity, revenue, recency, category purchase count, category affinity
- Browsing behavior: views, add-to-cart count, dwell seconds, browse recency
- Ranking support: price-fit score, item similarity score, business value score

### Target `y`

`target_purchase = 1` when the customer purchases the candidate product during the 45-day label window.

`target_purchase = 0` otherwise.

### Models Trained

- `RandomForestClassifier baseline`
- `GradientBoostingClassifier stronger_model`

Latest training run:

| Model | ROC AUC | Average Precision | Precision | Recall | F1 | Precision@5 | Recall@5 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RandomForestClassifier baseline | 0.6541 | 0.1524 | 0.1918 | 0.3022 | 0.2346 | 0.1663 | 0.5982 | 0.4473 |
| GradientBoostingClassifier stronger_model | 0.6719 | 0.1633 | 0.3611 | 0.0935 | 0.1486 | 0.1537 | 0.5316 | 0.4310 |

Selected runtime model: `GradientBoostingClassifier stronger_model`.

The selected model is saved to:

```text
backend/app/ml/artifacts/next_best_product_model.joblib
```

Metrics are saved to:

```text
backend/app/ml/artifacts/metrics.json
```

### Final Recommendation Score

The final ranking score is:

```text
0.75 * supervised ML purchase probability
+ 0.15 * item similarity support score
+ 0.10 * normalized business value score
```

This means content similarity supports the ranking, but the supervised ML model drives the recommendation.

## Local Setup

### 1. Backend

```bash
cd backend
python -m pip install -r requirements.txt
python -m backend.scripts.train_model
uvicorn app.main:app --reload
```

Backend API:

```text
http://localhost:8000/api/health
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend dashboard:

```text
http://localhost:5173
```

## API Endpoints

```text
GET /api/health
GET /api/customers
GET /api/products
GET /api/ml/features
GET /api/ml/metrics
GET /api/recommendations/{customer_id}?top_k=8
```

The recommendation response includes:

- Customer profile
- Model name
- Model target explanation
- Feature groups
- Recommended products
- Model-predicted purchase probability
- Similarity score
- Final ranking score
- Expected incremental revenue

## Retraining

Run:

```bash
python -m backend.scripts.train_model
```

This regenerates:

- Synthetic retail dataset
- Feature matrix
- Baseline model
- Stronger model
- Classification metrics
- Ranking metrics
- Saved model artifact
- Metrics JSON artifact

## Testing

### Backend Tests

```bash
pytest backend/tests
```

### Frontend Build

```bash
cd frontend
npm run build
```

## Repository

1. [brianmahlatini](https://github.com/brianmahlatini)
2. [**Sales-Recommendation-Next-Best-Product-Engine**](https://github.com/brianmahlatini/Sales-Recommendation-Next-Best-Product-Engine)

## Repository Navigation

- [**Code**](https://github.com/brianmahlatini/Sales-Recommendation-Next-Best-Product-Engine)
- [Issues](https://github.com/brianmahlatini/Sales-Recommendation-Next-Best-Product-Engine/issues)
- [Pull requests](https://github.com/brianmahlatini/Sales-Recommendation-Next-Best-Product-Engine/pulls)
- [Actions](https://github.com/brianmahlatini/Sales-Recommendation-Next-Best-Product-Engine/actions)
- [Projects](https://github.com/brianmahlatini/Sales-Recommendation-Next-Best-Product-Engine/projects)
- [Wiki](https://github.com/brianmahlatini/Sales-Recommendation-Next-Best-Product-Engine/wiki)
- [Security and quality](https://github.com/brianmahlatini/Sales-Recommendation-Next-Best-Product-Engine/security)
- [Insights](https://github.com/brianmahlatini/Sales-Recommendation-Next-Best-Product-Engine/pulse)
- [Settings](https://github.com/brianmahlatini/Sales-Recommendation-Next-Best-Product-Engine/settings)

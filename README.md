# Retail Demand Forecasting and Inventory Optimization

An end-to-end machine learning project that forecasts retail demand using XGBoost and converts those forecasts into inventory replenishment decisions using a Periodic Review (Order-Up-To) inventory policy.

Instead of stopping at demand prediction, this project shows how forecasting can be directly applied to inventory planning and operational decision-making.

---

# Overview

Accurate demand forecasting is essential for effective inventory management. Poor forecasts can lead to stockouts or excess inventory, both of which increase costs and reduce efficiency.

This project builds a complete pipeline using historical Walmart sales data. It combines demand forecasting with an inventory simulation module that generates reorder quantities, safety stock levels, and key inventory performance metrics.

The workflow is outlined below.

```text
Historical Sales
        │
        ▼
Data Loading & Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Demand Forecasting (XGBoost)
        │
        ▼
Inventory Optimization
        │
        ▼
Business KPIs & Reorder Recommendations
```

---

# Dataset

The project uses the Walmart Recruiting: Store Sales Forecasting dataset from Kaggle.

Files used:

* `train.csv`
* `features.csv`
* `stores.csv`

The dataset includes weekly sales data across multiple stores and departments, along with economic indicators, promotional markdowns, holidays, and store-level information.

---

# Features

### Demand Forecasting

* Time-aware train/test split
* XGBoost regression model
* Lag features (1, 4, 8, and 12 weeks)
* Rolling mean features
* Calendar-based features
* Promotional markdown features
* Store type encoding
* Feature importance analysis

### Inventory Optimization

* Periodic Review (Order-Up-To) inventory policy
* Safety stock calculation
* Lead time simulation
* Pipeline inventory tracking
* Dynamic reorder quantity generation
* Stockout simulation
* Holding and ordering cost estimation
* Inventory performance metrics

---

# Project Structure

```text
RetailDemandForecasting/
│
├── data/
├── models/
├── notebooks/
│   ├── 01_demand_forecasting.ipynb
│   └── 02_inventory_optimization.ipynb
├── results/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── inventory.py
│   ├── metrics.py
│   └── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SciPy
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook
* Git & GitHub

---

# Model Performance

| Metric   | Value       |
| -------- | ----------- |
| MAE      | **1454.25** |
| RMSE     | **3101.81** |
| R² Score | **0.9804**  |
| WMAPE    | **9.23%**   |

The model uses a chronological train-test split to prevent data leakage and better reflect real-world forecasting conditions.

## Installation

Clone the repository:

```bash
git clone https://github.com/brishty-xo/supply-chain-demand-forecasting.git
cd supply-chain-demand-forecasting
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run Demand Forecasting

Open and execute:

```text
notebooks/01_demand_forecasting.ipynb
```

This notebook:

- Loads and preprocesses the Walmart dataset
- Performs feature engineering
- Trains the XGBoost forecasting model
- Evaluates forecasting performance

### Run Inventory Optimization

Open and execute:

```text
notebooks/02_inventory_optimization.ipynb
```

This notebook:

- Uses forecasted demand
- Simulates inventory dynamics
- Generates reorder recommendations
- Computes inventory KPIs

## Key Results

- Achieved a **WMAPE of 9.23%** on the forecasting task.
- Implemented leakage-free time-series feature engineering using lag and rolling statistics.
- Simulated a Periodic Review (Order-Up-To) inventory policy using forecasted demand.
- Generated inventory metrics including fill rate, stockout rate, holding cost, and ordering cost.

## Future Improvements

Possible extensions include:

- Hyperparameter tuning using Optuna
- Multi-step demand forecasting
- Probabilistic demand forecasting
- Dynamic lead times
- Economic Order Quantity (EOQ) integration
- Containerized deployment using Docker

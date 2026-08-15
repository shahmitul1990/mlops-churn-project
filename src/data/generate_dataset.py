import numpy as np
import pandas as pd


np.random.seed(42)

n = 5000

data = pd.DataFrame({
    "customer_id": range(1, n + 1),

    "age": np.random.randint(18, 70, n),

    "tenure_months": np.random.randint(1, 73, n),

    "monthly_charges": np.round(
        np.random.uniform(20, 150, n), 2
    ),

    "contract_type": np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n,
        p=[0.55, 0.25, 0.20]
    ),

    "internet_service": np.random.choice(
        ["DSL", "Fiber optic", "No"],
        n,
        p=[0.30, 0.50, 0.20]
    ),

    "tech_support": np.random.choice(
        ["Yes", "No"],
        n,
        p=[0.35, 0.65]
    ),

    "online_security": np.random.choice(
        ["Yes", "No"],
        n,
        p=[0.40, 0.60]
    ),

    "payment_method": np.random.choice(
        ["Credit card", "Bank transfer", "Electronic check"],
        n,
        p=[0.35, 0.30, 0.35]
    ),

    "customer_service_calls": np.random.poisson(
        2, n
    )
})


churn_score = (
    0.8
    + 0.9 * (data["contract_type"] == "Month-to-month")
    + 0.6 * (data["internet_service"] == "Fiber optic")
    + 0.5 * (data["tech_support"] == "No")
    + 0.4 * (data["online_security"] == "No")
    + 0.15 * data["customer_service_calls"]
    - 0.025 * data["tenure_months"]
    - 0.01 * data["age"]
)


probability = 1 / (1 + np.exp(-churn_score))

data["churn"] = np.random.binomial(
    1,
    probability
)


output_path = "data/raw/customer_churn.csv"

data.to_csv(
    output_path,
    index=False
)

print(f"Dataset created successfully: {output_path}")
print(f"Number of rows: {len(data)}")
print(f"Number of columns: {len(data.columns)}")
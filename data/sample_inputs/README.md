# Sample Prediction Inputs & API Testing Guide

This directory provides standardized, validated sample customer payloads for testing the **Customer Churn Intelligence & Retention Optimization API**.

---

## 📁 Sample File Catalog

1. **`single_customer_high_risk.json`**:
   - Profile: 2 months tenure, Month-to-month contract, Fiber optic, Electronic check, $95.50/mo.
   - Expected Output: Calibrated Probability $\approx 0.73 - 0.78$, Risk Tier: `High Risk`, Recommendation: `Proactive Retention Outreach`.

2. **`single_customer_low_risk.json`**:
   - Profile: 62 months tenure, Two-year contract, DSL, Automatic credit card billing, Tech Support.
   - Expected Output: Calibrated Probability $\approx 0.04 - 0.08$, Risk Tier: `Low Risk`, Recommendation: `Loyal / Monitor`.

3. **`single_customer_borderline.json`**:
   - Profile: 12 months tenure, Month-to-month contract, Fiber optic, Bank transfer, Tech support.
   - Expected Output: Calibrated Probability $\approx 0.50 - 0.56$ (near optimal cutoff $\tau^* = 0.540$).

4. **`batch_customers_request.json`**:
   - Complete batch inference payload containing 4 accounts across diverse customer segments.

---

## 🚀 Testing Commands

### 1. Test Single Prediction via cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d @single_customer_high_risk.json
```

### 2. Test Batch Scoring via cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d @batch_customers_request.json
```

### 3. Test via Python (`httpx`)
```python
import json
import httpx

with open("single_customer_high_risk.json") as f:
    payload = json.load(f)

response = httpx.post("http://127.0.0.1:8000/predict", json=payload)
print("Status:", response.status_code)
print(json.dumps(response.json(), indent=2))
```

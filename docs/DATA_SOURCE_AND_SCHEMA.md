# IBM Telco Customer Churn: Data Source, Schema & Data Dictionary

## 1. Source Provenance & Lineage

- **Dataset Name**: IBM Telco Customer Churn (Telecommunications Retention Dataset)
- **Originator / Publisher**: IBM Digital Analytics / IBM Developer Ecosystem (ICP4D & Watson Studio)
- **Canonical Remote URL**:
  `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Secondary Mirror**: Kaggle Telco Customer Churn dataset (`blastchar/telco-customer-churn`)
- **License / Terms**: Apache License 2.0 / Open Data Commons Public Domain Dedication
- **Primary Business Objective**: Predict customer churn, identify retention drivers, simulate targeted marketing campaigns, and maximize customer lifetime value (CLV).

---

## 2. Dataset Shape & Volume

| Metric | Value |
| :--- | :--- |
| **Total Record Count (Rows)** | `7,043` customers |
| **Total Attributes (Columns)** | `21` features (1 identifier, 19 predictive features, 1 target label) |
| **Primary Identifier** | `customerID` (unique alphanumeric format: `XXXX-XXXXX`) |
| **Unique Identifier Count** | `7,043` (100% unique, 0 duplicate keys) |
| **Target Column** | `Churn` (`'No'`: 5,174 [73.46%], `'Yes'`: 1,869 [26.54%]) |
| **Class Imbalance Ratio** | ~2.77:1 (Negative:Positive) |

---

## 3. Verified Raw Column Schema & Data Dictionary

| # | Raw Column Name | Standardized Snake_Case | Raw Dtype | Logical Type | Domain / Valid Values | Description |
| :-: | :--- | :--- | :-: | :--- | :--- | :--- |
| 1 | `customerID` | `customer_id` | `object` / `string` | Identifier | Regex: `^\d{4}-[A-Z]{5}$` | Unique customer alphanumeric account ID |
| 2 | `gender` | `gender` | `object` | Categorical | `Male`, `Female` | Customer biological gender |
| 3 | `SeniorCitizen` | `senior_citizen` | `int64` | Binary Flag | `0` (No), `1` (Yes) | Whether the customer is aged 65 or older |
| 4 | `Partner` | `partner` | `object` | Binary Cat | `Yes`, `No` | Whether the customer has a domestic partner |
| 5 | `Dependents` | `dependents` | `object` | Binary Cat | `Yes`, `No` | Whether the customer lives with dependents (children/parents) |
| 6 | `tenure` | `tenure` | `int64` | Integer | `0` to `72` (months) | Number of months the customer has stayed with the company |
| 7 | `PhoneService` | `phone_service` | `object` | Binary Cat | `Yes`, `No` | Whether the customer subscribes to fixed home phone service |
| 8 | `MultipleLines` | `multiple_lines` | `object` | Categorical | `No`, `Yes`, `No phone service` | Whether the customer has multiple telephone lines |
| 9 | `InternetService` | `internet_service` | `object` | Categorical | `DSL`, `Fiber optic`, `No` | Customer’s internet service connectivity provider |
| 10 | `OnlineSecurity` | `online_security` | `object` | Categorical | `No`, `Yes`, `No internet service` | Subscription to cybersecurity/malware protection add-on |
| 11 | `OnlineBackup` | `online_backup` | `object` | Categorical | `No`, `Yes`, `No internet service` | Subscription to cloud-based automated file backup add-on |
| 12 | `DeviceProtection` | `device_protection` | `object` | Categorical | `No`, `Yes`, `No internet service` | Hardware warranty and protection plan for connected devices |
| 13 | `TechSupport` | `tech_support` | `object` | Categorical | `No`, `Yes`, `No internet service` | Access to priority technical support specialist hotline |
| 14 | `StreamingTV` | `streaming_tv` | `object` | Categorical | `No`, `Yes`, `No internet service` | TV streaming package from a 3rd party content provider |
| 15 | `StreamingMovies` | `streaming_movies` | `object` | Categorical | `No`, `Yes`, `No internet service` | Movie streaming package from a 3rd party content provider |
| 16 | `Contract` | `contract` | `object` | Categorical | `Month-to-month`, `One year`, `Two year` | Contractual commitment term structure |
| 17 | `PaperlessBilling` | `paperless_billing` | `object` | Binary Cat | `Yes`, `No` | Whether customer receives digital electronic invoicing |
| 18 | `PaymentMethod` | `payment_method` | `object` | Categorical | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | Payment disbursement vehicle |
| 19 | `MonthlyCharges` | `monthly_charges` | `float64` | Continuous | `18.25` to `118.75` (USD) | Amount charged to customer on monthly recurring cycle |
| 20 | `TotalCharges` | `total_charges` | `object` *(contains spaces)* | Continuous | `18.80` to `8684.80` (USD) | Cumulative charges billed over customer lifecycle |
| 21 | `Churn` | `churn` | `object` | Binary Target | `Yes`, `No` | Target variable: Did customer churn in the last month? |

---

## 4. Known Data Quality Quirks & Engineering Resolutions

### 4.1 Whitespace in `TotalCharges` (The 11 Zero-Tenure Records)
- **Anomaly**: Exactly 11 rows in the raw CSV have `TotalCharges == " "` (a single whitespace character) instead of a numeric value. This causes standard CSV parsers to treat the entire column as `object` (string).
- **Root Cause Analysis**: All 11 records have `tenure == 0`. These represent newly onboarded customers who signed contracts during the current billing cycle and have not yet completed their first monthly billing cycle.
- **Engineered Resolution**:
  - Convert `TotalCharges` using `pd.to_numeric(df["TotalCharges"], errors="coerce")`.
  - For rows where `tenure == 0`, impute `total_charges = 0.0`.
  - Validate that after imputation, exactly 0 nulls remain and all `total_charges >= 0.0`.

### 4.2 Dependent Service Categories
- When `PhoneService == 'No'`, `MultipleLines` is consistently `'No phone service'`.
- When `InternetService == 'No'`, all six internet add-on columns (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) take the value `'No internet service'`.
- Our validation suite enforces these hierarchical relationship constraints to ensure data consistency.

### 4.3 Standardized Relational Storage Model (3NF)
In the raw CSV, demographic data, subscribed services, and contract billing terms are denormalized into a single 21-column flat table. For production relational storage and SQL analytics, we normalize this into three relational tables linked by `customer_id`:
1. `customers`: Demographics (`gender`, `senior_citizen`, `partner`, `dependents`)
2. `customer_services`: Subscriptions (`phone_service`, `multiple_lines`, `internet_service`, add-ons)
3. `customer_contracts`: Terms & revenue (`tenure`, `contract`, `paperless_billing`, `payment_method`, `monthly_charges`, `total_charges`, `churn`)
4. `churn_predictions`: Downstream model inference audit table.

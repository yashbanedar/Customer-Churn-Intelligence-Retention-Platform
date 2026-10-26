-- ==============================================================================
-- Customer Churn Intelligence & Retention Optimization Platform
-- Production PostgreSQL Analytical Queries (Cohorts, Risk Segments, Revenue at Risk)
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Executive Portfolio KPI Summary: Total Accounts, Churn Rate, MRR at Risk
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(monthly_charges), 2) AS total_mrr,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END), 2) AS mrr_at_risk,
    ROUND(12.0 * SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END), 2) AS annualized_arr_at_risk,
    ROUND(AVG(monthly_charges), 2) AS avg_revenue_per_user_arpu
FROM customer_contracts;

-- ------------------------------------------------------------------------------
-- 2. Contract Commitment & Churn Risk Matrix
-- Shows dramatic retention advantage of 1-year and 2-year commitments vs M2M
-- ------------------------------------------------------------------------------
SELECT 
    contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charge,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END), 2) AS mrr_at_risk
FROM customer_contracts
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- ------------------------------------------------------------------------------
-- 3. Tenure Cohort Retention Decay Curve
-- Measures churn drop-off across customer lifecycle stages
-- ------------------------------------------------------------------------------
SELECT 
    CASE 
        WHEN tenure <= 6 THEN '1. 0 - 6 Months (Critical Onboarding)'
        WHEN tenure <= 12 THEN '2. 7 - 12 Months (First-Year Cliff)'
        WHEN tenure <= 24 THEN '3. 13 - 24 Months (Established)'
        WHEN tenure <= 48 THEN '4. 25 - 48 Months (Mature)'
        ELSE '5. 49+ Months (Long-Term Loyal)'
    END AS tenure_cohort,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_count,
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS cohort_churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_spend,
    ROUND(SUM(total_charges), 2) AS cumulative_revenue_generated
FROM customer_contracts
GROUP BY 1
ORDER BY 1;

-- ------------------------------------------------------------------------------
-- 4. Service Friction Matrix: Fiber Optic vs Tech Support & Online Security
-- Evaluates why Fiber Optic customers churn at higher rates without support
-- ------------------------------------------------------------------------------
SELECT 
    s.internet_service,
    s.tech_support,
    s.online_security,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) AS churn_count,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(c.monthly_charges), 2) AS avg_monthly_charges
FROM customer_services s
JOIN customer_contracts c ON s.customer_id = c.customer_id
GROUP BY s.internet_service, s.tech_support, s.online_security
HAVING COUNT(*) >= 50
ORDER BY churn_rate_pct DESC;

-- ------------------------------------------------------------------------------
-- 5. Payment Disbursement Risk: Electronic Check vs Automated Direct Debit
-- Dissects involuntary and transaction friction churn
-- ------------------------------------------------------------------------------
SELECT 
    c.payment_method,
    c.paperless_billing,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN c.churn = 'Yes' THEN c.monthly_charges ELSE 0 END), 2) AS monthly_revenue_at_risk
FROM customer_contracts c
GROUP BY c.payment_method, c.paperless_billing
ORDER BY churn_rate_pct DESC;

-- ------------------------------------------------------------------------------
-- 6. Senior Citizen Vulnerability & Support Gap Analysis
-- Highlights demographic churn rate when dedicated support is lacking
-- ------------------------------------------------------------------------------
SELECT 
    cu.senior_citizen,
    cu.partner,
    s.tech_support,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN co.churn = 'Yes' THEN 1 ELSE 0 END) AS churned_count,
    ROUND(100.0 * SUM(CASE WHEN co.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(co.monthly_charges), 2) AS avg_monthly_charge
FROM customers cu
JOIN customer_services s ON cu.customer_id = s.customer_id
JOIN customer_contracts co ON cu.customer_id = co.customer_id
GROUP BY cu.senior_citizen, cu.partner, s.tech_support
ORDER BY cu.senior_citizen DESC, churn_rate_pct DESC;

-- ------------------------------------------------------------------------------
-- 7. High-Value At-Risk VIP Segment (Immediate Intervention Candidates)
-- Monthly Charges > $80, Month-to-Month contract, Fiber Optic
-- ------------------------------------------------------------------------------
SELECT 
    co.customer_id,
    co.tenure,
    co.monthly_charges,
    co.total_charges,
    co.payment_method,
    s.internet_service,
    s.tech_support,
    s.online_security
FROM customer_contracts co
JOIN customer_services s ON co.customer_id = s.customer_id
WHERE co.contract = 'Month-to-month'
  AND co.monthly_charges >= 80.00
  AND co.churn = 'Yes'
ORDER BY co.monthly_charges DESC
LIMIT 50;

-- ------------------------------------------------------------------------------
-- 8. Add-On Feature Stacking & Churn Elasticity
-- Calculates how each additional security/backup/streaming add-on mitigates churn
-- ------------------------------------------------------------------------------
WITH add_on_counts AS (
    SELECT 
        customer_id,
        (CASE WHEN online_security = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN online_backup = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN tech_support = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN streaming_tv = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN streaming_movies = 'Yes' THEN 1 ELSE 0 END) AS active_addons_count
    FROM customer_services
    WHERE internet_service != 'No'
)
SELECT 
    a.active_addons_count,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) AS churn_count,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(c.monthly_charges), 2) AS avg_monthly_charges
FROM add_on_counts a
JOIN customer_contracts c ON a.customer_id = c.customer_id
GROUP BY a.active_addons_count
ORDER BY a.active_addons_count ASC;

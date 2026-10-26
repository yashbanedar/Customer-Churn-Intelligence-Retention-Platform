-- ==============================================================================
-- Customer Churn Intelligence & Retention Optimization Platform
-- Relational Database Schema DDL (PostgreSQL 18 & SQLite compatible)
-- ==============================================================================

-- Drop tables if re-creating
DROP TABLE IF EXISTS churn_predictions CASCADE;
DROP TABLE IF EXISTS customer_services CASCADE;
DROP TABLE IF EXISTS customer_contracts CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- 1. Core Customer Demographics Table
CREATE TABLE customers (
    customer_id VARCHAR(32) PRIMARY KEY,
    gender VARCHAR(10) NOT NULL,
    senior_citizen INTEGER NOT NULL CHECK (senior_citizen IN (0, 1)),
    partner VARCHAR(5) NOT NULL CHECK (partner IN ('Yes', 'No')),
    dependents VARCHAR(5) NOT NULL CHECK (dependents IN ('Yes', 'No')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Customer Services Table
CREATE TABLE customer_services (
    customer_id VARCHAR(32) PRIMARY KEY,
    phone_service VARCHAR(5) NOT NULL,
    multiple_lines VARCHAR(25) NOT NULL,
    internet_service VARCHAR(25) NOT NULL,
    online_security VARCHAR(25) NOT NULL,
    online_backup VARCHAR(25) NOT NULL,
    device_protection VARCHAR(25) NOT NULL,
    tech_support VARCHAR(25) NOT NULL,
    streaming_tv VARCHAR(25) NOT NULL,
    streaming_movies VARCHAR(25) NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- 3. Contracts and Billing History Table
CREATE TABLE customer_contracts (
    customer_id VARCHAR(32) PRIMARY KEY,
    tenure INTEGER NOT NULL CHECK (tenure >= 0),
    contract VARCHAR(25) NOT NULL,
    paperless_billing VARCHAR(5) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    monthly_charges NUMERIC(10, 2) NOT NULL CHECK (monthly_charges >= 0),
    total_charges NUMERIC(12, 2) NOT NULL CHECK (total_charges >= 0),
    churn VARCHAR(5) NOT NULL CHECK (churn IN ('Yes', 'No')),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- 4. Churn Predictions & Simulation Audit Table
CREATE TABLE churn_predictions (
    prediction_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(32) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    calibrated_probability NUMERIC(5, 4) NOT NULL,
    risk_tier VARCHAR(20) NOT NULL,
    optimal_threshold NUMERIC(5, 4) NOT NULL,
    recommended_action VARCHAR(50) NOT NULL,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- Indexes for Fast Analytical Querying
CREATE INDEX idx_contracts_churn ON customer_contracts(churn);
CREATE INDEX idx_contracts_tenure ON customer_contracts(tenure);
CREATE INDEX idx_contracts_contract ON customer_contracts(contract);
CREATE INDEX idx_services_internet ON customer_services(internet_service);

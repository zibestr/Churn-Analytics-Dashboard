CREATE TABLE IF NOT EXISTS churn_data (
    customerID VARCHAR(20) PRIMARY KEY NOT NULL,
    gender VARCHAR(10),
    SeniorCitizen INTEGER,
    Partner VARCHAR(5),
    Dependents VARCHAR(5),
    tenure INTEGER,
    PhoneService VARCHAR(5),
    MultipleLines VARCHAR(20),
    InternetService VARCHAR(20),
    OnlineSecurity VARCHAR(20),
    OnlineBackup VARCHAR(20),
    DeviceProtection VARCHAR(20),
    TechSupport VARCHAR(20),
    StreamingTV VARCHAR(20),
    StreamingMovies VARCHAR(20),
    Contract VARCHAR(20),
    PaperlessBilling VARCHAR(5),
    PaymentMethod VARCHAR(30),
    MonthlyCharges DECIMAL(10, 2),
    TotalCharges DECIMAL(10, 2),
    Churn VARCHAR(5)
);

COPY churn_data(
    customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, 
    MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, 
    TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, 
    PaymentMethod, MonthlyCharges, TotalCharges, Churn
)
FROM '/docker-entrypoint-initdb.d/data.csv'
DELIMITER ',' CSV HEADER;

CREATE ROLE readonly_user WITH LOGIN PASSWORD 're@d0nly_p@ssw0rd';
GRANT CONNECT ON DATABASE analytics TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly_user;
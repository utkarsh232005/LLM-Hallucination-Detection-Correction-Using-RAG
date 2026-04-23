CREATE DATABASE IF NOT EXISTS rag_app;
USE rag_app;

CREATE TABLE IF NOT EXISTS chat_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query TEXT,
    llm_response LONGTEXT,
    rag_response LONGTEXT,
    is_hallucinated BOOLEAN,
    hallucination_score FLOAT,
    classification VARCHAR(50),
    sentence_count INT,
    sources_count INT,
    sources JSON,
    model_id VARCHAR(100),
    response_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

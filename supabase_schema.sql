-- Supabase Database Schema for Smart Shop Application
-- Run this SQL in your Supabase SQL Editor to create the required tables

-- Table for interaction logging
CREATE TABLE IF NOT EXISTS interaction_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    original_query TEXT,
    optimized_query TEXT,
    llm_response_text TEXT,
    retrieved_products TEXT,
    selected_products TEXT,
    selected_product_ids_only TEXT,
    full_synthesis_prompt TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for chat history persistence
CREATE TABLE IF NOT EXISTS chat_history (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    products JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(session_id, message_index)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_interaction_logs_timestamp ON interaction_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_session_index ON chat_history(session_id, message_index);

-- Enable Row Level Security (RLS) - adjust policies as needed for your use case
ALTER TABLE interaction_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;

-- Create policies to allow all operations (adjust based on your security requirements)
CREATE POLICY "Allow all operations on interaction_logs" ON interaction_logs
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all operations on chat_history" ON chat_history
    FOR ALL USING (true) WITH CHECK (true);


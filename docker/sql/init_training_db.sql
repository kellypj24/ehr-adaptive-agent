-- Initialize database schema for EHR Agent Training

-- Track training attempts and their outcomes
CREATE TABLE IF NOT EXISTS training_attempts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    task TEXT NOT NULL,                    -- What the agent was trying to accomplish
    code_snippet TEXT NOT NULL,            -- Relevant portion of code (not full file)
    success BOOLEAN NOT NULL,
    error_message TEXT,                    -- Store error if failed
    error_type TEXT,                       -- Categorized error type for pattern matching
    file_location TEXT,                    -- Path to the full implementation
    git_commit_hash TEXT,                  -- For future git integration
    execution_time INTERVAL,
    metadata JSONB                         -- Flexible field for additional data
);

-- Store learned patterns and their evolution
CREATE TABLE IF NOT EXISTS learning_patterns (
    id SERIAL PRIMARY KEY,
    pattern_type TEXT NOT NULL,            -- 'error_solution', 'task_solution', etc.
    pattern_key TEXT NOT NULL,             -- Specific error type or task type
    code_pattern TEXT NOT NULL,            -- The successful pattern/solution
    success_count INTEGER DEFAULT 1,
    failure_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,                        -- Flexible field for additional pattern data
    UNIQUE(pattern_type, pattern_key)      -- Prevent duplicate patterns
);

-- Track tool usage and effectiveness
CREATE TABLE IF NOT EXISTS tool_usage (
    id SERIAL PRIMARY KEY,
    attempt_id INTEGER REFERENCES training_attempts(id),
    tool_name TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    success BOOLEAN NOT NULL,
    execution_time INTERVAL,
    error_message TEXT,
    parameters JSONB,                      -- Store tool parameters
    result_summary TEXT,                   -- Brief description of tool output
    metadata JSONB                         -- Flexible field for future use
);

-- Indexes for performance
CREATE INDEX idx_training_attempts_task ON training_attempts (task);
CREATE INDEX idx_training_attempts_error_type ON training_attempts (error_type);
CREATE INDEX idx_training_attempts_success ON training_attempts (success);
CREATE INDEX idx_learning_patterns_type_key ON learning_patterns (pattern_type, pattern_key);
CREATE INDEX idx_tool_usage_tool_name ON tool_usage (tool_name);
CREATE INDEX idx_tool_usage_success ON tool_usage (success);
CREATE INDEX idx_tool_usage_timestamp ON tool_usage (timestamp);

-- Add helpful documentation
COMMENT ON TABLE training_attempts IS 'Records of each code generation attempt by the AI agent, storing relevant code snippets and outcomes';
COMMENT ON TABLE learning_patterns IS 'Successful and failed patterns learned over time, used for improving future attempts';
COMMENT ON TABLE tool_usage IS 'Tracks which tools are used, their success rates, and usage patterns';

-- Add comments on JSONB fields for clarity
COMMENT ON COLUMN training_attempts.metadata IS 'Additional attempt-specific data in JSON format';
COMMENT ON COLUMN learning_patterns.metadata IS 'Additional pattern-specific data in JSON format';
COMMENT ON COLUMN tool_usage.parameters IS 'Parameters passed to the tool during execution';
COMMENT ON COLUMN tool_usage.metadata IS 'Additional tool-specific data in JSON format';
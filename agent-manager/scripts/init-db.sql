-- Initialize database for agent manager

-- Create database and user (if not exists)
CREATE DATABASE agent_manager;
CREATE USER agent_manager WITH ENCRYPTED PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE agent_manager TO agent_manager;

-- Connect to the agent_manager database
\c agent_manager

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO agent_manager;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO agent_manager;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO agent_manager;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Insert default resource quotas for development
INSERT INTO resource_quotas (user_id, resource_type, quota_limit, current_usage) VALUES
('admin', 'cpu', 100.0, 0.0),
('admin', 'memory', 200.0, 0.0),
('admin', 'gpu', 10.0, 0.0),
('user1', 'cpu', 10.0, 0.0),
('user1', 'memory', 20.0, 0.0),
('user1', 'gpu', 2.0, 0.0),
('researcher', 'cpu', 50.0, 0.0),
('researcher', 'memory', 100.0, 0.0),
('researcher', 'gpu', 5.0, 0.0)
ON CONFLICT DO NOTHING;
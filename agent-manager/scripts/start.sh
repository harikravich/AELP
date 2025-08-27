#!/bin/bash
set -e

echo "🚀 Starting GAELP Agent Manager"
echo "================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if required environment variables are set
required_vars=("DB_HOST" "DB_NAME" "DB_USER" "DB_PASSWORD")
for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        echo "⚠️  Warning: $var environment variable is not set"
    fi
done

# Change to script directory
cd "$(dirname "$0")/.."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check database connection
echo "🔍 Checking database connection..."
python -c "
import os
import psycopg2
try:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'agent_manager'),
        user=os.getenv('DB_USER', 'agent_manager'),
        password=os.getenv('DB_PASSWORD', 'password')
    )
    conn.close()
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    print('Make sure PostgreSQL is running and credentials are correct')
    exit(1)
"

# Check Redis connection
echo "🔍 Checking Redis connection..."
python -c "
import os
import redis
try:
    r = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', '6379')),
        db=int(os.getenv('REDIS_DB', '0'))
    )
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
    print('Make sure Redis is running')
    exit(1)
"

# Check Kubernetes access
echo "🔍 Checking Kubernetes access..."
if command -v kubectl &> /dev/null; then
    if kubectl cluster-info &> /dev/null; then
        echo "✅ Kubernetes cluster accessible"
    else
        echo "⚠️  Warning: Cannot access Kubernetes cluster"
        echo "   Make sure kubectl is configured correctly"
    fi
else
    echo "⚠️  Warning: kubectl not found"
    echo "   Install kubectl for Kubernetes functionality"
fi

# Initialize database if needed
echo "🗄️  Initializing database..."
python -c "
from core.database import create_tables
create_tables()
print('✅ Database tables initialized')
"

# Start the application
echo "🎯 Starting Agent Manager..."
echo "   API will be available at: http://localhost:${API_PORT:-8000}"
echo "   Metrics will be available at: http://localhost:${MONITORING_PROMETHEUS_PORT:-8080}/metrics"
echo "   Health check: http://localhost:${API_PORT:-8000}/health"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Run the main application
exec python main.py
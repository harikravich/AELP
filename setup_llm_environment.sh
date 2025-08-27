#!/bin/bash

# GAELP LLM Integration Environment Setup Script
# Sets up the environment for LLM-powered persona integration

set -e

echo "🚀 Setting up GAELP LLM Integration Environment"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "llm_persona_service.py" ]; then
    echo "❌ Error: llm_persona_service.py not found. Please run this script from the GAELP root directory."
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file template..."
    cat > .env << 'EOF'
# LLM API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Redis Configuration (if using external Redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# BigQuery Configuration
BIGQUERY_PROJECT=gaelp-project
BIGQUERY_DATASET=training_logs

# Cost and Rate Limiting
MAX_DAILY_COST=50.0
REQUESTS_PER_MINUTE=60
REQUESTS_PER_HOUR=1000

# Monitoring
LOG_LEVEL=INFO
ENABLE_MONITORING=true

# Safety Controls
REQUIRE_HUMAN_APPROVAL=false
MAX_VIOLATIONS_PER_DAY=5
EOF
    echo "✅ Created .env template file"
    echo "   Please edit .env and add your API keys"
else
    echo "✅ .env file already exists"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements_llm.txt" ]; then
    pip install -r requirements_llm.txt
    echo "✅ LLM dependencies installed"
else
    echo "⚠️  requirements_llm.txt not found, installing core dependencies..."
    pip install httpx redis google-cloud-monitoring anthropic openai pydantic
fi

# Start Redis if not running (Docker)
echo "🗄️  Checking Redis..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Starting Redis with Docker..."
    if command -v docker &> /dev/null; then
        docker run -d --name gaelp-redis -p 6379:6379 redis:alpine
        echo "✅ Redis started in Docker"
        sleep 2
    else
        echo "⚠️  Redis not running and Docker not available"
        echo "   Please install and start Redis manually, or install Docker"
        echo "   Redis is required for persona state management and caching"
    fi
else
    echo "✅ Redis is running"
fi

# Test Redis connection
echo "🔍 Testing Redis connection..."
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis connection successful"
else
    echo "❌ Redis connection failed"
    echo "   Please ensure Redis is running on localhost:6379"
fi

# Check for API keys
echo "🔑 Checking API key configuration..."
source .env 2>/dev/null || true

if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your_anthropic_api_key_here" ]; then
    echo "⚠️  Anthropic API key not configured"
else
    echo "✅ Anthropic API key configured"
fi

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "⚠️  OpenAI API key not configured"
else
    echo "✅ OpenAI API key configured"
fi

# Create directories for logs and data
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data/personas
mkdir -p data/campaigns
mkdir -p config
echo "✅ Directories created"

# Run quick integration test
echo "⚡ Running quick integration test..."
if command -v python3 &> /dev/null; then
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    python3 llm_integration_setup.py quick
else
    echo "⚠️  Python3 not found, skipping integration test"
fi

echo ""
echo "🎉 GAELP LLM Integration Environment Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys:"
echo "   - Get Anthropic API key: https://console.anthropic.com/"
echo "   - Get OpenAI API key: https://platform.openai.com/api-keys"
echo ""
echo "2. Run the full setup:"
echo "   python3 llm_integration_setup.py"
echo ""
echo "3. Test the integration:"
echo "   python3 run_full_demo_llm.py"
echo ""
echo "4. (Optional) Run the original demo for comparison:"
echo "   python3 run_full_demo.py"
echo ""
echo "📖 For more information, see the documentation in:"
echo "   - llm_persona_service.py (main service)"
echo "   - persona_factory.py (persona generation)"
echo "   - llm_integration_setup.py (setup and testing)"
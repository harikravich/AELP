#!/bin/bash

# GAELP MCP Connectors Setup Script
# This script helps set up the MCP connectors for advertising platforms

set -e

echo "🚀 Setting up GAELP MCP Connectors..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | sed 's/v//')
REQUIRED_VERSION="18.0.0"

if ! npx semver -r ">=$REQUIRED_VERSION" "$NODE_VERSION" &> /dev/null; then
    echo "❌ Node.js version $NODE_VERSION is too old. Please upgrade to Node.js 18+ first."
    exit 1
fi

echo "✅ Node.js version $NODE_VERSION detected"

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p config
mkdir -p logs
mkdir -p dist

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build TypeScript
echo "🔨 Building TypeScript..."
npm run build:mcp

# Copy example configuration if config file doesn't exist
if [ ! -f "config/mcp-config.json" ]; then
    echo "📝 Creating configuration template..."
    cp config/mcp-config.example.json config/mcp-config.json
    echo "⚠️  Please edit config/mcp-config.json with your API credentials"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOF
# MCP Connector Environment Variables
DEBUG=false
LOG_LEVEL=info

# Security
ENCRYPT_CREDENTIALS=true
CREDENTIAL_KEY=your-encryption-key-here

# Monitoring
HEALTH_CHECK_INTERVAL=300000
PERFORMANCE_MONITORING=true

# Development
NODE_ENV=production
EOF
    echo "⚠️  Please edit .env with your environment settings"
fi

# Create systemd service files (optional)
if command -v systemctl &> /dev/null; then
    echo "📋 Creating systemd service templates..."
    
    cat > gaelp-meta-mcp.service << EOF
[Unit]
Description=GAELP Meta Ads MCP Connector
After=network.target

[Service]
Type=simple
User=gaelp
WorkingDirectory=$(pwd)
Environment=NODE_ENV=production
ExecStart=/usr/bin/node dist/meta-ads/meta-mcp-server.js
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    cat > gaelp-google-mcp.service << EOF
[Unit]
Description=GAELP Google Ads MCP Connector
After=network.target

[Service]
Type=simple
User=gaelp
WorkingDirectory=$(pwd)
Environment=NODE_ENV=production
ExecStart=/usr/bin/node dist/google-ads/google-mcp-server.js
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    echo "💡 Systemd service files created. To install:"
    echo "   sudo cp gaelp-*-mcp.service /etc/systemd/system/"
    echo "   sudo systemctl enable gaelp-meta-mcp.service"
    echo "   sudo systemctl enable gaelp-google-mcp.service"
fi

# Create monitoring script
echo "📊 Creating monitoring script..."
cat > scripts/monitor.sh << 'EOF'
#!/bin/bash

# MCP Connectors Health Monitoring Script

check_service() {
    local service_name=$1
    local port=$2
    
    echo "Checking $service_name..."
    
    if pgrep -f "$service_name" > /dev/null; then
        echo "✅ $service_name is running"
        
        # Test MCP health endpoint if available
        if command -v curl &> /dev/null; then
            response=$(echo '{"method": "tools/call", "params": {"name": "health_check", "arguments": {}}}' | timeout 10 node dist/$service_name 2>/dev/null || echo "timeout")
            if [[ "$response" != "timeout" ]]; then
                echo "✅ $service_name health check passed"
            else
                echo "⚠️  $service_name health check failed or timed out"
            fi
        fi
    else
        echo "❌ $service_name is not running"
    fi
    echo ""
}

echo "🔍 GAELP MCP Connectors Health Check"
echo "===================================="

check_service "meta-ads/meta-mcp-server.js"
check_service "google-ads/google-mcp-server.js"

# Check log files for errors
echo "📋 Checking recent logs for errors..."
if [ -d "logs" ]; then
    find logs -name "*.log" -mtime -1 -exec grep -l "ERROR\|CRITICAL" {} \; | while read logfile; do
        echo "⚠️  Errors found in $logfile"
        tail -5 "$logfile"
        echo ""
    done
else
    echo "ℹ️  No log directory found"
fi

echo "Health check completed at $(date)"
EOF

chmod +x scripts/monitor.sh

# Create development scripts
echo "🛠️  Creating development scripts..."
cat > scripts/dev-setup.sh << 'EOF'
#!/bin/bash

# Development setup for MCP connectors

echo "🔧 Setting up development environment..."

# Set development environment
export NODE_ENV=development
export DEBUG=true

# Create development config
if [ ! -f "config/mcp-config.dev.json" ]; then
    cp config/mcp-config.example.json config/mcp-config.dev.json
    echo "📝 Created development config at config/mcp-config.dev.json"
fi

# Install dev dependencies
npm install --include=dev

# Setup git hooks if in git repo
if [ -d ".git" ]; then
    echo "🪝 Setting up git hooks..."
    cat > .git/hooks/pre-commit << 'HOOK'
#!/bin/bash
npm run type-check
HOOK
    chmod +x .git/hooks/pre-commit
fi

echo "✅ Development environment ready!"
echo "💡 Use 'npm run dev:meta' or 'npm run dev:google' to start development servers"
EOF

chmod +x scripts/dev-setup.sh

# Create testing script
cat > scripts/test-connectors.sh << 'EOF'
#!/bin/bash

# Test script for MCP connectors

echo "🧪 Testing MCP Connectors..."

test_connector() {
    local connector=$1
    local server_script="dist/$connector/$connector-mcp-server.js"
    
    echo "Testing $connector connector..."
    
    if [ ! -f "$server_script" ]; then
        echo "❌ Server script not found: $server_script"
        return 1
    fi
    
    # Test basic connection
    timeout 10 bash -c "echo '{\"method\": \"tools/list\", \"params\": {}}' | node $server_script" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ $connector connector basic test passed"
    else
        echo "❌ $connector connector basic test failed"
    fi
}

# Build first
npm run build:mcp

# Test connectors
test_connector "meta-ads"
test_connector "google-ads"

echo "🏁 Testing completed"
EOF

chmod +x scripts/test-connectors.sh

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit config/mcp-config.json with your API credentials"
echo "2. Edit .env with your environment settings"
echo "3. Test the connectors: ./scripts/test-connectors.sh"
echo "4. Start the services:"
echo "   - Meta Ads: npm run start:meta-mcp"
echo "   - Google Ads: npm run start:google-mcp"
echo ""
echo "For development:"
echo "- Run ./scripts/dev-setup.sh for development environment"
echo "- Use npm run dev:meta or npm run dev:google for development"
echo ""
echo "For monitoring:"
echo "- Run ./scripts/monitor.sh to check service health"
echo ""
echo "📚 See README.md for detailed documentation"
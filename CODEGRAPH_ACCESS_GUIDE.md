# GAELP Codegraph Access Guide

**Purpose:** Complete guide for analyzing GAELP codebase using Sourcegraph  
**Updated:** August 27, 2025  
**Use:** Reference this in future sessions for codebase analysis

---

## 🚀 QUICK START

### **Authentication Setup**
```bash
# 1. Install Sourcegraph CLI (if not installed)
curl -L https://github.com/sourcegraph/src-cli/releases/latest/download/src_linux_amd64 -o /tmp/src
chmod +x /tmp/src && sudo mv /tmp/src /usr/local/bin/src

# 2. Configure authentication
export SRC_ENDPOINT=https://gaelp.sourcegraph.app
export SRC_ACCESS_TOKEN="sgp_ws0198e95b5e347475a8fe969e67e3c881_4c7c67af55d0650dce83f7408e452317a5859150"

# 3. Test connection
src login
```

### **Basic Usage**
```bash
# Search all Python files
src search 'repo:github.com/harikravich/AELP file:\.py$ SEARCH_TERM'

# Search specific files
src search 'repo:github.com/harikravich/AELP file:gaelp_master_integration.py SEARCH_TERM'

# Find class definitions
src search 'repo:github.com/harikravich/AELP class.*Agent'

# Find method definitions
src search 'repo:github.com/harikravich/AELP def.*train'

# Find imports
src search 'repo:github.com/harikravich/AELP import.*stable_baselines'
```

---

## 📋 COMPONENT ANALYSIS COMMANDS

### **Core RL Components**
```bash
# RL Agent Implementations
src search 'repo:github.com/harikravich/AELP class.*PPO OR class.*Agent'

# Training Methods
src search 'repo:github.com/harikravich/AELP def.*train OR def.*learn OR def.*update'

# RL Imports
src search 'repo:github.com/harikravich/AELP import.*PPO OR import.*stable_baselines'
```

### **Critical Integrations**
```bash
# RecSim Integration
src search 'repo:github.com/harikravich/AELP file:recsim.*\.py import.*recsim_ng OR class.*RecSim'

# AuctionGym Integration  
src search 'repo:github.com/harikravich/AELP file:auction.*\.py class.*Auction OR def.*run_auction'

# GA4 Integration
src search 'repo:github.com/harikravich/AELP file:ga4.*\.py OR file:discovery.*\.py class.*GA4'
```

### **User Journey System**
```bash
# User Database
src search 'repo:github.com/harikravich/AELP file:user_journey.*\.py class.*Database OR class.*Journey'

# Attribution Models
src search 'repo:github.com/harikravich/AELP file:attribution.*\.py class.*Attribution'

# Conversion Tracking
src search 'repo:github.com/harikravich/AELP conversion OR purchase OR signup'
```

### **Production Systems**
```bash
# Safety Systems
src search 'repo:github.com/harikravich/AELP file:safety.*\.py class.*Safety'

# Budget Management
src search 'repo:github.com/harikravich/AELP file:budget.*\.py class.*Budget'

# Model Versioning
src search 'repo:github.com/harikravich/AELP file:model_versioning.*\.py'
```

---

## 🔍 ANALYSIS PATTERNS

### **Component Verification**
```bash
# Check if component has real implementation (not just imports)
src search 'repo:github.com/harikravich/AELP file:COMPONENT.py def.*' 

# Check for fallbacks/mocks in production
src search 'repo:github.com/harikravich/AELP file:COMPONENT.py fallback OR mock OR simplified'

# Verify imports work
src search 'repo:github.com/harikravich/AELP file:COMPONENT.py from.*import OR import.*'
```

### **Integration Testing**
```bash
# Check master integration imports
src search 'repo:github.com/harikravich/AELP file:gaelp_master_integration.py from.*import'

# Find orchestration methods
src search 'repo:github.com/harikravich/AELP file:gaelp_master_integration.py def.*run OR def.*simulate'

# Check component usage
src search 'repo:github.com/harikravich/AELP COMPONENT_NAME\.'
```

### **Quality Assessment**
```bash
# Find all fallback violations
src search 'repo:github.com/harikravich/AELP fallback OR simplified OR mock OR dummy'

# Check for hardcoded values
src search 'repo:github.com/harikravich/AELP =.*0\.[0-9] OR =.*[0-9]+\.[0-9]'

# Find TODO/FIXME items
src search 'repo:github.com/harikravich/AELP TODO OR FIXME OR HACK'
```

---

## 📊 COMPREHENSIVE ANALYSIS WORKFLOW

### **Step 1: Component Inventory**
```bash
# List all main Python files
src search 'repo:github.com/harikravich/AELP file:\.py$' | grep -oE '[a-zA-Z0-9_/-]+\.py' | sort | uniq > component_list.txt
```

### **Step 2: Architecture Mapping**
```bash
# Find all class definitions
src search 'repo:github.com/harikravich/AELP class.*' > classes.txt

# Find all main entry points
src search 'repo:github.com/harikravich/AELP if __name__' > entry_points.txt

# Map import relationships
src search 'repo:github.com/harikravich/AELP from.*import' > imports.txt
```

### **Step 3: Functionality Verification**
```bash
# Check each major component
for component in gaelp_master_integration journey_aware_rl_agent enhanced_simulator user_journey_database; do
    echo "=== $component ===" 
    src search "repo:github.com/harikravich/AELP file:${component}.py def.*"
done
```

### **Step 4: Integration Validation**
```bash
# Verify real implementations (not mocks)
src search 'repo:github.com/harikravich/AELP -file:test_ -file:demo_ -file:example_ class.*Mock'

# Check critical integrations
src search 'repo:github.com/harikravich/AELP import.*recsim_ng OR import.*stable_baselines3 OR import.*auction'
```

---

## 🎯 SPECIFIC ANALYSIS QUERIES

### **Production Readiness Check**
```bash
# Safety systems
src search 'repo:github.com/harikravich/AELP class.*Safety OR def.*validate.*bid'

# Error handling
src search 'repo:github.com/harikravich/AELP try: OR except:'

# Configuration management
src search 'repo:github.com/harikravich/AELP config OR settings OR parameters'
```

### **Business Logic Verification**
```bash
# Aura-specific logic
src search 'repo:github.com/harikravich/AELP aura OR balance OR parental'

# GA4 property integration
src search 'repo:github.com/harikravich/AELP 308028264 OR hari@aura.com'

# Conversion tracking
src search 'repo:github.com/harikravich/AELP trial OR subscription OR signup'
```

### **Performance & Scaling**
```bash
# Parallel processing
src search 'repo:github.com/harikravich/AELP parallel OR async OR concurrent'

# Optimization algorithms
src search 'repo:github.com/harikravich/AELP optimize OR algorithm OR strategy'

# Monitoring & metrics
src search 'repo:github.com/harikravich/AELP metric OR measure OR track'
```

---

## 🔧 TROUBLESHOOTING

### **Authentication Issues**
```bash
# Reset authentication
export SRC_ENDPOINT=https://gaelp.sourcegraph.app
export SRC_ACCESS_TOKEN="sgp_ws0198e95b5e347475a8fe969e67e3c881_4c7c67af55d0650dce83f7408e452317a5859150"
src login

# Test connection
src search 'repo:github.com/harikravich/AELP test' | head -1
```

### **Search Issues**
```bash
# If no results, check repository name
src search 'repo:github.com/harikravich/AELP .*' | head -1

# Use broader search if specific fails
src search 'repo:github.com/harikravich/AELP TERM' 
```

### **Rate Limiting**
```bash
# Add delays between searches if rate limited
sleep 1 && src search 'query'
```

---

## 💡 ANALYSIS BEST PRACTICES

### **Comprehensive Assessment**
1. **Start broad**: `src search 'repo:github.com/harikravich/AELP class.*'`
2. **Focus specific**: `src search 'repo:github.com/harikravich/AELP file:component.py def.*'`
3. **Verify integrations**: Check imports and method calls
4. **Test functionality**: Look for actual implementations vs mocks

### **Component Deep Dive**
1. **Find the file**: `src search 'repo:github.com/harikravich/AELP file:component.py'`
2. **Check classes**: `src search 'repo:github.com/harikravich/AELP file:component.py class.*'`
3. **Check methods**: `src search 'repo:github.com/harikravich/AELP file:component.py def.*'`
4. **Check usage**: `src search 'repo:github.com/harikravich/AELP ComponentName\.'`

### **Quality Verification**
1. **No fallbacks**: `src search 'repo:github.com/harikravich/AELP fallback OR simplified'`
2. **Real integrations**: `src search 'repo:github.com/harikravich/AELP import.*real_library'`
3. **Production ready**: `src search 'repo:github.com/harikravich/AELP class.*Safety OR def.*validate'`

---

## 📚 REFERENCE

### **Key Files to Always Check**
- `gaelp_master_integration.py` - Main orchestrator
- `journey_aware_rl_agent.py` - Core RL implementation  
- `enhanced_simulator.py` - Training environment
- `user_journey_database.py` - User persistence
- `discovery_engine.py` - Real data integration

### **Critical Search Patterns**
- `class.*Agent` - Find RL agents
- `def.*train` - Find training methods
- `import.*` - Find dependencies
- `fallback OR mock` - Find quality issues
- `property.*308028264` - Find Aura GA4 integration

This guide ensures consistent and thorough codebase analysis in future sessions.
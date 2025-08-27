# GA4 Data Calibration Plan for GAELP

## Current Status
- ✅ Service account created: `ga4-mcp-server@centering-line-469716-r7.iam.gserviceaccount.com`
- ✅ GA4 MCP server configured and connected
- ❌ Awaiting GA4 property access (needs Jason to add service account)
- Property ID: 308028264

## What We'll Extract for CALIBRATION (Not Training)

### 1. Conversion Rate Calibration
```python
calibration_params = {
    'google_search': {'cvr': actual_from_ga4},  # Replace synthetic 5%
    'facebook': {'cvr': actual_from_ga4},        # Replace synthetic 3%
    'tiktok': {'cvr': actual_from_ga4},          # Replace synthetic 2%
    'direct': {'cvr': actual_from_ga4},          # Replace synthetic 8%
}
```

### 2. User Journey Patterns
- **Touchpoint sequences**: Real paths users take before converting
- **Attribution windows**: Actual time between first touch and conversion
- **Drop-off rates**: Where users abandon in the funnel
- **Channel interactions**: How channels work together

### 3. Behavioral Patterns
- **Time of day**: When parents search for parental controls
- **Crisis signals**: Keywords/events indicating urgent need
- **Research patterns**: How long users research before buying
- **iOS vs Android**: Conversion rate differences (Balance is iOS only)

### 4. Campaign Performance
- **Actual CAC by channel**: Real customer acquisition costs
- **LTV by segment**: Lifetime value of different user types
- **Creative performance**: Which messages actually convert
- **Landing page effectiveness**: Which pages drive trials

## How We'll Use This Data

### CALIBRATION ONLY (Not Training)
```python
# Bad approach (overfitting):
rl_agent.train(real_ga4_data)  # NO! Would overfit to historical

# Good approach (calibration):
simulator.calibrate_parameters(real_ga4_data)  # YES! Makes simulation realistic
rl_agent.train(calibrated_simulation)  # Train on diverse simulated scenarios
```

### Specific Calibrations

1. **Conversion Rates**
   - Replace hardcoded 5% with actual channel CVRs
   - Adjust for iOS vs Android differences
   - Include seasonal variations

2. **User Behavior**
   - Update journey length distributions
   - Calibrate touchpoint probabilities
   - Fix attribution window timing

3. **Competition**
   - Infer competitor bid levels from impression share
   - Calibrate win rate curves (fix 100% bug)
   - Adjust quality score impacts

4. **Budget Pacing**
   - Use actual spend patterns through the day
   - Calibrate dayparting multipliers
   - Adjust for platform-specific pacing

## Next Steps Once Access Granted

1. **Run Initial Exploration**
   ```bash
   python3 explore_ga4_data.py
   ```

2. **Extract Calibration Data**
   - Pull 90 days of conversion data
   - Map user journey sequences
   - Calculate real CAC/LTV

3. **Update Simulator Parameters**
   - Replace synthetic values in `enhanced_simulator.py`
   - Update journey distributions in `user_journey_database.py`
   - Calibrate auction mechanics in `auction_gym_integration.py`

4. **Validate Calibration**
   - Compare simulated metrics to actuals
   - Ensure realistic behavior
   - Test edge cases

## UTM Strategy for Personal Testing

When we run personal ads, use these UTMs:
```
utm_source=gaelp_personal
utm_medium=cpc
utm_campaign=behavioral_health_test_1
utm_content=social_scanner_v1
utm_term={keyword}
```

This will let us track our test performance separately in GA4.

## Security Notes
- Service account has read-only access
- Credentials stored securely in `~/.config/gaelp/`
- Added to .gitignore to prevent commits
- File permissions set to 600 (owner only)

## NO FALLBACKS
- Will NOT use mock data
- Will NOT simplify if access fails
- Will wait for proper access
- Will use REAL data only
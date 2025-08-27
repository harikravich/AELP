# How to Create Your Own OAuth Client for GA4

## The Problem
- Google's test OAuth client (764086051850...) is completely blocked
- We need YOUR OWN OAuth client that can handle OKTA SSO properly

## Steps to Create Your OAuth Client

### 1. Go to Google Cloud Console
- Navigate to: https://console.cloud.google.com/
- Make sure you're logged in with hari@aura.com

### 2. Select or Create Project
- Use project: `centering-line-469716-r7` (if you have access)
- OR create your own project

### 3. Enable Google Analytics Data API
- Go to "APIs & Services" → "Library"
- Search for "Google Analytics Data API"
- Click on it and press "ENABLE"

### 4. Create OAuth 2.0 Credentials
- Go to "APIs & Services" → "Credentials"
- Click "+ CREATE CREDENTIALS" → "OAuth client ID"

### 5. Configure OAuth Consent Screen (if needed)
- Choose "Internal" (if available) or "External"
- App name: "GAELP GA4 Integration"
- User support email: hari@aura.com
- Developer contact: hari@aura.com
- Add scopes: `https://www.googleapis.com/auth/analytics.readonly`

### 6. Create the OAuth Client
- Application type: **Web application** (NOT Desktop)
- Name: "GAELP OAuth Client"
- Authorized redirect URIs - Add these:
  - `http://localhost:8080`
  - `http://localhost:8080/`
  - `http://localhost:8888`
  - `http://localhost:8888/`
  - `http://localhost:9090`
  - `http://localhost:9090/`

### 7. Download Credentials
- After creating, click the download icon
- Save as: `oauth_client_secret.json`
- Upload to server: `/home/hariravichandran/.config/gaelp/oauth_client_secret.json`

## Why This Works
- YOUR OAuth client can handle OKTA SSO redirects
- It's authorized for your organization
- Google won't block your own client
- OKTA SSO will work properly with hari@aura.com

## After Creating the Client
Run this updated script that uses YOUR OAuth client:
```bash
python3 ga4_oauth_with_your_client.py
```

The flow will be:
1. OAuth starts with YOUR client
2. Redirects to Google login
3. Google sees hari@aura.com → redirects to OKTA
4. You login with OKTA
5. OKTA sends you back to Google
6. Google sends you back to localhost with the code
7. Success!

## Alternative: Just Use Service Account
If creating OAuth client is too complex, the service account is simpler:
- Email: ga4-mcp-server@centering-line-469716-r7.iam.gserviceaccount.com
- Just need Jason to add it to GA4 with Viewer access
- No OAuth, no OKTA, no browser needed
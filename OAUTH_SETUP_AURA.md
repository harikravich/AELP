# OAuth Setup for Aura Organization

## You're in the right place!
You have access to `aura.com` organization (ID: 731063653202)

## Next Steps:

### 1. Find or Create a Project
Look for existing projects under aura.com:
- Click on "All" to see all projects
- Look for something like:
  - `aura-analytics`
  - `aura-marketing`
  - `aura-data`
  - Or any project with GA4/Analytics access

**OR Create New Project:**
- Click "NEW PROJECT"
- Project name: `gaelp-ga4-integration`
- Organization: `aura.com`
- Click CREATE

### 2. Once in a Project:

#### Enable the API:
1. Go to "APIs & Services" → "Library"
2. Search for: `Google Analytics Data API`
3. Click on it and press "ENABLE"

#### Create OAuth Consent Screen:
1. Go to "APIs & Services" → "OAuth consent screen"
2. Choose "Internal" (since you're in aura.com org)
3. Fill in:
   - App name: `GAELP GA4 Integration`
   - User support email: `hari@aura.com`
   - Developer contact: `hari@aura.com`
4. Add scopes:
   - Click "ADD OR REMOVE SCOPES"
   - Search for: `analytics.readonly`
   - Check: `https://www.googleapis.com/auth/analytics.readonly`
5. Save and Continue

#### Create OAuth Client:
1. Go to "APIs & Services" → "Credentials"
2. Click "+ CREATE CREDENTIALS" → "OAuth client ID"
3. Application type: **Web application**
4. Name: `GAELP OAuth Client`
5. Authorized redirect URIs - Add ALL of these:
   ```
   http://localhost:8080
   http://localhost:8080/
   http://localhost:8888
   http://localhost:8888/
   http://localhost:9090
   http://localhost:9090/
   http://127.0.0.1:8080
   http://127.0.0.1:8080/
   http://127.0.0.1:8888
   http://127.0.0.1:8888/
   ```
6. Click CREATE

#### Download the Credentials:
1. After creating, you'll see your client in the list
2. Click the download icon (⬇️) on the right
3. Save the JSON file

### 3. Upload to Server:
Upload the downloaded JSON to:
```
/home/hariravichandran/.config/gaelp/oauth_client_secret.json
```

### 4. Run the OAuth Script:
```bash
python3 ga4_oauth_with_your_client.py
```

## Why This Will Work:
- ✅ You're using Aura's organization (not Google's test client)
- ✅ Internal app means no review needed
- ✅ OKTA SSO will work properly with hari@aura.com
- ✅ You'll have proper GA4 access through your Aura account

## Important Notes:
- Since you're in aura.com org, choose "Internal" for consent screen
- This means only aura.com users can use it (perfect for you)
- No Google review process needed for internal apps
- OKTA SSO will work seamlessly
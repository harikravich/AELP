#!/bin/bash

# Script to get Facebook long-lived access token
echo "Facebook Ads API Token Generator"
echo "================================"
echo ""
echo "Step 1: Get your short-lived token from:"
echo "https://developers.facebook.com/tools/explorer/"
echo ""
read -p "Enter your App ID: " APP_ID
read -p "Enter your App Secret: " APP_SECRET
read -p "Enter your short-lived access token: " SHORT_TOKEN

echo ""
echo "Exchanging for long-lived token..."

# Exchange for long-lived token
RESPONSE=$(curl -s -X GET "https://graph.facebook.com/v18.0/oauth/access_token?grant_type=fb_exchange_token&client_id=${APP_ID}&client_secret=${APP_SECRET}&fb_exchange_token=${SHORT_TOKEN}")

echo ""
echo "Response:"
echo $RESPONSE | jq '.'

LONG_TOKEN=$(echo $RESPONSE | jq -r '.access_token')

if [ "$LONG_TOKEN" != "null" ] && [ -n "$LONG_TOKEN" ]; then
    echo ""
    echo "Success! Your long-lived token is:"
    echo "$LONG_TOKEN"
    echo ""
    echo "This token will last ~60 days"

    # Optionally save to .env
    read -p "Do you want to save this to your .env file? (y/n): " SAVE_ENV
    if [ "$SAVE_ENV" = "y" ]; then
        sed -i "s/export META_ACCESS_TOKEN=.*/export META_ACCESS_TOKEN=${LONG_TOKEN}/" /home/hariravichandran/AELP/.env
        sed -i "s/export META_APP_ID=.*/export META_APP_ID=${APP_ID}/" /home/hariravichandran/AELP/.env
        sed -i "s/export META_APP_SECRET=.*/export META_APP_SECRET=${APP_SECRET}/" /home/hariravichandran/AELP/.env
        echo "Updated .env file!"
    fi
else
    echo "Error getting long-lived token. Please check your credentials."
fi
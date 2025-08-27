"""
Aura Webhook Endpoint for Cross-Account Attribution

This Flask application serves as the webhook endpoint that receives conversion
events from Aura's system and attributes them back to original personal ad campaigns.

CRITICAL: This is the final piece that completes the attribution chain.
Must be deployed and accessible by Aura's webhook system.
"""

from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import os
import json
import hmac
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import sqlite3
import requests
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our cross-account attributor
from cross_account_attributor import ServerSideTracker, CrossAccountDashboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=[
    'https://aura.com',
    'https://www.aura.com',
    'https://admin.aura.com'
])

# Initialize tracker
tracker = ServerSideTracker(
    domain="teen-wellness-monitor.com",
    gtm_container_id="GTM-GAELP001"
)
dashboard = CrossAccountDashboard(tracker)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

@dataclass
class WebhookEvent:
    """Webhook event from Aura"""
    event_type: str
    user_id: str
    gaelp_uid: Optional[str]
    transaction_id: Optional[str]
    value: float
    currency: str
    timestamp: datetime
    raw_payload: Dict[str, Any]


class AuraWebhookProcessor:
    """Process webhook events from Aura"""
    
    def __init__(self, tracker: ServerSideTracker):
        self.tracker = tracker
        self.webhook_secret = os.environ.get('AURA_WEBHOOK_SECRET', 'default-secret-change-in-production')
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature from Aura"""
        
        if not signature:
            logger.warning("No webhook signature provided")
            return False
        
        # Extract signature from header (usually in format "sha256=<signature>")
        if signature.startswith('sha256='):
            provided_signature = signature[7:]
        else:
            provided_signature = signature
        
        # Calculate expected signature
        expected_signature = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Secure comparison
        return hmac.compare_digest(expected_signature, provided_signature)
    
    def parse_webhook_event(self, payload: Dict[str, Any]) -> Optional[WebhookEvent]:
        """Parse webhook payload into WebhookEvent"""
        
        try:
            # Extract event type
            event_type = payload.get('event', payload.get('type', 'unknown'))
            
            # Extract user information
            user_id = payload.get('user_id', payload.get('userId'))
            
            # Extract GAELP UID from multiple possible locations
            gaelp_uid = self._extract_gaelp_uid(payload)
            
            # Extract transaction details
            transaction_id = payload.get('transaction_id', payload.get('transactionId'))
            
            # Extract value and currency
            value = float(payload.get('value', payload.get('revenue', 0)))
            currency = payload.get('currency', 'USD')
            
            # Extract timestamp
            timestamp_str = payload.get('timestamp', payload.get('event_time'))
            if timestamp_str:
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.fromtimestamp(int(timestamp_str))
            else:
                timestamp = datetime.now()
            
            return WebhookEvent(
                event_type=event_type,
                user_id=user_id,
                gaelp_uid=gaelp_uid,
                transaction_id=transaction_id,
                value=value,
                currency=currency,
                timestamp=timestamp,
                raw_payload=payload
            )
            
        except Exception as e:
            logger.error(f"Failed to parse webhook event: {e}")
            return None
    
    def _extract_gaelp_uid(self, payload: Dict[str, Any]) -> Optional[str]:
        """Extract GAELP UID from various locations in the payload"""
        
        # Check direct properties
        gaelp_uid = payload.get('gaelp_uid')
        if gaelp_uid:
            return gaelp_uid
        
        # Check user properties
        user_properties = payload.get('user_properties', {})
        if isinstance(user_properties, dict):
            gaelp_uid = user_properties.get('gaelp_uid')
            if gaelp_uid:
                return gaelp_uid
        
        # Check custom dimensions
        custom_dimensions = payload.get('custom_dimensions', {})
        if isinstance(custom_dimensions, dict):
            for key, value in custom_dimensions.items():
                if 'gaelp_uid' in key.lower():
                    return value
        
        # Check event parameters
        event_params = payload.get('event_parameters', payload.get('eventParameters', {}))
        if isinstance(event_params, dict):
            gaelp_uid = event_params.get('gaelp_uid')
            if gaelp_uid:
                return gaelp_uid
        
        # Check nested user data
        user_data = payload.get('user_data', payload.get('userData', {}))
        if isinstance(user_data, dict):
            gaelp_uid = user_data.get('gaelp_uid')
            if gaelp_uid:
                return gaelp_uid
        
        # Check URL parameters if page_location is provided
        page_location = payload.get('page_location', payload.get('pageLocation'))
        if page_location and isinstance(page_location, str):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(page_location)
            query_params = parse_qs(parsed.query)
            if 'gaelp_uid' in query_params:
                return query_params['gaelp_uid'][0]
        
        return None
    
    def process_conversion_event(self, event: WebhookEvent) -> bool:
        """Process conversion event and update attribution"""
        
        if not event.gaelp_uid:
            logger.warning(f"No GAELP UID found for conversion event: {event.transaction_id}")
            return False
        
        try:
            # Convert to format expected by tracker
            webhook_payload = {
                'user_id': event.user_id,
                'transaction_id': event.transaction_id,
                'value': event.value,
                'currency': event.currency,
                'user_properties': {
                    'gaelp_uid': event.gaelp_uid
                },
                'event_timestamp': int(event.timestamp.timestamp()),
                'item_category': 'balance_subscription'  # Default for Aura
            }
            
            # Process through our tracker
            success = self.tracker.track_aura_conversion(webhook_payload)
            
            if success:
                logger.info(f"Successfully processed conversion: {event.transaction_id}, "
                           f"Value: ${event.value}, GAELP UID: {event.gaelp_uid}")
                
                # Send async notifications
                executor.submit(self._send_success_notifications, event)
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to process conversion event: {e}")
            return False
    
    def _send_success_notifications(self, event: WebhookEvent):
        """Send notifications about successful attribution (async)"""
        
        try:
            # Send to monitoring system
            self._send_to_monitoring({
                'event': 'attribution_success',
                'gaelp_uid': event.gaelp_uid,
                'transaction_id': event.transaction_id,
                'value': event.value,
                'timestamp': event.timestamp.isoformat()
            })
            
            # Send to analytics dashboard
            self._update_realtime_dashboard(event)
            
        except Exception as e:
            logger.error(f"Failed to send success notifications: {e}")
    
    def _send_to_monitoring(self, data: Dict[str, Any]):
        """Send event to monitoring system"""
        
        monitoring_url = os.environ.get('MONITORING_WEBHOOK_URL')
        if monitoring_url:
            try:
                requests.post(monitoring_url, json=data, timeout=5)
            except Exception as e:
                logger.error(f"Failed to send to monitoring: {e}")
    
    def _update_realtime_dashboard(self, event: WebhookEvent):
        """Update real-time dashboard with conversion"""
        
        # This could update a real-time dashboard or send to a message queue
        # For now, we'll just log it
        logger.info(f"Real-time dashboard update: Conversion ${event.value} attributed to {event.gaelp_uid}")


# Initialize processor
webhook_processor = AuraWebhookProcessor(tracker)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'aura-webhook-endpoint',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/webhook/aura/conversions', methods=['POST'])
def aura_conversion_webhook():
    """Main webhook endpoint for Aura conversions"""
    
    try:
        # Get raw payload for signature verification
        raw_payload = request.get_data()
        
        # Get signature from header
        signature = request.headers.get('X-Webhook-Signature') or request.headers.get('X-Hub-Signature-256')
        
        # Verify signature (in production)
        if os.environ.get('VERIFY_WEBHOOK_SIGNATURES', 'false').lower() == 'true':
            if not webhook_processor.verify_webhook_signature(raw_payload, signature):
                logger.warning("Invalid webhook signature")
                abort(401)
        
        # Parse JSON payload
        try:
            payload = request.get_json(force=True)
        except Exception as e:
            logger.error(f"Failed to parse JSON payload: {e}")
            abort(400)
        
        # Parse webhook event
        event = webhook_processor.parse_webhook_event(payload)
        if not event:
            logger.error("Failed to parse webhook event")
            abort(400)
        
        # Process conversion event
        success = webhook_processor.process_conversion_event(event)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Conversion processed successfully',
                'gaelp_uid': event.gaelp_uid,
                'transaction_id': event.transaction_id
            }), 200
        else:
            return jsonify({
                'status': 'error', 
                'message': 'Failed to process conversion'
            }), 500
            
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/webhook/test', methods=['POST'])
def test_webhook():
    """Test webhook endpoint for development"""
    
    # Generate test conversion data
    test_payload = {
        'event': 'purchase',
        'user_id': 'test_user_123',
        'transaction_id': f'test_txn_{int(datetime.now().timestamp())}',
        'value': 120.00,
        'currency': 'USD',
        'user_properties': {
            'gaelp_uid': request.json.get('gaelp_uid', 'test_gaelp_uid_123')
        },
        'timestamp': datetime.now().isoformat(),
        'item_category': 'balance_subscription'
    }
    
    # Process as regular webhook
    event = webhook_processor.parse_webhook_event(test_payload)
    if event:
        success = webhook_processor.process_conversion_event(event)
        
        return jsonify({
            'status': 'success' if success else 'error',
            'test_event': test_payload,
            'processed': success
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to create test event'
        }), 400


@app.route('/api/attribution/report', methods=['GET'])
def get_attribution_report():
    """Get attribution report"""
    
    days_back = request.args.get('days', 30, type=int)
    report = dashboard.get_attribution_report(days_back)
    
    return jsonify(report)


@app.route('/api/attribution/realtime', methods=['GET'])
def get_realtime_stats():
    """Get real-time attribution statistics"""
    
    stats = dashboard.get_real_time_tracking()
    return jsonify(stats)


@app.route('/api/attribution/test', methods=['POST'])
def test_full_attribution_flow():
    """Test the complete attribution flow"""
    
    try:
        # Get test parameters
        test_params = request.get_json() or {}
        
        # Step 1: Simulate ad click
        from cross_account_attributor import UserSignature, CrossAccountTrackingParams
        
        user_signature = UserSignature(
            ip_hash=hashlib.sha256("192.168.1.100".encode()).hexdigest()[:16],
            user_agent_hash=hashlib.sha256("Mozilla/5.0 Test Browser".encode()).hexdigest()[:16],
            screen_resolution="1920x1080",
            timezone="America/New_York",
            language="en-US",
            platform="Desktop"
        )
        
        tracking_params = CrossAccountTrackingParams(
            gaelp_uid="",  # Will be generated
            gaelp_source=test_params.get('source', 'test_ads'),
            gaelp_campaign=test_params.get('campaign', 'test_campaign_2024'),
            gaelp_creative=test_params.get('creative', 'test_creative_001'),
            gaelp_timestamp=int(datetime.now().timestamp()),
            gclid="test_gclid_123"
        )
        
        # Step 2: Track ad click
        landing_url = tracker.track_ad_click(
            tracking_params=tracking_params,
            user_signature=user_signature,
            landing_domain="teen-wellness-monitor.com"
        )
        
        # Extract UID
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(landing_url)
        url_params = parse_qs(parsed_url.query)
        gaelp_uid = url_params['gaelp_uid'][0]
        tracking_params.gaelp_uid = gaelp_uid
        
        # Step 3: Track landing page visit
        aura_url, resolved_uid = tracker.track_landing_page_visit(
            tracking_params=tracking_params,
            user_signature=user_signature,
            landing_domain="teen-wellness-monitor.com"
        )
        
        # Step 4: Simulate conversion
        test_conversion = {
            'user_id': 'test_user_' + gaelp_uid[-8:],
            'transaction_id': 'test_txn_' + str(int(datetime.now().timestamp())),
            'value': test_params.get('value', 120.00),
            'currency': 'USD',
            'user_properties': {
                'gaelp_uid': gaelp_uid
            },
            'timestamp': datetime.now().isoformat()
        }
        
        conversion_success = tracker.track_aura_conversion(test_conversion)
        
        # Generate report
        report = dashboard.get_attribution_report(1)  # Last 1 day
        
        return jsonify({
            'status': 'success',
            'test_flow': {
                'step1_ad_click': {'landing_url': landing_url[:100] + '...'},
                'step2_landing_visit': {'aura_url': aura_url[:100] + '...'},
                'step3_conversion': {'success': conversion_success},
                'gaelp_uid': gaelp_uid
            },
            'attribution_report': report
        })
        
    except Exception as e:
        logger.error(f"Test flow error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/tracking/events', methods=['POST'])
def track_custom_event():
    """Generic endpoint for tracking custom events"""
    
    try:
        event_data = request.get_json()
        
        # Extract required fields
        gaelp_uid = event_data.get('gaelp_uid')
        event_type = event_data.get('event_type', 'custom')
        
        if not gaelp_uid:
            return jsonify({'error': 'gaelp_uid is required'}), 400
        
        # Store event in database
        with sqlite3.connect(tracker.db_path) as conn:
            conn.execute("""
                INSERT INTO tracking_events 
                (gaelp_uid, event_type, timestamp, domain, parameters)
                VALUES (?, ?, ?, ?, ?)
            """, (
                gaelp_uid,
                event_type,
                int(datetime.now().timestamp()),
                event_data.get('domain', request.headers.get('Origin', 'unknown')),
                json.dumps(event_data)
            ))
        
        return jsonify({'status': 'success', 'event_type': event_type})
        
    except Exception as e:
        logger.error(f"Custom event tracking error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Development server
    print("🚀 Starting Aura Webhook Endpoint for Cross-Account Attribution")
    print("=" * 60)
    print(f"Health check: http://localhost:5000/health")
    print(f"Webhook endpoint: http://localhost:5000/webhook/aura/conversions")
    print(f"Test endpoint: http://localhost:5000/webhook/test")
    print(f"Attribution report: http://localhost:5000/api/attribution/report")
    print("=" * 60)
    
    # Run development server
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    )
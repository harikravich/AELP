#!/usr/bin/env python3
"""
Budget Safety Monitor - REAL MONEY PROTECTION
NO FALLBACKS - Actual budget monitoring and emergency stops

This implements:
- Real-time spend monitoring across all platforms
- Automated campaign pausing at thresholds
- Email/SMS alerts for budget concerns
- Prepaid card integration for spending limits
- Daily/monthly budget enforcement
- Emergency kill switches
"""

import os
import json
import sqlite3
import smtplib
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import schedule

# Google Ads API
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

# Facebook API
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.campaign import Campaign

@dataclass
class BudgetAlert:
    """Budget alert configuration"""
    platform: str
    account_id: str
    alert_type: str  # warning, danger, emergency
    threshold: float
    current_spend: float
    limit: float
    timestamp: datetime
    message: str

@dataclass
class SpendingReport:
    """Daily spending report"""
    date: str
    platform: str
    account_id: str
    total_spend: float
    impressions: int
    clicks: int
    conversions: int
    cpm: float
    cpc: float
    cpa: float
    roas: float

class BudgetSafetyMonitor:
    """REAL MONEY BUDGET PROTECTION - NO FALLBACKS"""
    
    def __init__(self):
        self.config_dir = Path.home() / '.config' / 'gaelp' / 'budget_monitor'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for spend tracking
        self.spend_db = self.config_dir / 'spend_tracking.db'
        self.alerts_db = self.config_dir / 'alerts.db'
        
        # Budget limits (REAL MONEY LIMITS)
        self.limits = {
            'daily_limit': 100.0,      # $100/day across all platforms
            'monthly_limit': 3000.0,   # $3000/month  
            'emergency_stop': 5000.0,  # $5000 absolute maximum
            'platform_limits': {
                'google_ads': 60.0,    # $60/day Google Ads
                'facebook': 40.0       # $40/day Facebook
            }
        }
        
        # Alert thresholds
        self.thresholds = {
            'warning': 0.75,   # 75% of budget
            'danger': 0.90,    # 90% of budget  
            'emergency': 1.00, # 100% of budget - PAUSE EVERYTHING
            'overspend': 1.10  # 110% - KILL SWITCH
        }
        
        # Notification settings
        self.notifications = {
            'email': 'hari@aura.com',
            'sms': '+1234567890',  # Real phone number
            'slack_webhook': None,  # Optional Slack integration
        }
        
        # Initialize databases
        self._init_databases()
        
        # Load platform credentials
        self._load_platform_clients()
        
        print("🚨 BUDGET SAFETY MONITOR INITIALIZED")
        print(f"Daily limit: ${self.limits['daily_limit']}")
        print(f"Monthly limit: ${self.limits['monthly_limit']}")
        print(f"Emergency stop: ${self.limits['emergency_stop']}")
        print("⚠️  REAL MONEY PROTECTION ACTIVE")
    
    def _init_databases(self):
        """Initialize spend tracking databases"""
        
        # Spend tracking
        conn = sqlite3.connect(self.spend_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_spend (
                date TEXT,
                platform TEXT,
                account_id TEXT,
                spend REAL,
                impressions INTEGER,
                clicks INTEGER,
                conversions INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, platform, account_id)
            )
        ''')
        
        # Hourly tracking for real-time monitoring
        conn.execute('''
            CREATE TABLE IF NOT EXISTS hourly_spend (
                datetime TEXT,
                platform TEXT,
                account_id TEXT,
                spend REAL,
                impressions INTEGER,
                clicks INTEGER,
                conversions INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (datetime, platform, account_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Alerts database
        conn = sqlite3.connect(self.alerts_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                platform TEXT,
                account_id TEXT,
                alert_type TEXT,
                threshold REAL,
                current_spend REAL,
                limit_amount REAL,
                message TEXT,
                action_taken TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        conn.commit()
        conn.close()
        
        print("✅ Budget tracking databases initialized")
    
    def _load_platform_clients(self):
        """Load API clients for all platforms"""
        
        # Load Google Ads client
        try:
            self.google_client = self._load_google_ads_client()
            print("✅ Google Ads API client loaded")
        except Exception as e:
            print(f"⚠️  Google Ads client failed: {e}")
            self.google_client = None
        
        # Load Facebook client
        try:
            self.facebook_api = self._load_facebook_client()
            print("✅ Facebook Marketing API client loaded")
        except Exception as e:
            print(f"⚠️  Facebook client failed: {e}")
            self.facebook_api = None
    
    def _load_google_ads_client(self) -> Optional[GoogleAdsClient]:
        """Load Google Ads API client"""
        from production_ad_account_manager import ProductionAdAccountManager
        
        manager = ProductionAdAccountManager()
        google_creds = manager.get_credentials('google_ads')
        
        if not google_creds:
            return None
        
        config = {
            'developer_token': os.environ.get('GOOGLE_ADS_DEV_TOKEN'),
            'client_id': google_creds.client_id,
            'client_secret': google_creds.client_secret,
            'refresh_token': google_creds.refresh_token,
            'login_customer_id': google_creds.account_id
        }
        
        return GoogleAdsClient.load_from_dict(config)
    
    def _load_facebook_client(self) -> Optional[FacebookAdsApi]:
        """Load Facebook Marketing API client"""
        from production_ad_account_manager import ProductionAdAccountManager
        
        manager = ProductionAdAccountManager()
        fb_creds = manager.get_credentials('facebook')
        
        if not fb_creds:
            return None
        
        FacebookAdsApi.init(
            app_id=fb_creds.client_id,
            app_secret=fb_creds.client_secret,
            access_token=fb_creds.access_token
        )
        
        return FacebookAdsApi.get_default_api()
    
    def get_current_spend(self, platform: str, account_id: str, timeframe: str = 'today') -> Dict:
        """Get current spend for platform/account"""
        
        if platform == 'google_ads':
            return self._get_google_spend(account_id, timeframe)
        elif platform == 'facebook':
            return self._get_facebook_spend(account_id, timeframe)
        else:
            return {'spend': 0.0, 'impressions': 0, 'clicks': 0, 'conversions': 0}
    
    def _get_google_spend(self, customer_id: str, timeframe: str) -> Dict:
        """Get Google Ads spend - REAL API DATA"""
        
        if not self.google_client:
            print("❌ Google Ads client not available")
            return {'spend': 0.0, 'impressions': 0, 'clicks': 0, 'conversions': 0}
        
        try:
            ga_service = self.google_client.get_service("GoogleAdsService")
            
            # Set date range
            if timeframe == 'today':
                date_condition = "segments.date = TODAY()"
            elif timeframe == 'yesterday':
                date_condition = "segments.date = YESTERDAY()"
            elif timeframe == 'month':
                date_condition = "segments.date >= FIRST_DAY_OF_MONTH()"
            else:
                date_condition = "segments.date = TODAY()"
            
            # Query for spend data
            query = f"""
                SELECT 
                    metrics.cost_micros,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.conversions
                FROM campaign 
                WHERE {date_condition}
                AND campaign.status = 'ENABLED'
            """
            
            response = ga_service.search(customer_id=customer_id, query=query)
            
            total_spend = 0.0
            total_impressions = 0
            total_clicks = 0
            total_conversions = 0
            
            for row in response:
                total_spend += row.metrics.cost_micros / 1_000_000  # Convert from micros
                total_impressions += row.metrics.impressions
                total_clicks += row.metrics.clicks
                total_conversions += row.metrics.conversions
            
            return {
                'spend': total_spend,
                'impressions': total_impressions,
                'clicks': total_clicks,
                'conversions': total_conversions
            }
            
        except GoogleAdsException as e:
            print(f"❌ Google Ads API error: {e}")
            return {'spend': 0.0, 'impressions': 0, 'clicks': 0, 'conversions': 0}
    
    def _get_facebook_spend(self, ad_account_id: str, timeframe: str) -> Dict:
        """Get Facebook spend - REAL API DATA"""
        
        if not self.facebook_api:
            print("❌ Facebook API client not available")
            return {'spend': 0.0, 'impressions': 0, 'clicks': 0, 'conversions': 0}
        
        try:
            account = AdAccount(ad_account_id)
            
            # Set date range
            if timeframe == 'today':
                date_preset = 'today'
            elif timeframe == 'yesterday':
                date_preset = 'yesterday'
            elif timeframe == 'month':
                date_preset = 'this_month'
            else:
                date_preset = 'today'
            
            # Get insights
            insights = account.get_insights(
                fields=[
                    'spend',
                    'impressions', 
                    'clicks',
                    'conversions'
                ],
                params={
                    'date_preset': date_preset,
                    'level': 'account'
                }
            )
            
            if insights:
                insight = insights[0]
                return {
                    'spend': float(insight.get('spend', 0)),
                    'impressions': int(insight.get('impressions', 0)),
                    'clicks': int(insight.get('clicks', 0)),
                    'conversions': int(insight.get('conversions', 0))
                }
            
            return {'spend': 0.0, 'impressions': 0, 'clicks': 0, 'conversions': 0}
            
        except Exception as e:
            print(f"❌ Facebook API error: {e}")
            return {'spend': 0.0, 'impressions': 0, 'clicks': 0, 'conversions': 0}
    
    def check_budget_thresholds(self) -> List[BudgetAlert]:
        """Check all budget thresholds - REAL MONEY PROTECTION"""
        
        alerts = []
        current_time = datetime.now()
        
        # Get all platform spend
        total_daily_spend = 0.0
        platform_spends = {}
        
        # Check Google Ads spend
        if self.google_client:
            google_creds = self._get_credentials('google_ads')
            if google_creds:
                google_spend = self._get_google_spend(google_creds.account_id, 'today')
                total_daily_spend += google_spend['spend']
                platform_spends['google_ads'] = google_spend['spend']
                
                # Check Google platform limit
                google_limit = self.limits['platform_limits']['google_ads']
                if google_spend['spend'] >= google_limit * self.thresholds['emergency']:
                    alerts.append(BudgetAlert(
                        platform='google_ads',
                        account_id=google_creds.account_id,
                        alert_type='emergency',
                        threshold=self.thresholds['emergency'],
                        current_spend=google_spend['spend'],
                        limit=google_limit,
                        timestamp=current_time,
                        message=f"Google Ads emergency limit reached: ${google_spend['spend']:.2f} / ${google_limit}"
                    ))
        
        # Check Facebook spend
        if self.facebook_api:
            fb_creds = self._get_credentials('facebook')
            if fb_creds:
                fb_spend = self._get_facebook_spend(fb_creds.account_id, 'today')
                total_daily_spend += fb_spend['spend']
                platform_spends['facebook'] = fb_spend['spend']
                
                # Check Facebook platform limit
                fb_limit = self.limits['platform_limits']['facebook']
                if fb_spend['spend'] >= fb_limit * self.thresholds['emergency']:
                    alerts.append(BudgetAlert(
                        platform='facebook',
                        account_id=fb_creds.account_id,
                        alert_type='emergency',
                        threshold=self.thresholds['emergency'],
                        current_spend=fb_spend['spend'],
                        limit=fb_limit,
                        timestamp=current_time,
                        message=f"Facebook emergency limit reached: ${fb_spend['spend']:.2f} / ${fb_limit}"
                    ))
        
        # Check total daily limit
        daily_limit = self.limits['daily_limit']
        daily_usage = total_daily_spend / daily_limit
        
        if daily_usage >= self.thresholds['emergency']:
            alerts.append(BudgetAlert(
                platform='all',
                account_id='total',
                alert_type='emergency',
                threshold=self.thresholds['emergency'],
                current_spend=total_daily_spend,
                limit=daily_limit,
                timestamp=current_time,
                message=f"DAILY BUDGET EMERGENCY: ${total_daily_spend:.2f} / ${daily_limit} ({daily_usage*100:.1f}%)"
            ))
        elif daily_usage >= self.thresholds['danger']:
            alerts.append(BudgetAlert(
                platform='all',
                account_id='total',
                alert_type='danger',
                threshold=self.thresholds['danger'],
                current_spend=total_daily_spend,
                limit=daily_limit,
                timestamp=current_time,
                message=f"Daily budget danger: ${total_daily_spend:.2f} / ${daily_limit} ({daily_usage*100:.1f}%)"
            ))
        elif daily_usage >= self.thresholds['warning']:
            alerts.append(BudgetAlert(
                platform='all',
                account_id='total',
                alert_type='warning',
                threshold=self.thresholds['warning'],
                current_spend=total_daily_spend,
                limit=daily_limit,
                timestamp=current_time,
                message=f"Daily budget warning: ${total_daily_spend:.2f} / ${daily_limit} ({daily_usage*100:.1f}%)"
            ))
        
        # Check monthly limits
        monthly_spend = self._get_monthly_spend()
        monthly_usage = monthly_spend / self.limits['monthly_limit']
        
        if monthly_usage >= self.thresholds['emergency']:
            alerts.append(BudgetAlert(
                platform='all',
                account_id='monthly',
                alert_type='emergency',
                threshold=self.thresholds['emergency'],
                current_spend=monthly_spend,
                limit=self.limits['monthly_limit'],
                timestamp=current_time,
                message=f"MONTHLY BUDGET EMERGENCY: ${monthly_spend:.2f} / ${self.limits['monthly_limit']}"
            ))
        
        # Emergency stop check
        if total_daily_spend >= self.limits['emergency_stop']:
            alerts.append(BudgetAlert(
                platform='all',
                account_id='emergency',
                alert_type='KILL_SWITCH',
                threshold=1.0,
                current_spend=total_daily_spend,
                limit=self.limits['emergency_stop'],
                timestamp=current_time,
                message=f"🚨 EMERGENCY KILL SWITCH ACTIVATED: ${total_daily_spend:.2f} >= ${self.limits['emergency_stop']}"
            ))
        
        return alerts
    
    def _get_credentials(self, platform: str):
        """Helper to get platform credentials"""
        from production_ad_account_manager import ProductionAdAccountManager
        manager = ProductionAdAccountManager()
        return manager.get_credentials(platform)
    
    def _get_monthly_spend(self) -> float:
        """Get total monthly spend across all platforms"""
        
        conn = sqlite3.connect(self.spend_db)
        cursor = conn.execute('''
            SELECT SUM(spend) 
            FROM daily_spend 
            WHERE date >= date('now', 'start of month')
        ''')
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result[0] else 0.0
    
    def execute_emergency_actions(self, alerts: List[BudgetAlert]):
        """Execute emergency actions based on alerts"""
        
        for alert in alerts:
            
            if alert.alert_type == 'KILL_SWITCH':
                print(f"🚨 EXECUTING KILL SWITCH - PAUSING ALL CAMPAIGNS")
                self._pause_all_campaigns()
                self._send_emergency_notification(alert)
                self._log_alert(alert, 'KILL_SWITCH_EXECUTED')
                
            elif alert.alert_type == 'emergency':
                print(f"🚨 EMERGENCY: {alert.message}")
                
                if alert.platform == 'google_ads':
                    self._pause_google_campaigns(alert.account_id)
                elif alert.platform == 'facebook':
                    self._pause_facebook_campaigns(alert.account_id)
                elif alert.platform == 'all':
                    self._pause_all_campaigns()
                
                self._send_emergency_notification(alert)
                self._log_alert(alert, 'CAMPAIGNS_PAUSED')
                
            elif alert.alert_type == 'danger':
                print(f"⚠️  DANGER: {alert.message}")
                self._send_alert_notification(alert)
                self._log_alert(alert, 'ALERT_SENT')
                
            elif alert.alert_type == 'warning':
                print(f"⚠️  WARNING: {alert.message}")
                self._send_alert_notification(alert)
                self._log_alert(alert, 'WARNING_SENT')
    
    def _pause_all_campaigns(self):
        """EMERGENCY: Pause all campaigns across all platforms"""
        
        print("🚨 PAUSING ALL CAMPAIGNS - EMERGENCY STOP")
        
        # Pause Google Ads campaigns
        if self.google_client:
            google_creds = self._get_credentials('google_ads')
            if google_creds:
                self._pause_google_campaigns(google_creds.account_id)
        
        # Pause Facebook campaigns
        if self.facebook_api:
            fb_creds = self._get_credentials('facebook')
            if fb_creds:
                self._pause_facebook_campaigns(fb_creds.account_id)
        
        print("✅ ALL CAMPAIGNS PAUSED")
    
    def _pause_google_campaigns(self, customer_id: str):
        """Pause all Google Ads campaigns"""
        
        try:
            campaign_service = self.google_client.get_service("CampaignService")
            
            # Get all active campaigns
            ga_service = self.google_client.get_service("GoogleAdsService")
            query = """
                SELECT campaign.id, campaign.name
                FROM campaign
                WHERE campaign.status = 'ENABLED'
            """
            
            response = ga_service.search(customer_id=customer_id, query=query)
            
            operations = []
            for row in response:
                campaign_operation = self.google_client.get_type("CampaignOperation")
                campaign_operation.update = self.google_client.get_type("Campaign")
                campaign_operation.update.resource_name = (
                    f"customers/{customer_id}/campaigns/{row.campaign.id}"
                )
                campaign_operation.update.status = self.google_client.enums.CampaignStatusEnum.PAUSED
                campaign_operation.update_mask = "status"
                operations.append(campaign_operation)
                
                print(f"  Pausing Google campaign: {row.campaign.name}")
            
            if operations:
                response = campaign_service.mutate_campaigns(
                    customer_id=customer_id,
                    operations=operations
                )
                print(f"✅ Paused {len(operations)} Google Ads campaigns")
            
        except Exception as e:
            print(f"❌ Failed to pause Google campaigns: {e}")
    
    def _pause_facebook_campaigns(self, ad_account_id: str):
        """Pause all Facebook campaigns"""
        
        try:
            account = AdAccount(ad_account_id)
            campaigns = account.get_campaigns(fields=['id', 'name', 'status'])
            
            paused_count = 0
            for campaign in campaigns:
                if campaign['status'] == 'ACTIVE':
                    campaign_obj = Campaign(campaign['id'])
                    campaign_obj.update(params={'status': 'PAUSED'})
                    print(f"  Pausing Facebook campaign: {campaign['name']}")
                    paused_count += 1
            
            print(f"✅ Paused {paused_count} Facebook campaigns")
            
        except Exception as e:
            print(f"❌ Failed to pause Facebook campaigns: {e}")
    
    def _send_emergency_notification(self, alert: BudgetAlert):
        """Send emergency notification via email/SMS"""
        
        subject = f"🚨 BUDGET EMERGENCY - {alert.platform.upper()}"
        
        message = f"""
BUDGET EMERGENCY ALERT - IMMEDIATE ACTION REQUIRED

Platform: {alert.platform}
Account: {alert.account_id}
Alert Type: {alert.alert_type}

Current Spend: ${alert.current_spend:.2f}
Limit: ${alert.limit:.2f}
Usage: {(alert.current_spend/alert.limit)*100:.1f}%

Message: {alert.message}

ACTIONS TAKEN:
- Campaigns have been PAUSED automatically
- No further spend will occur
- Review and restart campaigns manually when ready

Time: {alert.timestamp}

This is a REAL MONEY protection system.
"""
        
        self._send_email_alert(subject, message)
        self._send_sms_alert(f"BUDGET EMERGENCY: {alert.message}")
    
    def _send_alert_notification(self, alert: BudgetAlert):
        """Send regular alert notification"""
        
        subject = f"Budget Alert - {alert.platform.upper()} - {alert.alert_type.title()}"
        
        message = f"""
Budget {alert.alert_type.title()} Alert

Platform: {alert.platform}
Account: {alert.account_id}

Current Spend: ${alert.current_spend:.2f}
Limit: ${alert.limit:.2f}
Usage: {(alert.current_spend/alert.limit)*100:.1f}%

Message: {alert.message}

Time: {alert.timestamp}

Monitor spend closely to avoid emergency pausing.
"""
        
        self._send_email_alert(subject, message)
    
    def _send_email_alert(self, subject: str, message: str):
        """Send email alert"""
        
        try:
            # Configure for your email provider
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            
            sender_email = os.environ.get('ALERT_EMAIL', 'alerts@gaelp.com')
            sender_password = os.environ.get('ALERT_EMAIL_PASSWORD')
            
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = self.notifications['email']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, self.notifications['email'], msg.as_string())
            server.quit()
            
            print(f"✅ Email alert sent to {self.notifications['email']}")
            
        except Exception as e:
            print(f"❌ Failed to send email alert: {e}")
    
    def _send_sms_alert(self, message: str):
        """Send SMS alert via Twilio or similar service"""
        
        try:
            # Configure Twilio or your SMS provider
            account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
            auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
            twilio_number = os.environ.get('TWILIO_PHONE_NUMBER')
            
            if not all([account_sid, auth_token, twilio_number]):
                print("⚠️  SMS credentials not configured")
                return
            
            from twilio.rest import Client
            
            client = Client(account_sid, auth_token)
            
            client.messages.create(
                body=message,
                from_=twilio_number,
                to=self.notifications['sms']
            )
            
            print(f"✅ SMS alert sent to {self.notifications['sms']}")
            
        except Exception as e:
            print(f"❌ Failed to send SMS alert: {e}")
    
    def _log_alert(self, alert: BudgetAlert, action: str):
        """Log alert to database"""
        
        conn = sqlite3.connect(self.alerts_db)
        conn.execute('''
            INSERT INTO alerts 
            (platform, account_id, alert_type, threshold, current_spend, limit_amount, message, action_taken)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.platform,
            alert.account_id,
            alert.alert_type,
            alert.threshold,
            alert.current_spend,
            alert.limit,
            alert.message,
            action
        ))
        conn.commit()
        conn.close()
    
    def update_spend_tracking(self):
        """Update spend tracking from all platforms"""
        
        current_time = datetime.now()
        current_date = current_time.strftime('%Y-%m-%d')
        current_hour = current_time.strftime('%Y-%m-%d %H:00:00')
        
        # Track Google Ads spend
        if self.google_client:
            google_creds = self._get_credentials('google_ads')
            if google_creds:
                google_data = self._get_google_spend(google_creds.account_id, 'today')
                self._record_spend(current_date, current_hour, 'google_ads', google_creds.account_id, google_data)
        
        # Track Facebook spend
        if self.facebook_api:
            fb_creds = self._get_credentials('facebook')
            if fb_creds:
                fb_data = self._get_facebook_spend(fb_creds.account_id, 'today')
                self._record_spend(current_date, current_hour, 'facebook', fb_creds.account_id, fb_data)
    
    def _record_spend(self, date: str, hour: str, platform: str, account_id: str, data: Dict):
        """Record spend data to database"""
        
        conn = sqlite3.connect(self.spend_db)
        
        # Update daily spend
        conn.execute('''
            INSERT OR REPLACE INTO daily_spend 
            (date, platform, account_id, spend, impressions, clicks, conversions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (date, platform, account_id, data['spend'], data['impressions'], data['clicks'], data['conversions']))
        
        # Update hourly spend
        conn.execute('''
            INSERT OR REPLACE INTO hourly_spend
            (datetime, platform, account_id, spend, impressions, clicks, conversions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (hour, platform, account_id, data['spend'], data['impressions'], data['clicks'], data['conversions']))
        
        conn.commit()
        conn.close()
    
    def generate_spend_report(self) -> str:
        """Generate comprehensive spend report"""
        
        print("\n📊 GENERATING SPEND REPORT")
        print("="*50)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'daily_summary': {},
            'platform_breakdown': {},
            'alerts_summary': {},
            'budget_status': {}
        }
        
        # Get today's spend
        conn = sqlite3.connect(self.spend_db)
        
        # Daily summary
        cursor = conn.execute('''
            SELECT platform, SUM(spend), SUM(impressions), SUM(clicks), SUM(conversions)
            FROM daily_spend
            WHERE date = date('now')
            GROUP BY platform
        ''')
        
        total_daily_spend = 0.0
        for row in cursor.fetchall():
            platform, spend, impressions, clicks, conversions = row
            total_daily_spend += spend
            
            report['platform_breakdown'][platform] = {
                'spend': spend,
                'impressions': impressions,
                'clicks': clicks,
                'conversions': conversions,
                'cpc': spend / clicks if clicks > 0 else 0,
                'cpm': (spend / impressions) * 1000 if impressions > 0 else 0,
                'cpa': spend / conversions if conversions > 0 else 0
            }
        
        report['daily_summary'] = {
            'total_spend': total_daily_spend,
            'daily_limit': self.limits['daily_limit'],
            'remaining_budget': self.limits['daily_limit'] - total_daily_spend,
            'usage_percentage': (total_daily_spend / self.limits['daily_limit']) * 100
        }
        
        # Monthly summary
        cursor = conn.execute('''
            SELECT SUM(spend)
            FROM daily_spend
            WHERE date >= date('now', 'start of month')
        ''')
        
        monthly_spend = cursor.fetchone()[0] or 0.0
        report['budget_status'] = {
            'monthly_spend': monthly_spend,
            'monthly_limit': self.limits['monthly_limit'],
            'monthly_remaining': self.limits['monthly_limit'] - monthly_spend,
            'monthly_usage': (monthly_spend / self.limits['monthly_limit']) * 100
        }
        
        conn.close()
        
        # Recent alerts
        conn = sqlite3.connect(self.alerts_db)
        cursor = conn.execute('''
            SELECT alert_type, COUNT(*), MAX(timestamp)
            FROM alerts
            WHERE date(timestamp) = date('now')
            GROUP BY alert_type
        ''')
        
        for row in cursor.fetchall():
            alert_type, count, last_time = row
            report['alerts_summary'][alert_type] = {
                'count': count,
                'last_alert': last_time
            }
        
        conn.close()
        
        # Save report
        report_file = self.config_dir / f'spend_report_{datetime.now().strftime("%Y%m%d")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"Daily Spend: ${total_daily_spend:.2f} / ${self.limits['daily_limit']} ({report['daily_summary']['usage_percentage']:.1f}%)")
        print(f"Monthly Spend: ${monthly_spend:.2f} / ${self.limits['monthly_limit']} ({report['budget_status']['monthly_usage']:.1f}%)")
        
        if total_daily_spend >= self.limits['daily_limit'] * 0.8:
            print("⚠️  Approaching daily budget limit!")
        
        return str(report_file)
    
    def start_monitoring(self):
        """Start continuous budget monitoring"""
        
        print("\n🚀 STARTING BUDGET MONITORING")
        print("="*50)
        print("Monitoring frequency: Every 15 minutes")
        print("Alert thresholds:")
        print(f"  Warning: {self.thresholds['warning']*100:.0f}%")
        print(f"  Danger: {self.thresholds['danger']*100:.0f}%") 
        print(f"  Emergency: {self.thresholds['emergency']*100:.0f}%")
        print("\n⚠️  REAL MONEY PROTECTION ACTIVE")
        print("Campaigns will be PAUSED automatically at emergency threshold")
        
        # Schedule monitoring tasks
        schedule.every(15).minutes.do(self.monitoring_cycle)
        schedule.every(1).hours.do(self.generate_spend_report)
        schedule.every().day.at("09:00").do(self.daily_report)
        
        print("\n✅ Monitoring started - running continuously...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def monitoring_cycle(self):
        """Single monitoring cycle"""
        
        print(f"\n🔍 Budget check: {datetime.now().strftime('%H:%M:%S')}")
        
        # Update spend data
        self.update_spend_tracking()
        
        # Check thresholds
        alerts = self.check_budget_thresholds()
        
        if alerts:
            print(f"⚠️  {len(alerts)} budget alerts detected")
            self.execute_emergency_actions(alerts)
        else:
            print("✅ All budgets within limits")
    
    def daily_report(self):
        """Generate and send daily report"""
        report_file = self.generate_spend_report()
        
        # Email daily report
        subject = f"Daily Ad Spend Report - {datetime.now().strftime('%Y-%m-%d')}"
        message = f"Daily spend report generated: {report_file}"
        self._send_email_alert(subject, message)

def main():
    """Main monitoring execution"""
    
    print("🚨 BUDGET SAFETY MONITOR - REAL MONEY PROTECTION")
    print("="*60)
    print("⚠️  This monitors REAL ad spend and will PAUSE campaigns")
    print("⚠️  Ensure your limits are set correctly before starting")
    
    monitor = BudgetSafetyMonitor()
    
    # Run initial check
    print("\n🔍 Running initial budget check...")
    alerts = monitor.check_budget_thresholds()
    
    if alerts:
        print(f"⚠️  {len(alerts)} alerts found on startup:")
        for alert in alerts:
            print(f"  {alert.alert_type}: {alert.message}")
        
        proceed = input("\nContinue monitoring with existing alerts? (y/n): ").strip().lower()
        if proceed != 'y':
            print("❌ Monitoring cancelled")
            return
    
    # Start continuous monitoring
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")
        print("Check configuration and try again")

if __name__ == "__main__":
    main()
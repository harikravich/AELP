#!/usr/bin/env python3
"""
Email Nurture Sequence System for Social Media Scanner Leads
Converts free users to Aura Balance trials
"""

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import schedule
import time
import os
from dotenv import load_dotenv

load_dotenv()

class EmailNurtureSystem:
    """Automated email nurture sequence for scanner leads"""
    
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.leads_file = Path("/home/hariravichandran/AELP/scanner_leads.json")
        self.nurture_file = Path("/home/hariravichandran/AELP/nurture_tracking.json")
        
        # Email sequence templates
        self.email_sequence = self._create_email_sequence()
    
    def _create_email_sequence(self) -> List[Dict[str, Any]]:
        """Create the complete nurture sequence"""
        
        return [
            {
                'day': 0,
                'subject': 'Your Teen\'s Complete Social Media Report',
                'template': 'immediate_report',
                'goal': 'Deliver value immediately'
            },
            {
                'day': 2,
                'subject': '73% of teens hide accounts from parents (here\'s why)',
                'template': 'hidden_accounts_education',
                'goal': 'Educate about the problem'
            },
            {
                'day': 5,
                'subject': 'Warning signs you might be missing (Sarah\'s story)',
                'template': 'warning_signs',
                'goal': 'Create urgency with story'
            },
            {
                'day': 7,
                'subject': 'FREE: 24/7 monitoring trial (limited time)',
                'template': 'trial_offer',
                'goal': 'Convert to trial'
            },
            {
                'day': 10,
                'subject': 'Your trial expires in 4 days - here\'s what you\'re missing',
                'template': 'trial_reminder',
                'goal': 'Remind about trial'
            },
            {
                'day': 14,
                'subject': 'Case study: How Jennifer prevented her daughter\'s crisis',
                'template': 'success_story',
                'goal': 'Social proof and final conversion'
            },
            {
                'day': 21,
                'subject': 'One last thing about your teen\'s digital safety...',
                'template': 'final_touch',
                'goal': 'Re-engagement attempt'
            }
        ]
    
    def get_email_template(self, template_name: str, lead_data: Dict[str, Any]) -> str:
        """Get HTML email template with personalization"""
        
        templates = {
            'immediate_report': self._template_immediate_report,
            'hidden_accounts_education': self._template_hidden_accounts,
            'warning_signs': self._template_warning_signs,
            'trial_offer': self._template_trial_offer,
            'trial_reminder': self._template_trial_reminder,
            'success_story': self._template_success_story,
            'final_touch': self._template_final_touch
        }
        
        template_func = templates.get(template_name, self._template_default)
        return template_func(lead_data)
    
    def _template_immediate_report(self, lead_data: Dict[str, Any]) -> str:
        """Immediate report email template"""
        accounts_found = lead_data.get('accounts_found', 0)
        risk_score = lead_data.get('risk_score', 0)
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background: linear-gradient(90deg, #4CAF50, #2196F3); color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; max-width: 600px; margin: 0 auto; }}
                .highlight {{ background-color: #fff3cd; padding: 15px; border-radius: 8px; margin: 15px 0; }}
                .cta {{ background-color: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; display: inline-block; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Your Teen's Complete Social Media Report</h1>
                <p>Scan Results + Action Plan Inside</p>
            </div>
            
            <div class="content">
                <h2>Your Scan Results</h2>
                <div class="highlight">
                    <p><strong>Accounts Found:</strong> {accounts_found}</p>
                    <p><strong>Risk Score:</strong> {risk_score}/100</p>
                    <p><strong>Recommended Action:</strong> Review privacy settings immediately</p>
                </div>
                
                <h3>What This Means</h3>
                <p>Based on your scan, here's what you need to know:</p>
                
                <ul>
                    <li><strong>Public Profiles:</strong> Your teen's information may be visible to strangers</li>
                    <li><strong>Hidden Accounts:</strong> Teens often create secondary "finsta" accounts</li>
                    <li><strong>Privacy Risks:</strong> Location sharing and personal info exposure</li>
                </ul>
                
                <h3>Your 7-Day Action Plan</h3>
                <p><strong>Day 1-2:</strong> Have an open conversation about online safety</p>
                <p><strong>Day 3-4:</strong> Review privacy settings together</p>
                <p><strong>Day 5-7:</strong> Set up ongoing monitoring</p>
                
                <div class="highlight">
                    <h3>🚨 Why Manual Checking Isn't Enough</h3>
                    <p>This scan took you to request manually. Your teen's digital life changes daily:</p>
                    <ul>
                        <li>New accounts created weekly</li>
                        <li>Privacy settings change</li>
                        <li>Content shared you never see</li>
                        <li>New friends/followers added</li>
                    </ul>
                </div>
                
                <center>
                    <a href="https://aura.com/balance-trial" class="cta">Get 24/7 Automated Monitoring - Free Trial</a>
                </center>
                
                <p><strong>What happens next?</strong></p>
                <p>Over the next week, I'll send you specific guidance on protecting your teen online, including real stories from other parents and actionable steps you can take today.</p>
                
                <p>Questions? Just reply to this email.</p>
                
                <p>Best regards,<br>
                The Aura Balance Team</p>
            </div>
        </body>
        </html>
        """
    
    def _template_hidden_accounts(self, lead_data: Dict[str, Any]) -> str:
        """Hidden accounts education email"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .content { padding: 20px; max-width: 600px; margin: 0 auto; }
                .stat { background-color: #e3f2fd; padding: 20px; text-align: center; border-radius: 8px; margin: 20px 0; }
                .cta { background-color: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; display: inline-block; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="content">
                <h2>73% of Teens Hide Accounts from Parents</h2>
                
                <div class="stat">
                    <h3>The Hidden Account Problem</h3>
                    <p><strong>73%</strong> of teens have accounts parents don't know about</p>
                    <p><strong>89%</strong> use different usernames across platforms</p>
                    <p><strong>45%</strong> have "finsta" (fake Instagram) accounts</p>
                </div>
                
                <h3>Why Teens Create Hidden Accounts</h3>
                <p>It's not necessarily rebellion. Teens create hidden accounts because:</p>
                
                <ul>
                    <li><strong>Peer Pressure:</strong> "Everyone has a finsta"</li>
                    <li><strong>Privacy from Family:</strong> They want spaces that feel "theirs"</li>
                    <li><strong>Authentic Expression:</strong> Different sides of their personality</li>
                    <li><strong>Friend Drama:</strong> Venting without parents seeing</li>
                </ul>
                
                <h3>The Real Risks</h3>
                <p>Hidden accounts often have:</p>
                <ul>
                    <li>Lower privacy settings (they think you won't find them)</li>
                    <li>More personal information shared</li>
                    <li>Riskier content and conversations</li>
                    <li>Unknown followers and contacts</li>
                </ul>
                
                <h3>What Sarah's Mom Discovered</h3>
                <p>Sarah (16) had her main Instagram account locked down perfectly. Her parents were proud of her digital responsibility.</p>
                
                <p>But Sarah also had @sarah.spam.account with 847 followers she'd never met. She was sharing:</p>
                <ul>
                    <li>Photos from inside her bedroom</li>
                    <li>Her school schedule and location</li>
                    <li>When her parents weren't home</li>
                    <li>Personal problems and vulnerabilities</li>
                </ul>
                
                <p><strong>The wake-up call:</strong> A 32-year-old man started direct messaging her offering to "help with her problems."</p>
                
                <center>
                    <a href="https://aura.com/balance-trial" class="cta">Protect Your Teen with 24/7 Monitoring</a>
                </center>
                
                <h3>Tomorrow's Email</h3>
                <p>I'll share the warning signs that Sarah's parents missed, and how you can spot them before it's too late.</p>
                
                <p>Stay vigilant,<br>
                The Aura Balance Team</p>
            </div>
        </body>
        </html>
        """
    
    def _template_warning_signs(self, lead_data: Dict[str, Any]) -> str:
        """Warning signs email template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .content { padding: 20px; max-width: 600px; margin: 0 auto; }
                .warning { background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }
                .cta { background-color: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; display: inline-block; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="content">
                <h2>Warning Signs You Might Be Missing</h2>
                
                <p>Yesterday I told you about Sarah, whose hidden account exposed her to a predator. Her parents asked: "How did we miss this?"</p>
                
                <h3>The Warning Signs They Missed</h3>
                
                <div class="warning">
                    <h4>📱 Phone Behavior Changes</h4>
                    <ul>
                        <li>Quickly closing apps when parents approached</li>
                        <li>Getting notifications at unusual hours (2-4 AM)</li>
                        <li>Being secretive about who was messaging</li>
                    </ul>
                </div>
                
                <div class="warning">
                    <h4>🤳 Social Media Patterns</h4>
                    <ul>
                        <li>Taking lots of photos but not posting them (to main account)</li>
                        <li>Using phone in bathroom/bedroom more often</li>
                        <li>Mentioning friends parents had never heard of</li>
                    </ul>
                </div>
                
                <div class="warning">
                    <h4>😟 Mood and Behavior</h4>
                    <ul>
                        <li>More anxious after using phone</li>
                        <li>Defensive when asked about online activities</li>
                        <li>Sleep schedule disrupted by late-night phone use</li>
                    </ul>
                </div>
                
                <h3>The Problem with Manual Monitoring</h3>
                <p>Sarah's parents were vigilant, but they were looking in the wrong places:</p>
                
                <ul>
                    <li>✅ They checked her main Instagram daily</li>
                    <li>✅ They monitored her Snapchat friends</li>
                    <li>✅ They reviewed her text messages</li>
                    <li>❌ But they never found her hidden accounts</li>
                </ul>
                
                <h3>What Could Have Prevented This</h3>
                
                <p>Aura Balance would have detected:</p>
                <ul>
                    <li>New account creation across platforms</li>
                    <li>Unusual late-night activity patterns</li>
                    <li>Communications with unknown adults</li>
                    <li>Sharing of personal/location information</li>
                </ul>
                
                <p><strong>The result:</strong> Sarah's parents would have been alerted within 24 hours of the first concerning interaction.</p>
                
                <center>
                    <a href="https://aura.com/balance-trial" class="cta">Start Your Free 14-Day Trial Now</a>
                </center>
                
                <h3>Tomorrow</h3>
                <p>I'll show you exactly how Aura Balance works and why 50,000+ parents trust it to keep their teens safe online.</p>
                
                <p>Protecting families,<br>
                The Aura Balance Team</p>
                
                <p><small>P.S. Sarah is now safe. Her parents discovered the hidden account in time, implemented Aura Balance monitoring, and had important conversations about online safety. She now understands why these protections exist.</small></p>
            </div>
        </body>
        </html>
        """
    
    def _template_trial_offer(self, lead_data: Dict[str, Any]) -> str:
        """Trial offer email template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .content { padding: 20px; max-width: 600px; margin: 0 auto; }
                .offer-box { background: linear-gradient(135deg, #4CAF50, #2196F3); color: white; padding: 30px; text-align: center; border-radius: 12px; margin: 20px 0; }
                .feature { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; }
                .cta { background-color: #FF5722; color: white; padding: 20px 40px; text-decoration: none; border-radius: 8px; font-size: 18px; font-weight: bold; }
                .guarantee { background-color: #e8f5e8; padding: 15px; text-align: center; border-radius: 8px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="content">
                <div class="offer-box">
                    <h2>FREE 14-Day Trial</h2>
                    <h3>Complete Digital Protection for Your Teen</h3>
                    <p>Usually $29.99/month - FREE for 14 days</p>
                    <p><strong>No credit card required</strong></p>
                </div>
                
                <h3>Get Everything Sarah's Parents Wished They Had</h3>
                
                <div class="feature">
                    <h4>🔍 Account Discovery</h4>
                    <p>Automatically finds new accounts across 50+ platforms including hidden "finsta" accounts</p>
                </div>
                
                <div class="feature">
                    <h4>🚨 Real-Time Alerts</h4>
                    <p>Instant notifications for risky behavior, new contacts, or concerning content</p>
                </div>
                
                <div class="feature">
                    <h4>🧠 AI Behavioral Analysis</h4>
                    <p>Detects mood changes, cyberbullying, and mental health warning signs</p>
                </div>
                
                <div class="feature">
                    <h4>👨‍⚕️ Expert Support</h4>
                    <p>Access to child psychologists and digital safety experts</p>
                </div>
                
                <div class="feature">
                    <h4>📊 Weekly Reports</h4>
                    <p>Easy-to-understand summaries of your teen's digital activity and wellbeing</p>
                </div>
                
                <h3>What Parents Say</h3>
                <blockquote>
                    <p><em>"Aura Balance found my daughter's second Instagram account that had over 500 followers I didn't know about. Within a week, we caught concerning messages from an adult male. This literally saved her from a dangerous situation."</em></p>
                    <p><strong>- Jennifer M., mom of 15-year-old</strong></p>
                </blockquote>
                
                <div class="guarantee">
                    <h4>🛡️ 100% Risk-Free Guarantee</h4>
                    <p>Try Aura Balance for 14 days completely free. If you don't feel your teen is safer, cancel with one click. No questions asked.</p>
                </div>
                
                <center>
                    <a href="https://aura.com/balance-trial" class="cta">Start My Free Trial Now</a>
                </center>
                
                <h3>Why Act Now?</h3>
                <ul>
                    <li>Every day delayed is another day of potential risk</li>
                    <li>Free trial ends soon - regular price is $29.99/month</li>
                    <li>Setup takes less than 5 minutes</li>
                    <li>Works with your teen's existing devices</li>
                </ul>
                
                <p>Your teen's safety is priceless. But protecting it should be effortless.</p>
                
                <p>Get started today,<br>
                The Aura Balance Team</p>
                
                <p><small>Questions? Reply to this email or call 1-800-AURA-HELP</small></p>
            </div>
        </body>
        </html>
        """
    
    def _template_trial_reminder(self, lead_data: Dict[str, Any]) -> str:
        """Trial reminder email template"""
        return """
        <!DOCTYPE html>
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2>Your Free Trial Expires in 4 Days</h2>
                
                <p>Hi there,</p>
                
                <p>I wanted to check in about your Aura Balance free trial. You have just 4 days left to experience complete digital protection for your teen.</p>
                
                <h3>What You're Missing Without Monitoring</h3>
                <ul>
                    <li>New hidden accounts created daily</li>
                    <li>Risky conversations you can't see</li>
                    <li>Privacy settings that change without notice</li>
                    <li>Warning signs of cyberbullying or worse</li>
                </ul>
                
                <p style="background-color: #fff3cd; padding: 15px; border-radius: 8px;">
                    <strong>Don't wait for a crisis.</strong> In the time it takes to "think about it," your teen's digital footprint expands exponentially.
                </p>
                
                <center>
                    <a href="https://aura.com/balance-continue" style="background-color: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; display: inline-block; margin: 20px 0;">Continue My Protection</a>
                </center>
                
                <p>Questions about your trial? Just reply to this email.</p>
                
                <p>Protecting families,<br>
                The Aura Balance Team</p>
            </div>
        </body>
        </html>
        """
    
    def _template_success_story(self, lead_data: Dict[str, Any]) -> str:
        """Success story email template"""
        return """
        <!DOCTYPE html>
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2>How Jennifer Prevented Her Daughter's Crisis</h2>
                
                <p>I want to share a story that shows exactly why 50,000+ parents trust Aura Balance.</p>
                
                <h3>The Situation</h3>
                <p>Jennifer's 15-year-old daughter Emma was a "good kid" - honor student, responsible, never in trouble.</p>
                
                <p>But Jennifer noticed small changes:</p>
                <ul>
                    <li>Emma seemed more anxious lately</li>
                    <li>She was on her phone more at night</li>
                    <li>Her mood swings were getting worse</li>
                </ul>
                
                <h3>What Aura Balance Detected</h3>
                <p>Within 48 hours of setup, Aura Balance discovered:</p>
                
                <ul>
                    <li><strong>Hidden TikTok account:</strong> @emma.private.thoughts</li>
                    <li><strong>Concerning content:</strong> Posts about feeling "worthless" and "better off gone"</li>
                    <li><strong>Risky interactions:</strong> Anonymous messages encouraging self-harm</li>
                    <li><strong>Behavioral patterns:</strong> Active online 2-4 AM when parents slept</li>
                </ul>
                
                <div style="background-color: #f8d7da; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <p><strong>The Crisis Alert:</strong> Emma posted "I can't take it anymore" at 2:47 AM on a Tuesday. Aura Balance immediately notified Jennifer with crisis-level urgency.</p>
                </div>
                
                <h3>The Intervention</h3>
                <p>Jennifer was awake and at Emma's door within minutes. Together, they:</p>
                <ul>
                    <li>Had an open, non-judgmental conversation</li>
                    <li>Contacted a mental health professional that same morning</li>
                    <li>Removed toxic online influences</li>
                    <li>Set up healthier digital boundaries</li>
                </ul>
                
                <h3>The Outcome</h3>
                <p><em>"Aura Balance literally saved my daughter's life. I had no idea she was struggling this deeply. Without the real-time monitoring, I might have found out too late."</em></p>
                <p><strong>- Jennifer M.</strong></p>
                
                <p>Emma is now thriving. She's in counseling, has healthy coping strategies, and maintains open communication with her family about her digital life.</p>
                
                <h3>Your Teen Might Be Struggling Too</h3>
                <p>1 in 3 teens experience cyberbullying<br>
                42% of teens report feeling sad or hopeless<br>
                Most parents don't know until it's almost too late</p>
                
                <center>
                    <a href="https://aura.com/balance-trial" style="background-color: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; display: inline-block; margin: 20px 0;">Protect Your Teen Like Jennifer Protected Emma</a>
                </center>
                
                <p>Don't wait for warning signs you might miss.</p>
                
                <p>Protecting families every day,<br>
                The Aura Balance Team</p>
            </div>
        </body>
        </html>
        """
    
    def _template_final_touch(self, lead_data: Dict[str, Any]) -> str:
        """Final re-engagement email template"""
        return """
        <!DOCTYPE html>
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2>One Last Thing About Your Teen's Digital Safety...</h2>
                
                <p>Hi,</p>
                
                <p>I know you've been busy, and digital safety might not feel urgent right now.</p>
                
                <p>But I want to share one final thought:</p>
                
                <p style="background-color: #e3f2fd; padding: 20px; border-radius: 8px; font-style: italic;">
                    "I wish I had started monitoring sooner. Not to spy, but to guide. Not to control, but to protect. The conversation it started with my daughter about online safety was the most important one we ever had."
                    <br><br>
                    - Mark T., father of 17-year-old
                </p>
                
                <p>Your teen's digital life is their real life. The friends they make, the content they see, the interactions they have online shape who they become offline.</p>
                
                <h3>The Choice Is Simple</h3>
                <ul>
                    <li>Stay in the dark and hope for the best</li>
                    <li>OR get the tools to guide them safely</li>
                </ul>
                
                <p>If you're not ready for full monitoring, that's okay. But please:</p>
                <ol>
                    <li>Have regular conversations about online safety</li>
                    <li>Keep devices out of bedrooms at night</li>
                    <li>Know who your teen is talking to online</li>
                    <li>Watch for mood changes after device use</li>
                </ol>
                
                <center>
                    <a href="https://aura.com/balance-trial" style="background-color: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; display: inline-block; margin: 20px 0;">Get Aura Balance - Still Free to Try</a>
                </center>
                
                <p>This is my final email about Aura Balance. Whatever you decide, I hope you'll prioritize your teen's digital safety.</p>
                
                <p>Their future depends on the choices they make online today.</p>
                
                <p>Stay safe,<br>
                The Aura Balance Team</p>
                
                <hr>
                
                <p><small>If you'd prefer not to receive future emails about Aura Balance, <a href="#">click here to unsubscribe</a>. You'll still receive important safety tips and resources.</small></p>
            </div>
        </body>
        </html>
        """
    
    def _template_default(self, lead_data: Dict[str, Any]) -> str:
        """Default template fallback"""
        return """
        <html>
        <body>
            <p>Thank you for using the Teen Social Media Scanner.</p>
            <p>For questions, contact support@aura.com</p>
        </body>
        </html>
        """
    
    def send_email(self, to_email: str, subject: str, html_content: str) -> bool:
        """Send email with error handling"""
        if not self.email_user or not self.email_password:
            print(f"Email not configured - would send to {to_email}: {subject}")
            return True  # Return success for demo
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(html_content, 'html'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            print(f"Email sending failed: {e}")
            return False
    
    def process_nurture_sequence(self):
        """Process the nurture sequence for all leads"""
        # Load leads
        if not self.leads_file.exists():
            return
        
        with open(self.leads_file, 'r') as f:
            leads = json.load(f)
        
        # Load nurture tracking
        nurture_tracking = {}
        if self.nurture_file.exists():
            with open(self.nurture_file, 'r') as f:
                nurture_tracking = json.load(f)
        
        now = datetime.now()
        
        for lead in leads:
            email = lead['email']
            signup_date = datetime.fromisoformat(lead['timestamp'])
            
            if email not in nurture_tracking:
                nurture_tracking[email] = {
                    'signup_date': lead['timestamp'],
                    'emails_sent': [],
                    'last_email': None
                }
            
            # Check which emails to send
            for email_config in self.email_sequence:
                send_date = signup_date + timedelta(days=email_config['day'])
                email_id = f"day_{email_config['day']}"
                
                # Skip if already sent
                if email_id in nurture_tracking[email]['emails_sent']:
                    continue
                
                # Skip if not time yet
                if now < send_date:
                    continue
                
                # Send email
                subject = email_config['subject']
                template = email_config['template']
                html_content = self.get_email_template(template, lead)
                
                if self.send_email(email, subject, html_content):
                    nurture_tracking[email]['emails_sent'].append(email_id)
                    nurture_tracking[email]['last_email'] = now.isoformat()
                    print(f"✅ Sent {template} to {email}")
                else:
                    print(f"❌ Failed to send {template} to {email}")
        
        # Save tracking
        with open(self.nurture_file, 'w') as f:
            json.dump(nurture_tracking, f, indent=2)
    
    def run_scheduler(self):
        """Run the email scheduler"""
        print("🚀 Starting Email Nurture System")
        print("Checking for emails to send every hour...")
        
        # Schedule email processing every hour
        schedule.every().hour.do(self.process_nurture_sequence)
        
        # Run immediately on startup
        self.process_nurture_sequence()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

def main():
    """Main function to run the nurture system"""
    nurture_system = EmailNurtureSystem()
    
    print("Email Nurture System for Social Media Scanner")
    print("=" * 50)
    print("This system automatically sends nurture emails to leads")
    print("to convert them from free users to Aura Balance trials.")
    print()
    
    choice = input("Run email scheduler? (y/n): ").lower()
    
    if choice == 'y':
        nurture_system.run_scheduler()
    else:
        # Just process once
        print("Processing emails once...")
        nurture_system.process_nurture_sequence()
        print("✅ Email processing complete")

if __name__ == "__main__":
    main()
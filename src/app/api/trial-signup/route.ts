import { NextRequest, NextResponse } from 'next/server';

interface TrialSignupRequest {
  email: string;
  phone?: string;
  scanId?: string;
  source?: string;
  utm?: {
    campaign?: string;
    medium?: string;
    source?: string;
    term?: string;
    content?: string;
  };
}

interface TrialSignupResponse {
  success: boolean;
  trialId: string;
  message: string;
  nextSteps?: string[];
  errors?: string[];
}

// Email validation regex
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

// Phone validation regex (US format)
const PHONE_REGEX = /^(\+1|1)?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$/;

class TrialSignupService {
  async validateSignup(data: TrialSignupRequest): Promise<string[]> {
    const errors: string[] = [];
    
    // Email validation
    if (!data.email) {
      errors.push('Email address is required');
    } else if (!EMAIL_REGEX.test(data.email)) {
      errors.push('Please enter a valid email address');
    }
    
    // Phone validation (optional but validate if provided)
    if (data.phone && !PHONE_REGEX.test(data.phone.replace(/\s/g, ''))) {
      errors.push('Please enter a valid phone number');
    }
    
    // Check for disposable email domains
    const disposableDomains = [
      '10minutemail.com',
      'tempmail.org',
      'guerrillamail.com',
      'mailinator.com',
      'yopmail.com'
    ];
    
    const domain = data.email.split('@')[1]?.toLowerCase();
    if (domain && disposableDomains.includes(domain)) {
      errors.push('Please use a permanent email address');
    }
    
    return errors;
  }

  async checkExistingTrial(email: string): Promise<boolean> {
    // In production, check database for existing trials
    // For now, simulate some existing emails
    const existingEmails = [
      'test@example.com',
      'demo@test.com'
    ];
    
    return existingEmails.includes(email.toLowerCase());
  }

  async createTrial(data: TrialSignupRequest): Promise<string> {
    // Generate trial ID
    const trialId = `trial_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // In production, save to database and trigger welcome email sequence
    const trialData = {
      id: trialId,
      email: data.email,
      phone: data.phone,
      scanId: data.scanId,
      source: data.source || 'free-social-scan',
      utm: data.utm,
      createdAt: new Date().toISOString(),
      status: 'active',
      expiresAt: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString(), // 14 days
    };
    
    console.log('New trial signup:', trialData);
    
    // In production, integrate with:
    // - Email service (SendGrid, Mailchimp)
    // - CRM (HubSpot, Salesforce)
    // - Analytics (Mixpanel, Amplitude)
    // - Customer.io for onboarding sequences
    
    return trialId;
  }

  async sendWelcomeSequence(email: string, trialId: string): Promise<void> {
    // In production, trigger welcome email sequence
    console.log(`Triggering welcome sequence for ${email}, trial: ${trialId}`);
    
    // Example email sequence:
    // 1. Immediate: Setup instructions + app download
    // 2. Day 1: "How to connect your teen's device"
    // 3. Day 3: "Understanding your first insights"
    // 4. Day 7: "Success stories from other parents"
    // 5. Day 12: "Don't lose protection - convert today"
  }

  getNextSteps(): string[] {
    return [
      'Check your email for setup instructions',
      'Download the Aura Balance app from the App Store',
      'Follow the device setup guide we\'ve sent you',
      'Complete the initial monitoring setup',
      'Review your first behavioral insights within 24 hours'
    ];
  }
}

export async function POST(request: NextRequest) {
  try {
    const body: TrialSignupRequest = await request.json();
    const signupService = new TrialSignupService();
    
    // Validate the signup data
    const validationErrors = await signupService.validateSignup(body);
    if (validationErrors.length > 0) {
      return NextResponse.json({
        success: false,
        message: 'Please correct the following errors:',
        errors: validationErrors
      }, { status: 400 });
    }
    
    // Check for existing trial
    const hasExistingTrial = await signupService.checkExistingTrial(body.email);
    if (hasExistingTrial) {
      return NextResponse.json({
        success: false,
        message: 'An account with this email already exists. Please check your inbox for setup instructions.',
        errors: ['Email already registered for trial']
      }, { status: 409 });
    }
    
    // Create the trial
    const trialId = await signupService.createTrial(body);
    
    // Send welcome sequence
    await signupService.sendWelcomeSequence(body.email, trialId);
    
    // Log conversion for analytics
    console.log('Trial conversion:', {
      trialId,
      email: body.email,
      scanId: body.scanId,
      source: body.source,
      utm: body.utm,
      timestamp: new Date().toISOString(),
      userAgent: request.headers.get('user-agent'),
      referer: request.headers.get('referer'),
    });
    
    const response: TrialSignupResponse = {
      success: true,
      trialId,
      message: 'Your free trial has been activated! Check your email for setup instructions.',
      nextSteps: signupService.getNextSteps()
    };
    
    return NextResponse.json(response);
    
  } catch (error) {
    console.error('Trial signup error:', error);
    return NextResponse.json({
      success: false,
      message: 'Something went wrong. Please try again or contact support.',
      errors: ['Internal server error']
    }, { status: 500 });
  }
}

// Handle preflight requests
export async function OPTIONS() {
  return NextResponse.json({}, { status: 200 });
}
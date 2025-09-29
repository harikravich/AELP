import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { provider, config } = await req.json();

    console.log('Testing integration:', provider, config);

    let testResult = false;

    switch (provider) {
      case 'instagram':
        testResult = await testInstagramAPI(config);
        break;
      case 'email':
        testResult = await testEmailService(config);
        break;
      case 'sms':
        testResult = await testSMSService(config);
        break;
      case 'webhook':
        testResult = await testWebhook(config);
        break;
      case 'analytics':
        testResult = await testAnalytics(config);
        break;
      default:
        throw new Error(`Unknown integration provider: ${provider}`);
    }

    return new Response(
      JSON.stringify({ 
        success: testResult,
        provider,
        message: testResult ? 'Integration test successful' : 'Integration test failed'
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  } catch (error) {
    console.error('Error testing integration:', error);
    return new Response(
      JSON.stringify({ 
        success: false, 
        error: error.message 
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});

async function testInstagramAPI(config: any): Promise<boolean> {
  try {
    // Test Instagram API connection
    if (!config.access_token) {
      throw new Error('Access token is required');
    }

    const response = await fetch(`https://graph.instagram.com/me?access_token=${config.access_token}`);
    
    if (!response.ok) {
      throw new Error('Instagram API test failed');
    }

    const data = await response.json();
    console.log('Instagram API test successful:', data.id);
    return true;
  } catch (error) {
    console.error('Instagram API test failed:', error);
    return false;
  }
}

async function testEmailService(config: any): Promise<boolean> {
  try {
    // Test email service based on provider
    switch (config.provider) {
      case 'sendgrid':
        return await testSendGrid(config);
      case 'mailgun':
        return await testMailgun(config);
      case 'resend':
        return await testResend(config);
      default:
        throw new Error(`Unknown email provider: ${config.provider}`);
    }
  } catch (error) {
    console.error('Email service test failed:', error);
    return false;
  }
}

async function testSendGrid(config: any): Promise<boolean> {
  try {
    const response = await fetch('https://api.sendgrid.com/v3/user/profile', {
      headers: {
        'Authorization': `Bearer ${config.api_key}`,
        'Content-Type': 'application/json'
      }
    });

    return response.ok;
  } catch (error) {
    console.error('SendGrid test failed:', error);
    return false;
  }
}

async function testMailgun(config: any): Promise<boolean> {
  try {
    // Test Mailgun API
    const domain = config.domain || 'sandbox-123.mailgun.org';
    const response = await fetch(`https://api.mailgun.net/v3/${domain}/stats/total`, {
      headers: {
        'Authorization': `Basic ${btoa(`api:${config.api_key}`)}`
      }
    });

    return response.ok;
  } catch (error) {
    console.error('Mailgun test failed:', error);
    return false;
  }
}

async function testResend(config: any): Promise<boolean> {
  try {
    const response = await fetch('https://api.resend.com/domains', {
      headers: {
        'Authorization': `Bearer ${config.api_key}`,
        'Content-Type': 'application/json'
      }
    });

    return response.ok;
  } catch (error) {
    console.error('Resend test failed:', error);
    return false;
  }
}

async function testSMSService(config: any): Promise<boolean> {
  try {
    // Test Twilio SMS service
    const accountSid = config.account_sid;
    const authToken = config.auth_token;
    
    if (!accountSid || !authToken) {
      throw new Error('Account SID and Auth Token are required');
    }

    const response = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${accountSid}.json`, {
      headers: {
        'Authorization': `Basic ${btoa(`${accountSid}:${authToken}`)}`
      }
    });

    return response.ok;
  } catch (error) {
    console.error('SMS service test failed:', error);
    return false;
  }
}

async function testWebhook(config: any): Promise<boolean> {
  try {
    if (!config.endpoint_url) {
      throw new Error('Endpoint URL is required');
    }

    const headers: any = {
      'Content-Type': 'application/json'
    };

    // Add custom headers if provided
    if (config.headers) {
      try {
        const customHeaders = JSON.parse(config.headers);
        Object.assign(headers, customHeaders);
      } catch (e) {
        console.warn('Invalid JSON in custom headers, ignoring');
      }
    }

    // Add secret key if provided
    if (config.secret_key) {
      headers['X-Webhook-Secret'] = config.secret_key;
    }

    const response = await fetch(config.endpoint_url, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        test: true,
        timestamp: new Date().toISOString(),
        message: 'Test webhook from landing page builder'
      })
    });

    // Accept 2xx status codes as successful
    return response.status >= 200 && response.status < 300;
  } catch (error) {
    console.error('Webhook test failed:', error);
    return false;
  }
}

async function testAnalytics(config: any): Promise<boolean> {
  try {
    // For analytics, we just validate the configuration format
    // Real validation would require actual API calls to each service
    
    let hasValidConfig = false;

    if (config.google_analytics_id) {
      // Basic GA4 measurement ID format check
      hasValidConfig = /^G-[A-Z0-9]+$/.test(config.google_analytics_id);
    }

    if (config.facebook_pixel_id) {
      // Basic Facebook Pixel ID format check
      hasValidConfig = /^\d+$/.test(config.facebook_pixel_id);
    }

    if (config.hotjar_id) {
      // Basic Hotjar site ID format check
      hasValidConfig = /^\d+$/.test(config.hotjar_id);
    }

    return hasValidConfig;
  } catch (error) {
    console.error('Analytics test failed:', error);
    return false;
  }
}
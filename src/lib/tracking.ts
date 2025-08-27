// Comprehensive conversion tracking for behavioral health landing pages
// Integrates with GA4, Facebook Pixel, and custom analytics

interface ConversionEvent {
  event_name: string;
  value?: number;
  currency?: string;
  transaction_id?: string;
  items?: Array<{
    item_id: string;
    item_name: string;
    category: string;
    quantity: number;
    price: number;
  }>;
  custom_parameters?: Record<string, any>;
}

interface UTMParameters {
  utm_source?: string;
  utm_medium?: string;
  utm_campaign?: string;
  utm_term?: string;
  utm_content?: string;
}

class BehavioralHealthTracking {
  private isClient: boolean;
  
  constructor() {
    this.isClient = typeof window !== 'undefined';
  }

  // Extract UTM parameters from URL
  getUTMParameters(): UTMParameters {
    if (!this.isClient) return {};
    
    const urlParams = new URLSearchParams(window.location.search);
    return {
      utm_source: urlParams.get('utm_source') || undefined,
      utm_medium: urlParams.get('utm_medium') || undefined,
      utm_campaign: urlParams.get('utm_campaign') || undefined,
      utm_term: urlParams.get('utm_term') || undefined,
      utm_content: urlParams.get('utm_content') || undefined,
    };
  }

  // Store attribution data in session
  storeAttribution(): void {
    if (!this.isClient) return;
    
    const utmParams = this.getUTMParameters();
    const attribution = {
      ...utmParams,
      referrer: document.referrer,
      landing_page: window.location.pathname,
      timestamp: new Date().toISOString(),
      session_id: this.getOrCreateSessionId(),
    };
    
    sessionStorage.setItem('attribution', JSON.stringify(attribution));
  }

  // Get or create session ID for tracking
  private getOrCreateSessionId(): string {
    if (!this.isClient) return '';
    
    let sessionId = sessionStorage.getItem('session_id');
    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      sessionStorage.setItem('session_id', sessionId);
    }
    return sessionId;
  }

  // Track page view with behavioral health context
  trackPageView(pageName: string, additionalData: Record<string, any> = {}): void {
    if (!this.isClient) return;
    
    const attribution = this.getStoredAttribution();
    const eventData = {
      page_title: document.title,
      page_location: window.location.href,
      page_path: window.location.pathname,
      behavioral_health_page: pageName,
      session_id: this.getOrCreateSessionId(),
      ...attribution,
      ...additionalData,
    };

    // GA4 tracking
    this.trackGA4('page_view', eventData);
    
    // Facebook Pixel
    this.trackFacebookPixel('PageView', eventData);
    
    // Custom analytics
    this.trackCustomEvent('page_view', eventData);
    
    console.log('Page view tracked:', eventData);
  }

  // Track quiz/form interactions
  trackFormInteraction(formType: string, step: string, additionalData: Record<string, any> = {}): void {
    if (!this.isClient) return;
    
    const eventData = {
      form_type: formType,
      form_step: step,
      session_id: this.getOrCreateSessionId(),
      timestamp: new Date().toISOString(),
      ...this.getStoredAttribution(),
      ...additionalData,
    };

    this.trackGA4('form_interaction', eventData);
    this.trackFacebookPixel('InitiateCheckout', eventData);
    this.trackCustomEvent('form_interaction', eventData);
  }

  // Track social scanner usage
  trackScanInitiated(platform: string, handleLength: number): void {
    if (!this.isClient) return;
    
    const eventData = {
      platform,
      handle_length: handleLength,
      session_id: this.getOrCreateSessionId(),
      ...this.getStoredAttribution(),
    };

    this.trackGA4('scan_initiated', eventData);
    this.trackFacebookPixel('Search', eventData);
    this.trackCustomEvent('scan_initiated', eventData);
  }

  // Track scan results viewed
  trackScanResults(accountsFound: number, riskScore: number): void {
    if (!this.isClient) return;
    
    const eventData = {
      accounts_found: accountsFound,
      risk_score: riskScore,
      high_risk: riskScore >= 7,
      session_id: this.getOrCreateSessionId(),
      ...this.getStoredAttribution(),
    };

    this.trackGA4('scan_results_viewed', eventData);
    this.trackFacebookPixel('ViewContent', {
      content_type: 'scan_results',
      value: riskScore,
      ...eventData,
    });
    this.trackCustomEvent('scan_results_viewed', eventData);
  }

  // Track trial signup conversion
  trackTrialConversion(trialId: string, value: number = 32.00): void {
    if (!this.isClient) return;
    
    const attribution = this.getStoredAttribution();
    const conversionData = {
      transaction_id: trialId,
      value: value,
      currency: 'USD',
      items: [{
        item_id: 'aura-balance-trial',
        item_name: 'Aura Balance 14-Day Free Trial',
        category: 'parental-monitoring',
        quantity: 1,
        price: value,
      }],
      session_id: this.getOrCreateSessionId(),
      ...attribution,
    };

    // GA4 Enhanced Ecommerce
    this.trackGA4('purchase', conversionData);
    
    // Google Ads Conversion
    this.trackGoogleAdsConversion(trialId, value);
    
    // Facebook Pixel Purchase
    this.trackFacebookPixel('Purchase', {
      value: value,
      currency: 'USD',
      content_type: 'product',
      contents: conversionData.items,
    });
    
    // Custom conversion tracking
    this.trackCustomEvent('trial_conversion', conversionData);
    
    console.log('Trial conversion tracked:', conversionData);
  }

  // Track CTA clicks with context
  trackCTAClick(ctaText: string, location: string, additionalData: Record<string, any> = {}): void {
    if (!this.isClient) return;
    
    const eventData = {
      cta_text: ctaText,
      cta_location: location,
      page_path: window.location.pathname,
      session_id: this.getOrCreateSessionId(),
      ...this.getStoredAttribution(),
      ...additionalData,
    };

    this.trackGA4('cta_click', eventData);
    this.trackFacebookPixel('AddToCart', eventData);
    this.trackCustomEvent('cta_click', eventData);
  }

  // Internal tracking methods
  private trackGA4(eventName: string, parameters: Record<string, any>): void {
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('event', eventName, parameters);
    }
  }

  private trackGoogleAdsConversion(transactionId: string, value: number): void {
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('event', 'conversion', {
        'send_to': 'AW-CONVERSION_ID/TRIAL_SIGNUP_LABEL',
        'value': value,
        'currency': 'USD',
        'transaction_id': transactionId,
      });
    }
  }

  private trackFacebookPixel(eventName: string, parameters: Record<string, any> = {}): void {
    if (typeof window !== 'undefined' && (window as any).fbq) {
      (window as any).fbq('track', eventName, parameters);
    }
  }

  private trackCustomEvent(eventName: string, data: Record<string, any>): void {
    // Send to custom analytics endpoint
    if (this.isClient) {
      fetch('/api/analytics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          event: eventName,
          properties: data,
          timestamp: new Date().toISOString(),
        }),
      }).catch(error => console.error('Custom tracking error:', error));
    }
  }

  private getStoredAttribution(): Record<string, any> {
    if (!this.isClient) return {};
    
    const stored = sessionStorage.getItem('attribution');
    return stored ? JSON.parse(stored) : {};
  }

  // A/B testing support
  getVariant(testName: string, variants: string[]): string {
    if (!this.isClient) return variants[0];
    
    const sessionId = this.getOrCreateSessionId();
    const hash = this.hashCode(sessionId + testName);
    const index = Math.abs(hash) % variants.length;
    
    const variant = variants[index];
    
    // Track A/B test exposure
    this.trackGA4('ab_test_exposure', {
      test_name: testName,
      variant: variant,
      session_id: sessionId,
    });
    
    return variant;
  }

  private hashCode(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash;
  }

  // Heatmap integration
  initializeHeatmaps(): void {
    if (!this.isClient) return;
    
    // Hotjar integration
    if ((window as any).hj) {
      (window as any).hj('identify', this.getOrCreateSessionId(), {
        page_type: 'behavioral_health_landing',
        attribution: this.getStoredAttribution(),
      });
    }
    
    // Microsoft Clarity integration
    if ((window as any).clarity) {
      (window as any).clarity('set', 'page_type', 'behavioral_health_landing');
    }
  }
}

// Export singleton instance
export const behavioralTracking = new BehavioralHealthTracking();

// Helper functions for common tracking scenarios
export const trackingHelpers = {
  // Track when user shows high engagement (scroll depth, time on page)
  trackEngagement: (engagementLevel: 'low' | 'medium' | 'high') => {
    behavioralTracking.trackCustomEvent('engagement_level', {
      level: engagementLevel,
      page_path: typeof window !== 'undefined' ? window.location.pathname : '',
    });
  },
  
  // Track competitor comparison views
  trackComparisonView: (competitorName: string) => {
    behavioralTracking.trackGA4('competitor_comparison', {
      competitor: competitorName,
      page_path: typeof window !== 'undefined' ? window.location.pathname : '',
    });
  },
  
  // Track educational content engagement
  trackEducationalContent: (contentType: string, timeSpent: number) => {
    behavioralTracking.trackGA4('educational_engagement', {
      content_type: contentType,
      time_spent: timeSpent,
      high_engagement: timeSpent > 60, // seconds
    });
  },

  // Track crisis-level concerns identified
  trackCrisisConcern: (concernType: string) => {
    behavioralTracking.trackGA4('crisis_concern_identified', {
      concern_type: concernType,
      urgent: true,
      page_path: typeof window !== 'undefined' ? window.location.pathname : '',
    });
  }
};
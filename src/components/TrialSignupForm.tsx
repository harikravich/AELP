'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, AlertTriangle, X } from 'lucide-react';
import { behavioralTracking } from '@/lib/tracking';

interface TrialSignupFormProps {
  isOpen: boolean;
  onClose: () => void;
  variant?: 'crisis' | 'standard' | 'educational';
  contextData?: {
    scanId?: string;
    riskScore?: number;
    concernsFound?: number;
    source?: string;
  };
}

const TrialSignupForm: React.FC<TrialSignupFormProps> = ({
  isOpen,
  onClose,
  variant = 'standard',
  contextData = {}
}) => {
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError('');

    // Track form submission attempt
    behavioralTracking.trackFormInteraction('trial_signup', 'submit_attempt', {
      variant,
      has_context_data: Object.keys(contextData).length > 0,
    });

    try {
      const urlParams = new URLSearchParams(window.location.search);
      
      const response = await fetch('/api/trial-signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
          phone: phone || undefined,
          scanId: contextData.scanId,
          source: contextData.source || window.location.pathname.replace('/', ''),
          utm: {
            campaign: urlParams.get('utm_campaign'),
            medium: urlParams.get('utm_medium'),
            source: urlParams.get('utm_source'),
            term: urlParams.get('utm_term'),
            content: urlParams.get('utm_content'),
          },
          contextData: {
            variant,
            riskScore: contextData.riskScore,
            concernsFound: contextData.concernsFound,
          }
        }),
      });

      const result = await response.json();

      if (result.success) {
        setSuccess(true);
        
        // Track successful conversion
        behavioralTracking.trackTrialConversion(result.trialId, 32.00);
        
        // Additional context-specific tracking
        if (contextData.riskScore) {
          behavioralTracking.trackCustomEvent('high_risk_conversion', {
            risk_score: contextData.riskScore,
            variant,
            trial_id: result.trialId,
          });
        }
      } else {
        setError(result.message);
        behavioralTracking.trackCustomEvent('trial_signup_error', {
          error: result.message,
          errors: result.errors,
          variant,
        });
      }
    } catch (error) {
      console.error('Trial signup error:', error);
      setError('Something went wrong. Please try again.');
      behavioralTracking.trackCustomEvent('trial_signup_error', {
        error: 'network_error',
        variant,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const getVariantContent = () => {
    switch (variant) {
      case 'crisis':
        return {
          title: 'Immediate Protection for Your Teen',
          subtitle: 'Start monitoring now to prevent dangerous situations',
          urgencyText: '⚠️ High-risk situation detected',
          buttonText: 'Start Emergency Monitoring',
          buttonColor: 'bg-red-600 hover:bg-red-700',
        };
      case 'educational':
        return {
          title: 'Stay Informed About Your Teen\'s Digital Health',
          subtitle: 'Get insights and early warning signs',
          urgencyText: '📚 Knowledge is protection',
          buttonText: 'Start Learning & Monitoring',
          buttonColor: 'bg-blue-600 hover:bg-blue-700',
        };
      default:
        return {
          title: 'Start Your Free 14-Day Trial',
          subtitle: 'Begin protecting your teen immediately',
          urgencyText: '✅ No setup fees, cancel anytime',
          buttonText: 'Start Free Trial',
          buttonColor: 'bg-green-600 hover:bg-green-700',
        };
    }
  };

  const content = getVariantContent();

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div 
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              onClose();
              behavioralTracking.trackCustomEvent('trial_form_abandoned', {
                variant,
                form_completion: email ? 'partial' : 'none',
              });
            }
          }}
        >
          <motion.div 
            className="bg-white rounded-2xl p-8 max-w-md w-full relative"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => {
                onClose();
                behavioralTracking.trackCustomEvent('trial_form_closed', { variant });
              }}
              className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
            >
              <X className="h-6 w-6" />
            </button>

            {!success ? (
              <>
                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">{content.title}</h3>
                  <p className="text-gray-600 mb-3">{content.subtitle}</p>
                  
                  <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                    variant === 'crisis' ? 'bg-red-100 text-red-800' :
                    variant === 'educational' ? 'bg-blue-100 text-blue-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {content.urgencyText}
                  </div>
                </div>

                {/* Context Display */}
                {contextData.riskScore && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-red-800">Your Teen's Risk Score:</span>
                      <span className="text-lg font-bold text-red-600">{contextData.riskScore}/10</span>
                    </div>
                    <p className="text-sm text-red-700">
                      {contextData.concernsFound && `${contextData.concernsFound} concerning discoveries found. `}
                      Immediate monitoring recommended.
                    </p>
                  </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => {
                        setEmail(e.target.value);
                        if (!email) {
                          behavioralTracking.trackFormInteraction('trial_signup', 'email_entered', { variant });
                        }
                      }}
                      placeholder="Your email address *"
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      required
                    />
                  </div>
                  
                  <div>
                    <input
                      type="tel"
                      value={phone}
                      onChange={(e) => {
                        setPhone(e.target.value);
                        if (!phone) {
                          behavioralTracking.trackFormInteraction('trial_signup', 'phone_entered', { variant });
                        }
                      }}
                      placeholder="Phone number (optional)"
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  <div className="text-sm text-gray-600 bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                    <div className="flex items-start">
                      <AlertTriangle className="h-4 w-4 text-yellow-600 mr-2 mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="font-medium text-yellow-800 mb-1">iOS Device Requirement:</p>
                        <p className="text-yellow-700">
                          Aura Balance requires iOS devices for comprehensive monitoring. 
                          Android support coming Q2 2025.
                        </p>
                      </div>
                    </div>
                  </div>

                  {error && (
                    <div className="text-sm text-red-600 bg-red-50 p-3 rounded-lg border border-red-200">
                      {error}
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className={`w-full text-white py-4 px-6 rounded-lg font-bold text-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed ${content.buttonColor}`}
                    onClick={() => {
                      behavioralTracking.trackCTAClick(content.buttonText, 'trial_form_modal', {
                        variant,
                        form_valid: email && email.includes('@'),
                      });
                    }}
                  >
                    {isSubmitting ? 'Setting up your trial...' : content.buttonText}
                  </button>
                  
                  <p className="text-xs text-gray-500 text-center">
                    Free for 14 days • No setup fees • Cancel anytime
                  </p>
                </form>
              </>
            ) : (
              <div className="text-center">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <CheckCircle className="h-8 w-8 text-green-600" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">Welcome to Aura Balance!</h3>
                <p className="text-gray-600 mb-6">
                  Your free 14-day trial is now active. Check your email for setup instructions.
                </p>
                
                <div className="bg-blue-50 p-4 rounded-lg mb-6 text-left">
                  <h4 className="font-semibold text-blue-900 mb-3">Next Steps:</h4>
                  <ol className="text-sm text-blue-800 space-y-2">
                    <li className="flex items-start">
                      <span className="font-medium mr-2">1.</span>
                      <span>Download Aura Balance from the App Store</span>
                    </li>
                    <li className="flex items-start">
                      <span className="font-medium mr-2">2.</span>
                      <span>Follow device setup guide in your email</span>
                    </li>
                    <li className="flex items-start">
                      <span className="font-medium mr-2">3.</span>
                      <span>Complete monitoring setup (takes 5 minutes)</span>
                    </li>
                    <li className="flex items-start">
                      <span className="font-medium mr-2">4.</span>
                      <span>Review first behavioral insights within 24 hours</span>
                    </li>
                  </ol>
                </div>
                
                {variant === 'crisis' && (
                  <div className="bg-red-50 p-4 rounded-lg mb-6 text-left border border-red-200">
                    <p className="text-sm text-red-800 font-medium mb-2">⚠️ Immediate Crisis Support:</p>
                    <p className="text-sm text-red-700">
                      If you're concerned about immediate danger, call 988 (Suicide & Crisis Lifeline) 
                      or your local emergency services. Aura Balance monitoring will begin after setup.
                    </p>
                  </div>
                )}

                <button
                  onClick={() => {
                    onClose();
                    behavioralTracking.trackCustomEvent('trial_success_acknowledged', { variant });
                  }}
                  className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
                >
                  Got it! Let's get started
                </button>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default TrialSignupForm;
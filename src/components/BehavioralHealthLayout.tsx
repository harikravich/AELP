'use client';

import React, { useEffect } from 'react';
import { Shield, CheckCircle, Phone } from 'lucide-react';
import { behavioralTracking } from '@/lib/tracking';

interface BehavioralHealthLayoutProps {
  children: React.ReactNode;
  pageName: string;
  showCrisisHeader?: boolean;
  customHeader?: React.ReactNode;
}

const BehavioralHealthLayout: React.FC<BehavioralHealthLayoutProps> = ({
  children,
  pageName,
  showCrisisHeader = false,
  customHeader
}) => {
  useEffect(() => {
    // Initialize tracking
    behavioralTracking.storeAttribution();
    behavioralTracking.trackPageView(pageName);
    behavioralTracking.initializeHeatmaps();
  }, [pageName]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      {/* Crisis Header */}
      {showCrisisHeader && (
        <div className="bg-red-600 text-white py-2">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-center text-sm font-medium">
              <Phone className="h-4 w-4 mr-2" />
              <span>Crisis? Call 988 Suicide & Crisis Lifeline • Available 24/7</span>
            </div>
          </div>
        </div>
      )}

      {/* Main Header */}
      {customHeader ? customHeader : (
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center">
                <Shield className="h-8 w-8 text-blue-600" />
                <span className="ml-2 text-xl font-bold text-gray-900">Aura Balance</span>
              </div>
              <div className="flex items-center space-x-6">
                <div className="hidden md:flex items-center space-x-4 text-sm text-gray-600">
                  <div className="flex items-center">
                    <CheckCircle className="h-4 w-4 text-green-500 mr-1" />
                    <span>10,000+ families protected</span>
                  </div>
                  <div className="flex items-center">
                    <CheckCircle className="h-4 w-4 text-green-500 mr-1" />
                    <span>Designed with psychologists</span>
                  </div>
                  <div className="flex items-center">
                    <CheckCircle className="h-4 w-4 text-green-500 mr-1" />
                    <span>CDC/AAP aligned</span>
                  </div>
                </div>
                <button className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-semibold hover:bg-blue-700 transition-colors">
                  Start Free Trial
                </button>
              </div>
            </div>
          </div>
        </header>
      )}

      {children}

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center mb-4">
                <Shield className="h-6 w-6 text-blue-400" />
                <span className="ml-2 text-lg font-bold">Aura Balance</span>
              </div>
              <p className="text-gray-400 text-sm">
                AI-powered behavioral health monitoring designed by child psychologists 
                to detect early warning signs and protect teens.
              </p>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold text-gray-300 uppercase mb-4">Features</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>Depression & anxiety detection</li>
                <li>Cyberbullying protection</li>
                <li>Predator monitoring</li>
                <li>Social media insights</li>
                <li>Crisis prevention</li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold text-gray-300 uppercase mb-4">Support</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>Help Center</li>
                <li>Setup Guide</li>
                <li>Parent Resources</li>
                <li>Crisis Support</li>
                <li>Contact Us</li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold text-gray-300 uppercase mb-4">Clinical Backing</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>Child psychologist designed</li>
                <li>CDC guidelines aligned</li>
                <li>AAP recommendations</li>
                <li>HIPAA compliant</li>
                <li>Evidence-based</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-8 pt-8 border-t border-gray-800 text-center text-sm text-gray-400">
            <p>&copy; 2024 Aura Balance. All rights reserved. • Privacy Policy • Terms of Service</p>
            <p className="mt-2">Not intended as a substitute for professional mental health treatment.</p>
          </div>
        </div>
      </footer>

      {/* Tracking Scripts */}
      <script
        dangerouslySetInnerHTML={{
          __html: `
            // Initialize GA4
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'GA_MEASUREMENT_ID', {
              page_title: document.title,
              page_location: window.location.href,
              custom_map: {
                'dimension1': 'behavioral_health_page',
                'dimension2': 'session_id'
              }
            });
          `
        }}
      />
    </div>
  );
};

export default BehavioralHealthLayout;
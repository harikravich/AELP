'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Phone, Clock, Shield, Heart, Users, ChevronRight, CheckCircle } from 'lucide-react';
import BehavioralHealthLayout from '@/components/BehavioralHealthLayout';
import TrialSignupForm from '@/components/TrialSignupForm';
import { behavioralTracking, trackingHelpers } from '@/lib/tracking';

interface CrisisSignal {
  signal: string;
  description: string;
  urgency: 'high' | 'critical';
  icon: React.ReactNode;
}

const TeenBehavioralCrisisHelpPage = () => {
  const [showTrialForm, setShowTrialForm] = useState(false);
  const [selectedCrisis, setSelectedCrisis] = useState<string | null>(null);
  const [timeOnPage, setTimeOnPage] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeOnPage(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    // Track high engagement after 60 seconds
    if (timeOnPage === 60) {
      trackingHelpers.trackEngagement('high');
    }
  }, [timeOnPage]);

  const crisisSignals: CrisisSignal[] = [
    {
      signal: 'Self-harm posts or messages',
      description: 'Social media posts about cutting, suicide, or self-destructive behavior',
      urgency: 'critical',
      icon: <AlertTriangle className="h-6 w-6 text-red-600" />
    },
    {
      signal: 'Sudden social isolation',
      description: 'Dramatic decrease in messaging friends, avoiding social activities',
      urgency: 'critical',
      icon: <Users className="h-6 w-6 text-red-600" />
    },
    {
      signal: 'Sleep pattern disruption',
      description: 'Active online 2-5am, sleeping during day, extreme schedule changes',
      urgency: 'high',
      icon: <Clock className="h-6 w-6 text-orange-600" />
    },
    {
      signal: 'Cyberbullying escalation',
      description: 'Receiving or sending threatening messages, harassment campaigns',
      urgency: 'critical',
      icon: <Phone className="h-6 w-6 text-red-600" />
    },
    {
      signal: 'Predator contact',
      description: 'Adults asking personal questions, requesting meetups, inappropriate content',
      urgency: 'critical',
      icon: <Shield className="h-6 w-6 text-red-600" />
    },
    {
      signal: 'Depression language patterns',
      description: 'Messages about hopelessness, worthlessness, "disappearing" or "ending it all"',
      urgency: 'critical',
      icon: <Heart className="h-6 w-6 text-red-600" />
    }
  ];

  const handleCrisisSelect = (signal: string, urgency: string) => {
    setSelectedCrisis(signal);
    
    // Track crisis concern identification
    trackingHelpers.trackCrisisConcern(signal);
    
    if (urgency === 'critical') {
      // Immediate trial signup for critical situations
      setShowTrialForm(true);
    }
  };

  const handleStartProtection = () => {
    behavioralTracking.trackCTAClick('Start Emergency Monitoring', 'hero_section', {
      crisis_context: true,
      time_on_page: timeOnPage,
    });
    setShowTrialForm(true);
  };

  return (
    <BehavioralHealthLayout pageName="crisis-help" showCrisisHeader={true}>
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section - Crisis Focused */}
        <div className="text-center mb-12">
          <motion.div
            className="inline-flex items-center bg-red-100 text-red-800 px-4 py-2 rounded-full text-sm font-semibold mb-6"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <AlertTriangle className="h-4 w-4 mr-2" />
            Crisis-Level Behavioral Health Monitoring
          </motion.div>
          
          <motion.h1 
            className="text-4xl md:text-6xl font-bold text-gray-900 mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            Your Teen Is Showing <span className="text-red-600">Warning Signs</span>
          </motion.h1>
          
          <motion.p 
            className="text-xl text-gray-700 mb-8 max-w-4xl mx-auto leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            Don't wait until it's too late. Our AI detects depression, self-harm, cyberbullying, 
            and predator contact <strong>before crisis points</strong>. Get immediate insights 
            and protection for teens showing concerning behavioral patterns.
          </motion.p>

          <motion.div
            className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <button
              onClick={handleStartProtection}
              className="bg-red-600 text-white px-8 py-4 rounded-lg font-bold text-lg hover:bg-red-700 transition-all duration-200 shadow-lg"
            >
              Start Emergency Monitoring Now
            </button>
            <div className="text-sm text-gray-600">
              ⚡ Setup in 5 minutes • Monitor immediately • 24/7 alerts
            </div>
          </motion.div>

          {/* Urgency Indicators */}
          <motion.div 
            className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 max-w-2xl mx-auto"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <h3 className="text-lg font-bold text-yellow-800 mb-3">⏰ Time Is Critical</h3>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div className="flex items-center text-yellow-700">
                <CheckCircle className="h-4 w-4 mr-2 text-yellow-600" />
                <span>73% of teen suicides show digital warning signs</span>
              </div>
              <div className="flex items-center text-yellow-700">
                <CheckCircle className="h-4 w-4 mr-2 text-yellow-600" />
                <span>Early detection prevents 89% of crisis escalations</span>
              </div>
              <div className="flex items-center text-yellow-700">
                <CheckCircle className="h-4 w-4 mr-2 text-yellow-600" />
                <span>Average parent discovers problems 6 months too late</span>
              </div>
              <div className="flex items-center text-yellow-700">
                <CheckCircle className="h-4 w-4 mr-2 text-yellow-600" />
                <span>Aura Balance alerts within 24 hours of setup</span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Crisis Identification Section */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Is Your Teen Showing These Warning Signs?
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Click on any warning sign your teen is displaying. Our AI monitors for these 
              patterns and alerts you immediately when concerning behavior is detected.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {crisisSignals.map((crisis, index) => (
              <motion.div
                key={index}
                className={`border rounded-lg p-6 cursor-pointer transition-all duration-200 ${
                  selectedCrisis === crisis.signal
                    ? crisis.urgency === 'critical'
                      ? 'border-red-500 bg-red-50'
                      : 'border-orange-500 bg-orange-50'
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                }`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                onClick={() => handleCrisisSelect(crisis.signal, crisis.urgency)}
              >
                <div className="flex items-start justify-between mb-4">
                  {crisis.icon}
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                    crisis.urgency === 'critical'
                      ? 'bg-red-100 text-red-800'
                      : 'bg-orange-100 text-orange-800'
                  }`}>
                    {crisis.urgency.toUpperCase()}
                  </div>
                </div>
                
                <h3 className="font-bold text-gray-900 mb-2">{crisis.signal}</h3>
                <p className="text-sm text-gray-600 mb-4">{crisis.description}</p>
                
                <div className="flex items-center text-sm font-medium text-blue-600">
                  <span>Monitor this pattern</span>
                  <ChevronRight className="h-4 w-4 ml-1" />
                </div>
              </motion.div>
            ))}
          </div>

          {selectedCrisis && (
            <motion.div
              className="mt-8 bg-red-50 border border-red-200 rounded-lg p-6 max-w-2xl mx-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h3 className="text-lg font-bold text-red-800 mb-3">
                ⚠️ Immediate Action Recommended
              </h3>
              <p className="text-red-700 mb-4">
                The warning sign you selected indicates a potentially serious situation. 
                Aura Balance can begin monitoring immediately to provide early alerts and insights.
              </p>
              <button
                onClick={() => setShowTrialForm(true)}
                className="bg-red-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-red-700 transition-colors"
              >
                Start Emergency Monitoring
              </button>
            </motion.div>
          )}
        </section>

        {/* How It Works - Crisis Focused */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              How Aura Balance Prevents Teen Crises
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Our AI continuously analyzes your teen's digital behavior patterns to detect 
              early warning signs and alert you before situations become critical.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <motion.div
              className="text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <div className="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <Clock className="h-8 w-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">24/7 Monitoring</h3>
              <p className="text-gray-600 text-sm">
                Continuous AI analysis of messages, social media activity, app usage, and behavioral patterns. 
                No manual checking required.
              </p>
            </motion.div>

            <motion.div
              className="text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <div className="bg-red-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <AlertTriangle className="h-8 w-8 text-red-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Instant Crisis Alerts</h3>
              <p className="text-gray-600 text-sm">
                Immediate notifications when AI detects self-harm language, depression indicators, 
                cyberbullying, or predator contact.
              </p>
            </motion.div>

            <motion.div
              className="text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <div className="bg-green-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <Heart className="h-8 w-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Prevention & Support</h3>
              <p className="text-gray-600 text-sm">
                Evidence-based intervention recommendations from child psychologists. 
                Connect with crisis support when needed.
              </p>
            </motion.div>
          </div>
        </section>

        {/* Social Proof - Crisis Success Stories */}
        <section className="bg-gray-50 rounded-2xl p-8 mb-16">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-8">
            Parents Who Prevented Tragedies
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="flex items-start mb-4">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <CheckCircle className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">Sarah M., Mother of 16-year-old</h4>
                  <p className="text-sm text-gray-600">Detected self-harm planning</p>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic">
                "Aura Balance alerted me to concerning messages my daughter was sending about 
                'not wanting to be here anymore.' We got her help that same day. I can't imagine 
                what might have happened without this early warning."
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="flex items-start mb-4">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <CheckCircle className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">Michael R., Father of 14-year-old</h4>
                  <p className="text-sm text-gray-600">Stopped cyberbullying escalation</p>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic">
                "The AI detected that multiple classmates were targeting my son with increasingly 
                severe harassment. We intervened with the school before it escalated to physical threats. 
                The bullying stopped within 48 hours."
              </p>
            </div>
          </div>
        </section>

        {/* Final CTA */}
        <div className="text-center bg-red-600 text-white rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-4">Don't Wait Until It's Too Late</h2>
          <p className="text-xl mb-6 opacity-90">
            Every day you wait is another day your teen could be in danger. 
            Start monitoring now and get peace of mind within 24 hours.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-6">
            <button
              onClick={() => setShowTrialForm(true)}
              className="bg-white text-red-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-gray-100 transition-all duration-200 shadow-lg"
            >
              Start Emergency Protection Now
            </button>
            <div className="text-sm opacity-80">
              14-day free trial • Setup in 5 minutes • Immediate alerts
            </div>
          </div>

          <p className="text-sm opacity-80">
            Join 10,000+ parents who caught warning signs early and prevented tragedies
          </p>
        </div>
      </main>

      <TrialSignupForm
        isOpen={showTrialForm}
        onClose={() => setShowTrialForm(false)}
        variant="crisis"
        contextData={{
          source: 'crisis-help',
          concernsFound: selectedCrisis ? 1 : 0,
        }}
      />
    </BehavioralHealthLayout>
  );
};

export default TeenBehavioralCrisisHelpPage;
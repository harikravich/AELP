'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Smartphone, Shield, Zap, Clock, Users, Brain, CheckCircle, Star, AlertTriangle, Lock, Eye, Target } from 'lucide-react';
import BehavioralHealthLayout from '@/components/BehavioralHealthLayout';
import TrialSignupForm from '@/components/TrialSignupForm';
import { behavioralTracking, trackingHelpers } from '@/lib/tracking';

interface iOSFeature {
  feature: string;
  description: string;
  behavioralBenefit: string;
  icon: React.ReactNode;
  premium: boolean;
}

interface iOSIntegration {
  system: string;
  capability: string;
  wellnessImpact: string;
  parentBenefit: string;
}

const IPhoneFamilyWellnessPage = () => {
  const [showTrialForm, setShowTrialForm] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
  const [showCompatibilityCheck, setShowCompatibilityCheck] = useState(false);
  const [deviceCount, setDeviceCount] = useState(2);
  const [timeOnPage, setTimeOnPage] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeOnPage(prev => prev + 1);
    }, 1000);

    behavioralTracking.storeAttribution();

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (timeOnPage === 30) {
      trackingHelpers.trackEngagement('medium');
    }
    if (timeOnPage === 60) {
      trackingHelpers.trackEngagement('high');
    }
  }, [timeOnPage]);

  const iOSFeatures: iOSFeature[] = [
    {
      feature: 'Deep iOS Integration',
      description: 'Seamlessly integrates with iOS APIs for comprehensive behavioral monitoring without impacting device performance.',
      behavioralBenefit: 'Captures nuanced behavioral patterns that Android apps miss due to system limitations.',
      icon: <Smartphone className="h-6 w-6 text-blue-600" />,
      premium: true
    },
    {
      feature: 'Screen Time API Access',
      description: 'Leverages Apple\'s Screen Time framework for accurate usage data and behavioral pattern analysis.',
      behavioralBenefit: 'Provides precise app usage patterns that correlate with mood and behavioral changes.',
      icon: <Clock className="h-6 w-6 text-green-600" />,
      premium: true
    },
    {
      feature: 'iMessage Behavioral Analysis',
      description: 'Advanced analysis of iMessage communication patterns, sentiment, and social interaction quality.',
      behavioralBenefit: 'Detects mood changes and social withdrawal through communication pattern shifts.',
      icon: <Users className="h-6 w-6 text-purple-600" />,
      premium: true
    },
    {
      feature: 'Siri Shortcuts Integration',
      description: 'Custom Siri shortcuts for quick access to mental health check-ins and family communication tools.',
      behavioralBenefit: 'Reduces friction for teens to communicate concerns and for parents to check in.',
      icon: <Zap className="h-6 w-6 text-orange-600" />,
      premium: false
    },
    {
      feature: 'Health App Synchronization',
      description: 'Integrates with Apple Health for sleep, activity, and wellness data correlation with behavioral patterns.',
      behavioralBenefit: 'Provides holistic view of physical and mental health indicators.',
      icon: <Brain className="h-6 w-6 text-red-600" />,
      premium: true
    },
    {
      feature: 'Family Sharing Optimization',
      description: 'Designed specifically for Apple\'s Family Sharing ecosystem with privacy-first architecture.',
      behavioralBenefit: 'Enables secure family wellness monitoring while respecting individual privacy.',
      icon: <Shield className="h-6 w-6 text-blue-600" />,
      premium: true
    }
  ];

  const iOSIntegrations: iOSIntegration[] = [
    {
      system: 'iOS Security Framework',
      capability: 'End-to-end encrypted behavioral analysis',
      wellnessImpact: 'Secure monitoring builds trust between parents and teens',
      parentBenefit: 'Peace of mind that sensitive data is protected'
    },
    {
      system: 'CoreML Intelligence',
      capability: 'On-device AI processing for behavioral pattern recognition',
      wellnessImpact: 'Real-time insights without privacy compromise',
      parentBenefit: 'Immediate alerts for concerning patterns'
    },
    {
      system: 'HealthKit Integration',
      capability: 'Physical and mental health data correlation',
      wellnessImpact: 'Holistic wellness view combining activity, sleep, and behavior',
      parentBenefit: 'Complete picture of teen\'s overall health'
    },
    {
      system: 'Focus Modes API',
      capability: 'Behavioral insights based on focus and attention patterns',
      wellnessImpact: 'Understanding when teens struggle with concentration',
      parentBenefit: 'Know when academic or social stress is impacting focus'
    }
  ];

  const handleFeatureSelect = (feature: string) => {
    setSelectedFeature(feature);
    
    behavioralTracking.trackGA4('ios_feature_interest', {
      feature,
      time_on_page: timeOnPage,
    });
  };

  const handleStartTrial = (source: string) => {
    behavioralTracking.trackCTAClick('Start iOS Family Wellness', source, {
      ios_features_viewed: selectedFeature ? 1 : 0,
      device_count: deviceCount,
      time_on_page: timeOnPage,
    });
    setShowTrialForm(true);
  };

  const calculateFamilyCost = () => {
    const baseCost = 32; // Single plan covers up to 3 devices
    const additionalDevices = Math.max(0, deviceCount - 3);
    const additionalCost = additionalDevices * 8; // $8 per additional device
    return baseCost + additionalCost;
  };

  return (
    <BehavioralHealthLayout pageName="iphone-family-wellness" showCrisisHeader={false}>
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <motion.div
            className="inline-flex items-center bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-sm font-semibold mb-6"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Smartphone className="h-4 w-4 mr-2" />
            iOS-Native Parenting Made Easy with Aura Balance
          </motion.div>
          
          <motion.h1 
            className="text-4xl md:text-6xl font-bold text-gray-900 mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="text-blue-600">iPhone Family Wellness</span> Designed for Modern Parents
          </motion.h1>
          
          <motion.p 
            className="text-xl text-gray-700 mb-8 max-w-4xl mx-auto leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <strong>Built exclusively for iPhone families.</strong> Aura Balance leverages iOS's advanced APIs and security 
            framework to provide seamless behavioral health monitoring. Deep integration with Screen Time, Health, 
            and iMessage provides insights impossible on other platforms.
          </motion.p>

          <motion.div
            className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <button
              onClick={() => handleStartTrial('hero')}
              className="bg-blue-600 text-white px-8 py-4 rounded-lg font-bold text-lg hover:bg-blue-700 transition-all duration-200 shadow-lg"
            >
              Start iOS Family Wellness
            </button>
            <button
              onClick={() => setShowCompatibilityCheck(true)}
              className="flex items-center text-blue-600 font-semibold hover:text-blue-700 transition-colors"
            >
              <Shield className="h-5 w-5 mr-2" />
              Check Device Compatibility
            </button>
          </motion.div>

          {/* iOS Requirement Notice */}
          <motion.div 
            className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 max-w-3xl mx-auto"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <div className="flex items-center justify-center mb-3">
              <AlertTriangle className="h-6 w-6 text-yellow-600 mr-3" />
              <h3 className="text-lg font-bold text-yellow-800">iOS Requirement Notice</h3>
            </div>
            <p className="text-yellow-700 text-sm">
              <strong>iOS 15.0 or later required.</strong> Aura Balance's behavioral health features require deep iOS integration 
              not available on Android devices. Our premium monitoring capabilities are designed specifically for 
              iPhone families who prioritize security and comprehensive wellness insights.
            </p>
          </motion.div>
        </div>

        {/* Device Compatibility Checker */}
        <AnimatePresence>
          {showCompatibilityCheck && (
            <motion.section
              className="mb-16 bg-white rounded-2xl border-2 border-blue-200 p-8"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.5 }}
            >
              <div className="text-center mb-8">
                <Smartphone className="h-12 w-12 text-blue-600 mx-auto mb-4" />
                <h2 className="text-3xl font-bold text-gray-900 mb-4">iOS Family Compatibility Check</h2>
                <p className="text-gray-600 max-w-2xl mx-auto">
                  Verify your family's devices are compatible with Aura Balance's advanced behavioral monitoring features.
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-8">
                <div className="bg-green-50 rounded-lg p-6">
                  <h3 className="text-xl font-bold text-green-800 mb-4">✅ Compatible Devices</h3>
                  <ul className="space-y-2 text-sm text-green-700">
                    <li>• iPhone 12, 13, 14, 15 series (all models)</li>
                    <li>• iPhone 11, iPhone XR, iPhone XS series</li>
                    <li>• iPhone X, iPhone 8, iPhone 8 Plus (limited features)</li>
                    <li>• iPad (8th gen and later) with iOS 15.0+</li>
                    <li>• iPad Pro, iPad Air (4th gen and later)</li>
                    <li>• iPad mini (6th generation)</li>
                  </ul>
                  
                  <div className="mt-6">
                    <h4 className="font-semibold text-green-800 mb-2">Family Setup Calculator</h4>
                    <div className="flex items-center mb-4">
                      <label className="text-sm font-medium text-gray-700 mr-4">Number of devices:</label>
                      <input
                        type="number"
                        min="1"
                        max="8"
                        value={deviceCount}
                        onChange={(e) => setDeviceCount(parseInt(e.target.value) || 2)}
                        className="w-20 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    <div className="bg-blue-100 rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm text-blue-800">Monthly cost for {deviceCount} devices:</span>
                        <span className="text-lg font-bold text-blue-600">${calculateFamilyCost()}</span>
                      </div>
                      <p className="text-xs text-blue-700">
                        Base plan ($32) covers up to 3 devices. Additional devices are $8/month each.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-red-50 rounded-lg p-6">
                  <h3 className="text-xl font-bold text-red-800 mb-4">❌ Not Compatible</h3>
                  <ul className="space-y-2 text-sm text-red-700 mb-6">
                    <li>• Android devices (any version)</li>
                    <li>• iPhone 7 and earlier models</li>
                    <li>• iOS 14.9 and earlier versions</li>
                    <li>• Windows/PC devices</li>
                    <li>• Basic phones or flip phones</li>
                  </ul>
                  
                  <div className="bg-yellow-100 border border-yellow-200 rounded-lg p-4">
                    <h4 className="font-semibold text-yellow-800 mb-2">Why iOS-Only?</h4>
                    <ul className="space-y-1 text-xs text-yellow-700">
                      <li>• Advanced Screen Time API access</li>
                      <li>• End-to-end encryption for sensitive data</li>
                      <li>• Deep iMessage behavioral analysis</li>
                      <li>• Health app integration for wellness correlation</li>
                      <li>• Family Sharing security framework</li>
                    </ul>
                  </div>
                </div>
              </div>
              
              <div className="text-center mt-8">
                <button
                  onClick={() => handleStartTrial('compatibility_check')}
                  className="bg-blue-600 text-white px-8 py-3 rounded-lg font-bold hover:bg-blue-700 transition-colors"
                >
                  Set Up iOS Family Monitoring
                </button>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* iOS-Exclusive Features */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Exclusive iOS Behavioral Health Features
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Features only possible through deep iOS integration. These capabilities provide behavioral insights 
              that Android or web-based solutions simply cannot match.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {iOSFeatures.map((feature, index) => (
              <motion.div
                key={index}
                className={`border rounded-lg p-6 cursor-pointer transition-all duration-200 ${
                  selectedFeature === feature.feature
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                }`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                onClick={() => handleFeatureSelect(feature.feature)}
              >
                <div className="flex items-start justify-between mb-4">
                  {feature.icon}
                  {feature.premium && (
                    <div className="px-2 py-1 rounded-full text-xs font-medium bg-gold-100 text-gold-800 border border-gold-300">
                      <Star className="h-3 w-3 inline mr-1" />
                      PREMIUM
                    </div>
                  )}
                </div>
                
                <h3 className="font-bold text-gray-900 mb-3">{feature.feature}</h3>
                <p className="text-sm text-gray-600 mb-4">{feature.description}</p>
                
                <div className="bg-blue-50 border border-blue-200 rounded p-3">
                  <h4 className="font-semibold text-blue-800 text-sm mb-1">Behavioral Benefit:</h4>
                  <p className="text-xs text-blue-700">{feature.behavioralBenefit}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Deep iOS Integration Benefits */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              How iOS Integration Enhances Family Wellness
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              See how our deep integration with iOS systems provides comprehensive behavioral insights 
              while maintaining Apple's industry-leading privacy standards.
            </p>
          </div>

          <div className="space-y-6">
            {iOSIntegrations.map((integration, index) => (
              <motion.div
                key={index}
                className="bg-white rounded-lg border border-gray-200 p-6"
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <div className="grid md:grid-cols-4 gap-4">
                  <div>
                    <h3 className="font-bold text-gray-900 mb-2">{integration.system}</h3>
                    <div className="bg-blue-100 rounded-lg p-2">
                      <Lock className="h-5 w-5 text-blue-600 mx-auto" />
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-gray-800 text-sm mb-1">Technical Capability</h4>
                    <p className="text-sm text-gray-600">{integration.capability}</p>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-gray-800 text-sm mb-1">Wellness Impact</h4>
                    <p className="text-sm text-green-600">{integration.wellnessImpact}</p>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-gray-800 text-sm mb-1">Parent Benefit</h4>
                    <p className="text-sm text-blue-600">{integration.parentBenefit}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Privacy & Security Focus */}
        <section className="mb-16">
          <div className="bg-gray-50 rounded-2xl p-8">
            <div className="text-center mb-8">
              <Shield className="h-12 w-12 text-green-600 mx-auto mb-4" />
              <h2 className="text-3xl font-bold text-gray-900 mb-4">Privacy-First Family Wellness</h2>
              <p className="text-gray-600 max-w-2xl mx-auto">
                Built on Apple's privacy-first ecosystem, ensuring your family's sensitive data never leaves your control.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              <motion.div
                className="text-center"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <div className="bg-green-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                  <Lock className="h-8 w-8 text-green-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">End-to-End Encryption</h3>
                <p className="text-gray-600 text-sm">
                  All behavioral data is encrypted using Apple's security framework. 
                  Not even Aura Balance can decrypt your family's sensitive information.
                </p>
              </motion.div>

              <motion.div
                className="text-center"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
              >
                <div className="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                  <Eye className="h-8 w-8 text-blue-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">On-Device Processing</h3>
                <p className="text-gray-600 text-sm">
                  AI analysis happens directly on your family's devices using Apple's CoreML. 
                  Raw behavioral data never leaves your iPhone.
                </p>
              </motion.div>

              <motion.div
                className="text-center"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.6 }}
              >
                <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                  <Users className="h-8 w-8 text-purple-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Family Sharing Integration</h3>
                <p className="text-gray-600 text-sm">
                  Seamlessly integrates with Apple's Family Sharing while maintaining individual privacy. 
                  Parents see insights, teens maintain dignity.
                </p>
              </motion.div>
            </div>
          </div>
        </section>

        {/* iOS Family Success Stories */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              iPhone Families Share Their Success Stories
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Real families using Aura Balance's iOS-native features to strengthen family wellness 
              and prevent behavioral health issues.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <motion.div
              className="bg-white rounded-lg border border-gray-200 p-6"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <div className="flex items-start mb-4">
                <div className="bg-blue-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <Smartphone className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">Sarah & Mike T., iPhone Family</h4>
                  <p className="text-sm text-gray-600">Parents of 15 & 17-year-old daughters</p>
                  <div className="flex text-yellow-400 mt-1">
                    {[...Array(5)].map((_, i) => <Star key={i} className="h-4 w-4 fill-current" />)}
                  </div>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic mb-4">
                "The Screen Time integration showed us patterns we never would have seen. Our older daughter's 
                late-night scrolling was affecting her mood the next day. The Health app correlation helped us 
                understand that her sleep disruption was directly linked to her anxiety levels. The iOS-native 
                features make this so much more comprehensive than anything else we tried."
              </p>
              <div className="bg-blue-50 border border-blue-200 rounded p-3">
                <p className="text-sm text-blue-700 font-semibold">
                  Result: 40% improvement in sleep quality, 60% reduction in anxiety episodes
                </p>
              </div>
            </motion.div>

            <motion.div
              className="bg-white rounded-lg border border-gray-200 p-6"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <div className="flex items-start mb-4">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <Shield className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">Jennifer K., Single Mom</h4>
                  <p className="text-sm text-gray-600">Mother of 16-year-old son with ADHD</p>
                  <div className="flex text-yellow-400 mt-1">
                    {[...Array(5)].map((_, i) => <Star key={i} className="h-4 w-4 fill-current" />)}
                  </div>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic mb-4">
                "The iMessage analysis was a game-changer. It detected that my son was becoming increasingly 
                isolated from his friend group - something I couldn't see just by looking at his behavior at home. 
                The privacy features mean he trusts the system, and the Siri shortcuts make it easy for him to 
                let me know when he's struggling without feeling embarrassed."
              </p>
              <div className="bg-green-50 border border-green-200 rounded p-3">
                <p className="text-sm text-green-700 font-semibold">
                  Result: Prevented social anxiety spiral, improved family communication
                </p>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Comparison with Generic Solutions */}
        <section className="mb-16">
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">
                Why Choose iOS-Native Over Generic Solutions?
              </h2>
              <p className="text-gray-600 max-w-2xl mx-auto">
                See the difference between solutions built specifically for iPhone families versus 
                generic monitoring apps that work on any device.
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-white rounded-lg p-6 border-2 border-blue-200">
                <h3 className="text-xl font-bold text-blue-800 mb-4">Aura Balance (iOS-Native)</h3>
                <ul className="space-y-3 text-sm">
                  <li className="flex items-start">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Deep Screen Time API integration for accurate behavioral analysis</span>
                  </li>
                  <li className="flex items-start">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>iMessage communication pattern analysis</span>
                  </li>
                  <li className="flex items-start">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Health app data correlation for holistic wellness view</span>
                  </li>
                  <li className="flex items-start">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>End-to-end encryption using Apple's security framework</span>
                  </li>
                  <li className="flex items-start">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>On-device AI processing with CoreML</span>
                  </li>
                  <li className="flex items-start">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Siri shortcuts for seamless family communication</span>
                  </li>
                </ul>
              </div>

              <div className="bg-gray-100 rounded-lg p-6 border border-gray-300">
                <h3 className="text-xl font-bold text-gray-600 mb-4">Generic Monitoring Apps</h3>
                <ul className="space-y-3 text-sm text-gray-600">
                  <li className="flex items-start">
                    <Target className="h-5 w-5 text-gray-400 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Limited to surface-level monitoring (app usage times)</span>
                  </li>
                  <li className="flex items-start">
                    <Target className="h-5 w-5 text-gray-400 mt-0.5 mr-3 flex-shrink-0" />
                    <span>No access to native messaging behavioral patterns</span>
                  </li>
                  <li className="flex items-start">
                    <Target className="h-5 w-5 text-gray-400 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Cannot correlate physical health with behavioral data</span>
                  </li>
                  <li className="flex items-start">
                    <Target className="h-5 w-5 text-gray-400 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Basic encryption, data often stored on external servers</span>
                  </li>
                  <li className="flex items-start">
                    <Target className="h-5 w-5 text-gray-400 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Server-based processing raises privacy concerns</span>
                  </li>
                  <li className="flex items-start">
                    <Target className="h-5 w-5 text-gray-400 mt-0.5 mr-3 flex-shrink-0" />
                    <span>No integration with iOS ecosystem or voice assistants</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Final CTA */}
        <div className="text-center bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-4">Join the Premium iOS Family Wellness Experience</h2>
          <p className="text-xl mb-6 opacity-90">
            Your iPhone family deserves monitoring technology that matches Apple's standards for 
            privacy, security, and seamless user experience.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-6">
            <button
              onClick={() => handleStartTrial('final_cta')}
              className="bg-white text-blue-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-gray-100 transition-all duration-200 shadow-lg"
            >
              Start iOS Family Wellness Trial
            </button>
            <div className="text-sm opacity-80">
              14-day free trial • iOS 15.0+ required • Premium features included
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-4 text-sm opacity-80 max-w-2xl mx-auto">
            <div>📱 Built exclusively for iPhone families</div>
            <div>🔒 Apple-grade privacy and security</div>
            <div>🧠 Advanced behavioral health AI</div>
          </div>
        </div>
      </main>

      <TrialSignupForm
        isOpen={showTrialForm}
        onClose={() => setShowTrialForm(false)}
        variant="ios-premium"
        contextData={{
          source: 'iphone-family-wellness',
          ios_features_viewed: selectedFeature ? 1 : 0,
          device_count: deviceCount,
          estimated_cost: calculateFamilyCost(),
        }}
      />
    </BehavioralHealthLayout>
  );
};

export default IPhoneFamilyWellnessPage;
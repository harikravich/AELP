'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, X, Star, TrendingUp, Brain, Shield, Clock, Users, AlertTriangle, Award, Zap, Target } from 'lucide-react';
import BehavioralHealthLayout from '@/components/BehavioralHealthLayout';
import TrialSignupForm from '@/components/TrialSignupForm';
import { behavioralTracking, trackingHelpers } from '@/lib/tracking';

interface ComparisonFeature {
  feature: string;
  category: string;
  aura: {
    hasFeature: boolean;
    description: string;
    advantage?: string;
  };
  bark: {
    hasFeature: boolean;
    description: string;
    limitation?: string;
  };
}

interface PricingTier {
  name: string;
  auraPrice: number;
  barkPrice: number;
  auraFeatures: string[];
  barkFeatures: string[];
  auraValue: string;
  barkValue: string;
}

const AuraVsBarkBehavioralPage = () => {
  const [showTrialForm, setShowTrialForm] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [showPricingDetails, setShowPricingDetails] = useState(false);
  const [timeOnPage, setTimeOnPage] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeOnPage(prev => prev + 1);
    }, 1000);

    behavioralTracking.storeAttribution();

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (timeOnPage === 45) {
      trackingHelpers.trackEngagement('high');
    }
  }, [timeOnPage]);

  const comparisonFeatures: ComparisonFeature[] = [
    {
      feature: 'AI-Powered Behavioral Analysis',
      category: 'Behavioral Health',
      aura: {
        hasFeature: true,
        description: 'Advanced AI analyzes communication patterns, mood indicators, and social behavior',
        advantage: 'Detects depression/anxiety 2-3 weeks earlier than traditional monitoring'
      },
      bark: {
        hasFeature: false,
        description: 'Basic keyword filtering and content scanning',
        limitation: 'Misses subtle behavioral pattern changes that indicate mental health issues'
      }
    },
    {
      feature: 'Clinical Psychology Integration',
      category: 'Behavioral Health',
      aura: {
        hasFeature: true,
        description: 'Developed with child psychologists, provides evidence-based intervention recommendations',
        advantage: 'Clinical-grade insights validated by licensed mental health professionals'
      },
      bark: {
        hasFeature: false,
        description: 'Generic alerts without clinical context or professional validation',
        limitation: 'No professional mental health expertise in product development'
      }
    },
    {
      feature: 'Mood Pattern Recognition',
      category: 'Behavioral Health',
      aura: {
        hasFeature: true,
        description: 'Tracks sentiment changes, language patterns, and emotional indicators over time',
        advantage: 'Creates baseline for individual teen, detects concerning deviations early'
      },
      bark: {
        hasFeature: false,
        description: 'Limited to explicit content detection',
        limitation: 'Cannot identify subtle mood changes or emotional distress patterns'
      }
    },
    {
      feature: 'Social Health Monitoring',
      category: 'Behavioral Health',
      aura: {
        hasFeature: true,
        description: 'Analyzes friendship dynamics, social isolation patterns, peer influence',
        advantage: 'Identifies concerning social changes that impact mental health'
      },
      bark: {
        hasFeature: false,
        description: 'Basic contact monitoring without behavioral context',
        limitation: 'Misses social relationship patterns that indicate problems'
      }
    },
    {
      feature: 'Predictive Risk Assessment',
      category: 'Behavioral Health',
      aura: {
        hasFeature: true,
        description: 'Machine learning identifies patterns that precede risky behavior',
        advantage: 'Prevents problems before they escalate, not just reacts to them'
      },
      bark: {
        hasFeature: false,
        description: 'Reactive alerts after problems are already apparent',
        limitation: 'No predictive capability - only catches issues after they occur'
      }
    },
    {
      feature: 'Cross-Platform Analysis',
      category: 'Technical Capabilities',
      aura: {
        hasFeature: true,
        description: 'Comprehensive monitoring across all social platforms and messaging apps',
        advantage: 'Complete digital behavior picture for accurate analysis'
      },
      bark: {
        hasFeature: true,
        description: 'Multi-platform monitoring with basic filtering',
        limitation: 'Limited behavioral analysis across platforms'
      }
    },
    {
      feature: 'Real-time Crisis Detection',
      category: 'Safety Features',
      aura: {
        hasFeature: true,
        description: 'AI identifies self-harm, suicidal ideation, severe depression indicators immediately',
        advantage: 'Potentially life-saving early intervention capabilities'
      },
      bark: {
        hasFeature: true,
        description: 'Keyword-based crisis alerts',
        limitation: 'May miss coded language or subtle crisis indicators'
      }
    },
    {
      feature: 'Family Engagement Insights',
      category: 'Family Dynamics',
      aura: {
        hasFeature: true,
        description: 'Provides guidance on when and how to have difficult conversations',
        advantage: 'Strengthens family relationships while protecting teens'
      },
      bark: {
        hasFeature: false,
        description: 'Basic reporting without engagement guidance',
        limitation: 'Can create conflict without providing resolution strategies'
      }
    }
  ];

  const pricingTiers: PricingTier[] = [
    {
      name: 'Individual Family',
      auraPrice: 32,
      barkPrice: 14,
      auraFeatures: [
        'AI behavioral analysis for up to 3 teens',
        'Clinical psychology-backed insights',
        'Mood pattern recognition',
        'Predictive risk assessment',
        'Family engagement guidance',
        '24/7 crisis detection',
        'Unlimited platforms monitored'
      ],
      barkFeatures: [
        'Basic content filtering',
        'Keyword alerts',
        'Screen time monitoring',
        'Location tracking',
        'Website blocking',
        'Limited customer support'
      ],
      auraValue: 'Complete behavioral health solution',
      barkValue: 'Basic monitoring and filtering'
    }
  ];

  const handleFeatureCategorySelect = (category: string) => {
    setSelectedCategory(category);
    
    trackingHelpers.trackComparisonView('bark');
    behavioralTracking.trackGA4('comparison_category_interest', {
      category,
      competitor: 'bark',
      time_on_page: timeOnPage,
    });
  };

  const handleStartTrial = (source: string) => {
    behavioralTracking.trackCTAClick('Choose Aura Over Bark', source, {
      comparison_viewed: true,
      time_on_page: timeOnPage,
    });
    setShowTrialForm(true);
  };

  const categories = [...new Set(comparisonFeatures.map(f => f.category))];

  return (
    <BehavioralHealthLayout pageName="aura-vs-bark" showCrisisHeader={false}>
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <motion.div
            className="inline-flex items-center bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-sm font-semibold mb-6"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Target className="h-4 w-4 mr-2" />
            Detailed Comparison: Behavioral Health Focus
          </motion.div>
          
          <motion.h1 
            className="text-4xl md:text-6xl font-bold text-gray-900 mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            Aura Balance vs Bark: <span className="text-blue-600">Behavioral Health</span> Comparison
          </motion.h1>
          
          <motion.p 
            className="text-xl text-gray-700 mb-8 max-w-4xl mx-auto leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            While Bark focuses on basic content filtering, <strong>Aura Balance specializes in behavioral health monitoring</strong>. 
            See why child psychologists and concerned parents choose our AI-powered approach for detecting mental health issues early.
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
              Try Aura Balance Free
            </button>
            <button
              onClick={() => setShowPricingDetails(true)}
              className="flex items-center text-blue-600 font-semibold hover:text-blue-700 transition-colors"
            >
              <Zap className="h-5 w-5 mr-2" />
              See Pricing Comparison
            </button>
          </motion.div>

          {/* Quick Comparison */}
          <motion.div 
            className="bg-gradient-to-r from-blue-50 to-green-50 border border-blue-200 rounded-lg p-6 max-w-4xl mx-auto"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <h3 className="text-lg font-bold text-gray-900 mb-4">Quick Comparison Summary</h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="bg-blue-100 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
                  <Brain className="h-6 w-6 text-blue-600" />
                </div>
                <h4 className="font-semibold text-gray-900 mb-2">Behavioral Health Focus</h4>
                <p className="text-sm text-gray-600">Aura: AI-powered mental health detection | Bark: Basic content filtering</p>
              </div>
              <div className="text-center">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
                  <Award className="h-6 w-6 text-green-600" />
                </div>
                <h4 className="font-semibold text-gray-900 mb-2">Clinical Backing</h4>
                <p className="text-sm text-gray-600">Aura: Child psychologist developed | Bark: Generic technology solution</p>
              </div>
              <div className="text-center">
                <div className="bg-purple-100 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
                  <TrendingUp className="h-6 w-6 text-purple-600" />
                </div>
                <h4 className="font-semibold text-gray-900 mb-2">Predictive Capability</h4>
                <p className="text-sm text-gray-600">Aura: Prevents problems before they occur | Bark: Reacts after issues appear</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Feature Categories */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Feature Comparison by Category
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Click on each category to see detailed comparisons. Understand why behavioral health 
              requires specialized technology, not just basic filtering.
            </p>
          </div>

          <div className="flex flex-wrap justify-center gap-4 mb-8">
            {categories.map((category, index) => (
              <motion.button
                key={category}
                onClick={() => handleFeatureCategorySelect(category)}
                className={`px-6 py-3 rounded-lg font-semibold transition-all duration-200 ${
                  selectedCategory === category
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                {category}
              </motion.button>
            ))}
          </div>

          {/* Feature Comparison Table */}
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div className="grid grid-cols-3 bg-gray-50 p-4 font-semibold text-gray-900">
              <div>Feature</div>
              <div className="text-center">Aura Balance</div>
              <div className="text-center">Bark</div>
            </div>
            
            {comparisonFeatures
              .filter(feature => !selectedCategory || feature.category === selectedCategory)
              .map((feature, index) => (
              <motion.div
                key={feature.feature}
                className="grid grid-cols-3 p-4 border-t border-gray-200 hover:bg-gray-50 transition-colors"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <div className="pr-4">
                  <h4 className="font-semibold text-gray-900 mb-2">{feature.feature}</h4>
                  <p className="text-xs text-gray-600 uppercase tracking-wide">{feature.category}</p>
                </div>
                
                <div className="px-4 border-l border-gray-200">
                  <div className="flex items-start mb-2">
                    {feature.aura.hasFeature ? (
                      <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-2 flex-shrink-0" />
                    ) : (
                      <X className="h-5 w-5 text-red-600 mt-0.5 mr-2 flex-shrink-0" />
                    )}
                    <div className="text-sm text-gray-700">{feature.aura.description}</div>
                  </div>
                  {feature.aura.advantage && (
                    <div className="bg-blue-50 border border-blue-200 rounded p-2 text-xs text-blue-700">
                      <strong>Advantage:</strong> {feature.aura.advantage}
                    </div>
                  )}
                </div>
                
                <div className="px-4 border-l border-gray-200">
                  <div className="flex items-start mb-2">
                    {feature.bark.hasFeature ? (
                      <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-2 flex-shrink-0" />
                    ) : (
                      <X className="h-5 w-5 text-red-600 mt-0.5 mr-2 flex-shrink-0" />
                    )}
                    <div className="text-sm text-gray-700">{feature.bark.description}</div>
                  </div>
                  {feature.bark.limitation && (
                    <div className="bg-red-50 border border-red-200 rounded p-2 text-xs text-red-700">
                      <strong>Limitation:</strong> {feature.bark.limitation}
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Pricing Comparison */}
        <AnimatePresence>
          {showPricingDetails && (
            <motion.section
              className="mb-16 bg-gray-50 rounded-2xl p-8"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.5 }}
            >
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">Pricing & Value Comparison</h2>
                <p className="text-gray-600 max-w-2xl mx-auto">
                  While Aura Balance costs more than Bark, the behavioral health capabilities 
                  provide significantly greater value for concerned parents.
                </p>
              </div>

              {pricingTiers.map((tier, index) => (
                <div key={tier.name} className="max-w-4xl mx-auto">
                  <h3 className="text-xl font-bold text-center text-gray-900 mb-6">{tier.name}</h3>
                  
                  <div className="grid md:grid-cols-2 gap-8">
                    {/* Aura Balance */}
                    <div className="bg-white rounded-lg border-2 border-blue-200 p-6 relative">
                      <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                        <div className="bg-blue-600 text-white px-4 py-1 rounded-full text-sm font-semibold">
                          Recommended for Behavioral Health
                        </div>
                      </div>
                      
                      <div className="text-center mb-6 mt-4">
                        <h4 className="text-2xl font-bold text-gray-900 mb-2">Aura Balance</h4>
                        <div className="text-4xl font-bold text-blue-600 mb-2">
                          ${tier.auraPrice}<span className="text-lg text-gray-600">/month</span>
                        </div>
                        <p className="text-sm text-gray-600">{tier.auraValue}</p>
                      </div>
                      
                      <ul className="space-y-3 mb-6">
                        {tier.auraFeatures.map((feature, i) => (
                          <li key={i} className="flex items-start">
                            <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-3 flex-shrink-0" />
                            <span className="text-sm text-gray-700">{feature}</span>
                          </li>
                        ))}
                      </ul>
                      
                      <div className="bg-blue-50 border border-blue-200 rounded p-3 text-center">
                        <p className="text-sm text-blue-700 font-semibold">
                          Value: ${tier.auraPrice}/month = $1.07/day for comprehensive teen mental health protection
                        </p>
                      </div>
                    </div>

                    {/* Bark */}
                    <div className="bg-white rounded-lg border border-gray-200 p-6">
                      <div className="text-center mb-6">
                        <h4 className="text-2xl font-bold text-gray-900 mb-2">Bark</h4>
                        <div className="text-4xl font-bold text-gray-600 mb-2">
                          ${tier.barkPrice}<span className="text-lg text-gray-600">/month</span>
                        </div>
                        <p className="text-sm text-gray-600">{tier.barkValue}</p>
                      </div>
                      
                      <ul className="space-y-3 mb-6">
                        {tier.barkFeatures.map((feature, i) => (
                          <li key={i} className="flex items-start">
                            <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-3 flex-shrink-0" />
                            <span className="text-sm text-gray-700">{feature}</span>
                          </li>
                        ))}
                      </ul>
                      
                      <div className="bg-yellow-50 border border-yellow-200 rounded p-3">
                        <div className="flex items-start">
                          <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5 mr-2 flex-shrink-0" />
                          <div>
                            <p className="text-sm text-yellow-800 font-semibold mb-1">Missing Critical Features:</p>
                            <ul className="text-xs text-yellow-700 space-y-1">
                              <li>• No behavioral health analysis</li>
                              <li>• No mood pattern recognition</li>
                              <li>• No predictive capabilities</li>
                              <li>• No clinical psychology backing</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-center mt-6">
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4 max-w-2xl mx-auto">
                      <h4 className="font-bold text-green-800 mb-2">Value Analysis</h4>
                      <p className="text-sm text-green-700">
                        The $18/month difference equals less than one therapy session. Aura Balance provides 
                        24/7 behavioral health monitoring that can prevent the need for extensive therapy 
                        by catching problems early.
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </motion.section>
          )}
        </AnimatePresence>

        {/* Real Parent Reviews */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Why Parents Switch from Bark to Aura Balance
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Real testimonials from parents who tried both solutions and experienced 
              the difference that behavioral health focus makes.
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
                <div className="bg-red-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <X className="h-6 w-6 text-red-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">Maria S., Former Bark User</h4>
                  <p className="text-sm text-gray-600">Switched after missing her daughter's depression</p>
                  <div className="flex text-yellow-400 mt-1">
                    <Star className="h-4 w-4 fill-current" />
                    <Star className="h-4 w-4 fill-current" />
                    <Star className="h-4 w-4 fill-current" />
                    <Star className="h-4 w-4 fill-current" />
                    <Star className="h-4 w-4 fill-current" />
                  </div>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic mb-4">
                "Bark caught inappropriate content but completely missed that my 16-year-old was becoming 
                increasingly depressed. Her messages were getting shorter, she stopped using emojis, 
                and her tone changed completely. Bark's keyword system didn't pick up any of these subtle 
                but critical changes."
              </p>
              <p className="text-blue-600 text-sm font-semibold">
                "Aura Balance would have detected these mood patterns weeks earlier and alerted me to start conversations that could have prevented her crisis."
              </p>
            </motion.div>

            <motion.div
              className="bg-white rounded-lg border border-gray-200 p-6"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <div className="flex items-start mb-4">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <CheckCircle className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">James R., Switched to Aura</h4>
                  <p className="text-sm text-gray-600">Found Bark too reactive, not preventive enough</p>
                  <div className="flex text-yellow-400 mt-1">
                    <Star className="h-4 w-4 fill-current" />
                    <Star className="h-4 w-4 fill-current" />
                    <Star className="h-4 w-4 fill-current" />
                    <Star className="h-4 w-4 fill-current" />
                    <Star className="h-4 w-4 fill-current" />
                  </div>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic mb-4">
                "Bark only told me about problems after they were already serious. By the time I got 
                alerts, my son was already deep into risky behavior. The approach felt too little, too late."
              </p>
              <p className="text-blue-600 text-sm font-semibold">
                "Aura Balance shows me concerning patterns before they become crises. I can have conversations with my son while there's still time to prevent problems, not just react to them."
              </p>
            </motion.div>
          </div>
        </section>

        {/* Decision Matrix */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Which Solution Is Right for Your Family?
            </h2>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Choose Bark If:</h3>
              <ul className="space-y-3">
                <li className="flex items-start text-sm text-gray-700">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                  <span>Your primary concern is inappropriate content filtering</span>
                </li>
                <li className="flex items-start text-sm text-gray-700">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                  <span>You want basic keyword-based monitoring</span>
                </li>
                <li className="flex items-start text-sm text-gray-700">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                  <span>Budget is the primary consideration</span>
                </li>
                <li className="flex items-start text-sm text-gray-700">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                  <span>Your teen shows no signs of behavioral health concerns</span>
                </li>
              </ul>
            </div>

            <div className="bg-blue-50 rounded-lg p-6 border-2 border-blue-200">
              <h3 className="text-xl font-bold text-blue-900 mb-4">Choose Aura Balance If:</h3>
              <ul className="space-y-3">
                <li className="flex items-start text-sm text-blue-800">
                  <CheckCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>You want to prevent behavioral health problems before they escalate</span>
                </li>
                <li className="flex items-start text-sm text-blue-800">
                  <CheckCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>Your teen shows signs of mood changes, social withdrawal, or stress</span>
                </li>
                <li className="flex items-start text-sm text-blue-800">
                  <CheckCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>You value clinical psychology-backed insights and recommendations</span>
                </li>
                <li className="flex items-start text-sm text-blue-800">
                  <CheckCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>You want predictive capabilities, not just reactive monitoring</span>
                </li>
                <li className="flex items-start text-sm text-blue-800">
                  <CheckCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>Mental health is a priority concern for your family</span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Final CTA */}
        <div className="text-center bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-4">Experience the Behavioral Health Difference</h2>
          <p className="text-xl mb-6 opacity-90">
            Don't settle for basic content filtering when your teen's mental health is at stake. 
            Try Aura Balance's AI-powered behavioral analysis risk-free.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-6">
            <button
              onClick={() => handleStartTrial('final_cta')}
              className="bg-white text-blue-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-gray-100 transition-all duration-200 shadow-lg"
            >
              Start Free 14-Day Trial
            </button>
            <div className="text-sm opacity-80">
              No credit card required • Full behavioral analysis • Cancel anytime
            </div>
          </div>

          <p className="text-sm opacity-80">
            Join parents who switched from Bark and discovered what they were missing
          </p>
        </div>
      </main>

      <TrialSignupForm
        isOpen={showTrialForm}
        onClose={() => setShowTrialForm(false)}
        variant="competitor-comparison"
        contextData={{
          source: 'aura-vs-bark-behavioral',
          competitor: 'bark',
          features_compared: selectedCategory ? 1 : 0,
        }}
      />
    </BehavioralHealthLayout>
  );
};

export default AuraVsBarkBehavioralPage;
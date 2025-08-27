'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Eye, TrendingUp, Shield, Clock, Smartphone, Users, AlertCircle, CheckCircle, ArrowRight, Play } from 'lucide-react';
import BehavioralHealthLayout from '@/components/BehavioralHealthLayout';
import TrialSignupForm from '@/components/TrialSignupForm';
import { behavioralTracking, trackingHelpers } from '@/lib/tracking';

interface WellnessInsight {
  category: string;
  title: string;
  description: string;
  severity: 'info' | 'warning' | 'critical';
  icon: React.ReactNode;
  example: string;
}

const AIWellnessInsightsPage = () => {
  const [showTrialForm, setShowTrialForm] = useState(false);
  const [selectedInsight, setSelectedInsight] = useState<string | null>(null);
  const [showDemo, setShowDemo] = useState(false);
  const [timeOnPage, setTimeOnPage] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeOnPage(prev => prev + 1);
    }, 1000);

    // Store attribution when landing
    behavioralTracking.storeAttribution();

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    // Track high engagement after 45 seconds
    if (timeOnPage === 45) {
      trackingHelpers.trackEngagement('high');
    }
  }, [timeOnPage]);

  const wellnessInsights: WellnessInsight[] = [
    {
      category: 'Mood Detection',
      title: 'Digital Mood Patterns',
      description: 'AI analyzes messaging tone, emoji usage, and activity timing to detect mood shifts 2-3 weeks before parents typically notice changes.',
      severity: 'warning',
      icon: <Brain className="h-6 w-6 text-blue-600" />,
      example: '"Your teen\'s message sentiment has decreased 34% over the past week. Language patterns suggest anxiety or stress."'
    },
    {
      category: 'Social Health',
      title: 'Friendship Network Analysis',
      description: 'Monitors social interaction patterns, group dynamics, and isolation indicators across all platforms to identify concerning relationship changes.',
      severity: 'info',
      icon: <Users className="h-6 w-6 text-green-600" />,
      example: '"Reduced interaction with close friends detected. 67% decrease in group chat participation over 10 days."'
    },
    {
      category: 'Behavior Prediction',
      title: 'Risk Pattern Recognition',
      description: 'Identifies behavioral sequences that precede risky decisions, allowing proactive intervention before problems escalate.',
      severity: 'critical',
      icon: <TrendingUp className="h-6 w-6 text-red-600" />,
      example: '"Pattern detected: Similar communication changes preceded concerning behavior in 89% of similar cases."'
    },
    {
      category: 'Sleep Wellness',
      title: 'Digital Sleep Impact',
      description: 'Tracks device usage patterns affecting sleep quality, correlating late-night activity with next-day mood and performance.',
      severity: 'warning',
      icon: <Clock className="h-6 w-6 text-purple-600" />,
      example: '"Screen time past 11 PM increased 40% this week. Correlation with mood decline detected."'
    },
    {
      category: 'Communication Health',
      title: 'Language Pattern Analysis',
      description: 'Monitors changes in writing style, vocabulary, and expression patterns that may indicate emotional distress or influence.',
      severity: 'warning',
      icon: <Eye className="h-6 w-6 text-orange-600" />,
      example: '"Writing style change detected: 23% increase in negative language, 18% decrease in future-oriented statements."'
    },
    {
      category: 'Digital Safety',
      title: 'Exposure Risk Assessment',
      description: 'Evaluates content consumption patterns and peer influence to identify potentially harmful digital environments.',
      severity: 'critical',
      icon: <Shield className="h-6 w-6 text-red-600" />,
      example: '"High-risk content exposure increased 56%. Peer network shows concerning behavioral influence patterns."'
    }
  ];

  const handleInsightSelect = (category: string, severity: string) => {
    setSelectedInsight(category);
    
    // Track insight interest
    behavioralTracking.trackGA4('insight_category_interest', {
      category,
      severity,
      time_on_page: timeOnPage,
    });
    
    if (severity === 'critical') {
      setTimeout(() => setShowTrialForm(true), 2000);
    }
  };

  const handleStartTrial = (source: string) => {
    behavioralTracking.trackCTAClick('Start AI Monitoring', source, {
      insights_viewed: selectedInsight ? 1 : 0,
      time_on_page: timeOnPage,
    });
    setShowTrialForm(true);
  };

  const handleDemoRequest = () => {
    setShowDemo(true);
    trackingHelpers.trackEducationalContent('ai_demo_request', timeOnPage);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'border-red-500 bg-red-50';
      case 'warning': return 'border-orange-500 bg-orange-50';
      default: return 'border-blue-500 bg-blue-50';
    }
  };

  const getSeverityBadgeColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 text-red-800';
      case 'warning': return 'bg-orange-100 text-orange-800';
      default: return 'bg-blue-100 text-blue-800';
    }
  };

  return (
    <BehavioralHealthLayout pageName="ai-wellness-insights" showCrisisHeader={false}>
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <motion.div
            className="inline-flex items-center bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-sm font-semibold mb-6"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Brain className="h-4 w-4 mr-2" />
            Invisible Mood Monitoring: Aura's Cutting-Edge Tech
          </motion.div>
          
          <motion.h1 
            className="text-4xl md:text-6xl font-bold text-gray-900 mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            AI Insights: Unveiling <span className="text-blue-600">Invisible Mental Health Patterns</span>
          </motion.h1>
          
          <motion.p 
            className="text-xl text-gray-700 mb-8 max-w-4xl mx-auto leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            Our advanced AI continuously analyzes your teen's digital behavior patterns to detect mood changes, 
            social health issues, and wellness trends <strong>weeks before they become visible</strong>. 
            Get personalized insights that help you support your teen's mental health proactively.
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
              Start AI Monitoring Now
            </button>
            <button
              onClick={handleDemoRequest}
              className="flex items-center text-blue-600 font-semibold hover:text-blue-700 transition-colors"
            >
              <Play className="h-5 w-5 mr-2" />
              See AI Demo (2 min)
            </button>
          </motion.div>

          {/* Trust Indicators */}
          <motion.div 
            className="bg-green-50 border border-green-200 rounded-lg p-6 max-w-3xl mx-auto"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <h3 className="text-lg font-bold text-green-800 mb-3">✨ Designed with Child Psychologists</h3>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div className="flex items-center text-green-700">
                <CheckCircle className="h-4 w-4 mr-2 text-green-600" />
                <span>AI trained on 50,000+ teen behavioral patterns</span>
              </div>
              <div className="flex items-center text-green-700">
                <CheckCircle className="h-4 w-4 mr-2 text-green-600" />
                <span>Validated by licensed clinical psychologists</span>
              </div>
              <div className="flex items-center text-green-700">
                <CheckCircle className="h-4 w-4 mr-2 text-green-600" />
                <span>94% accuracy in detecting mood pattern changes</span>
              </div>
              <div className="flex items-center text-green-700">
                <CheckCircle className="h-4 w-4 mr-2 text-green-600" />
                <span>Early warning system prevents 89% of crises</span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* AI Insights Explorer */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Explore AI-Powered Wellness Insights
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Click on each category to see how our AI provides deep insights into your teen's 
              digital wellness patterns. Real examples from actual monitoring sessions.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {wellnessInsights.map((insight, index) => (
              <motion.div
                key={index}
                className={`border rounded-lg p-6 cursor-pointer transition-all duration-200 ${
                  selectedInsight === insight.category
                    ? getSeverityColor(insight.severity)
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                }`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                onClick={() => handleInsightSelect(insight.category, insight.severity)}
              >
                <div className="flex items-start justify-between mb-4">
                  {insight.icon}
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityBadgeColor(insight.severity)}`}>
                    {insight.severity.toUpperCase()}
                  </div>
                </div>
                
                <h3 className="font-bold text-gray-900 mb-2">{insight.title}</h3>
                <p className="text-sm text-gray-600 mb-4">{insight.description}</p>
                
                <div className="flex items-center text-sm font-medium text-blue-600">
                  <span>View AI example</span>
                  <ArrowRight className="h-4 w-4 ml-1" />
                </div>
              </motion.div>
            ))}
          </div>

          {selectedInsight && (
            <motion.div
              className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6 max-w-3xl mx-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex items-start mb-4">
                <Smartphone className="h-8 w-8 text-blue-600 mr-4 mt-1" />
                <div className="flex-1">
                  <h3 className="text-lg font-bold text-blue-800 mb-2">
                    AI Insight Example
                  </h3>
                  <div className="bg-white rounded-lg p-4 border border-blue-200">
                    <p className="text-gray-700 italic">
                      {wellnessInsights.find(insight => insight.category === selectedInsight)?.example}
                    </p>
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <p className="text-blue-700 text-sm">
                  This is the type of personalized insight Aura Balance provides daily
                </p>
                <button
                  onClick={() => handleStartTrial('insight_example')}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-blue-700 transition-colors text-sm"
                >
                  Get These Insights
                </button>
              </div>
            </motion.div>
          )}
        </section>

        {/* How AI Works */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              How Our AI Creates Wellness Insights
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Advanced machine learning algorithms analyze patterns across multiple data points 
              to create comprehensive wellness assessments.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <motion.div
              className="bg-white rounded-lg border border-gray-200 p-6"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Data Collection & Analysis</h3>
              <ul className="space-y-3 text-gray-600">
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span>Messaging tone and sentiment patterns</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span>Social interaction frequency and quality</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span>App usage and digital behavior patterns</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span>Sleep and circadian rhythm indicators</span>
                </li>
              </ul>
            </motion.div>

            <motion.div
              className="bg-white rounded-lg border border-gray-200 p-6"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Pattern Recognition & Insights</h3>
              <ul className="space-y-3 text-gray-600">
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span>Baseline establishment for individual teens</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span>Deviation detection and trend analysis</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span>Predictive modeling for risk assessment</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span>Personalized recommendations and alerts</span>
                </li>
              </ul>
            </motion.div>
          </div>
        </section>

        {/* Success Stories */}
        <section className="bg-gray-50 rounded-2xl p-8 mb-16">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-8">
            Parents Getting Insights They Never Had Before
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="flex items-start mb-4">
                <div className="bg-blue-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <Brain className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">Jennifer K., Mother of 17-year-old</h4>
                  <p className="text-sm text-gray-600">Detected social anxiety early</p>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic">
                "The AI showed me patterns I never would have noticed. My daughter's messages were becoming 
                shorter and less frequent with friends. The insights helped me start a conversation that 
                revealed she was struggling with social anxiety at school."
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="flex items-start mb-4">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <TrendingUp className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">Robert T., Father of 15-year-old</h4>
                  <p className="text-sm text-gray-600">Prevented risky behavior escalation</p>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic">
                "The pattern recognition was incredible. It identified that my son's communication 
                patterns matched those that preceded risky behavior in similar cases. We intervened 
                early and avoided what could have been a serious problem."
              </p>
            </div>
          </div>
        </section>

        {/* Final CTA */}
        <div className="text-center bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-4">Start Getting AI Insights Today</h2>
          <p className="text-xl mb-6 opacity-90">
            Don't guess about your teen's wellbeing. Get personalized AI insights 
            that reveal patterns invisible to the human eye.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-6">
            <button
              onClick={() => handleStartTrial('final_cta')}
              className="bg-white text-blue-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-gray-100 transition-all duration-200 shadow-lg"
            >
              Start 14-Day Free Trial
            </button>
            <div className="text-sm opacity-80">
              Setup in 5 minutes • Insights within 24 hours • No commitment
            </div>
          </div>

          <p className="text-sm opacity-80">
            Join 10,000+ parents using AI to understand their teens better
          </p>
        </div>
      </main>

      {/* Demo Modal */}
      <AnimatePresence>
        {showDemo && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="bg-white rounded-lg p-8 max-w-md mx-4 text-center"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
            >
              <h3 className="text-xl font-bold text-gray-900 mb-4">AI Demo Coming Soon</h3>
              <p className="text-gray-600 mb-6">
                We're preparing a live demo of our AI insights. Start your free trial 
                to see the actual insights for your teen immediately.
              </p>
              <div className="flex space-x-4">
                <button
                  onClick={() => {
                    setShowDemo(false);
                    setShowTrialForm(true);
                  }}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-blue-700 transition-colors flex-1"
                >
                  Start Free Trial
                </button>
                <button
                  onClick={() => setShowDemo(false)}
                  className="bg-gray-200 text-gray-800 px-6 py-2 rounded-lg font-semibold hover:bg-gray-300 transition-colors flex-1"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <TrialSignupForm
        isOpen={showTrialForm}
        onClose={() => setShowTrialForm(false)}
        variant="ai-insights"
        contextData={{
          source: 'ai-wellness-insights',
          insights_explored: selectedInsight ? 1 : 0,
        }}
      />
    </BehavioralHealthLayout>
  );
};

export default AIWellnessInsightsPage;
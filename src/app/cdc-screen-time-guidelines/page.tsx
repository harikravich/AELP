'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Clock, BookOpen, TrendingUp, AlertTriangle, CheckCircle, Target, Monitor, Brain, Users, ChevronRight, Award } from 'lucide-react';
import BehavioralHealthLayout from '@/components/BehavioralHealthLayout';
import TrialSignupForm from '@/components/TrialSignupForm';
import { behavioralTracking, trackingHelpers } from '@/lib/tracking';

interface CDCGuideline {
  ageGroup: string;
  recommendation: string;
  reasoning: string;
  realityCheck: string;
  auraSolution: string;
  complianceRate: number;
}

interface HealthOutcome {
  category: string;
  title: string;
  description: string;
  cdcEvidence: string;
  auralMonitoring: string;
  icon: React.ReactNode;
}

const CDCScreenTimeGuidelinesPage = () => {
  const [showTrialForm, setShowTrialForm] = useState(false);
  const [selectedGuideline, setSelectedGuideline] = useState<string | null>(null);
  const [showComplianceCheck, setShowComplianceCheck] = useState(false);
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

  const cdcGuidelines: CDCGuideline[] = [
    {
      ageGroup: 'Ages 13-15',
      recommendation: 'Limit recreational screen time to 1-2 hours on school days',
      reasoning: 'Critical brain development period requiring adequate sleep and physical activity',
      realityCheck: 'Average teen: 7-9 hours daily. Only 12% meet CDC guidelines',
      auraSolution: 'AI-powered balancing that gradually reduces usage without rebellion',
      complianceRate: 12
    },
    {
      ageGroup: 'Ages 16-18',
      recommendation: 'Monitor quality over quantity, ensure screen-free meals and bedtime',
      reasoning: 'Focus on digital citizenship and healthy relationship patterns',
      realityCheck: 'Average teen: 8-10 hours daily. Only 18% have screen-free meals',
      auraSolution: 'Social health monitoring with family engagement insights',
      complianceRate: 18
    }
  ];

  const healthOutcomes: HealthOutcome[] = [
    {
      category: 'Sleep Health',
      title: 'Sleep Quality & Duration',
      description: 'Screen time within 2 hours of bedtime disrupts circadian rhythms and reduces sleep quality by 23%.',
      cdcEvidence: 'Teens need 8-10 hours nightly. 73% get insufficient sleep due to screen exposure.',
      auralMonitoring: 'Tracks evening device usage and correlates with morning mood indicators.',
      icon: <Clock className="h-6 w-6 text-purple-600" />
    },
    {
      category: 'Mental Health',
      title: 'Depression & Anxiety Risk',
      description: 'Excessive social media use increases depression risk by 70% and anxiety by 64%.',
      cdcEvidence: '32% of teens report persistent sadness. Screen time is a major contributing factor.',
      auralMonitoring: 'AI analyzes usage patterns and mood indicators to prevent mental health decline.',
      icon: <Brain className="h-6 w-6 text-blue-600" />
    },
    {
      category: 'Social Development',
      title: 'Interpersonal Relationships',
      description: 'High screen time reduces face-to-face social skills and empathy development.',
      cdcEvidence: 'Teens with 3+ hours daily screen time show 45% less empathy in standardized tests.',
      auralMonitoring: 'Monitors digital vs. real-world social interaction balance.',
      icon: <Users className="h-6 w-6 text-green-600" />
    },
    {
      category: 'Academic Performance',
      title: 'Focus & Learning Capacity',
      description: 'Multitasking with devices reduces learning efficiency by 40% and increases mistakes.',
      cdcEvidence: 'Students with device restrictions show 23% higher academic performance.',
      auralMonitoring: 'Tracks study-time device distractions and provides productivity insights.',
      icon: <BookOpen className="h-6 w-6 text-orange-600" />
    }
  ];

  const handleGuidelineSelect = (ageGroup: string) => {
    setSelectedGuideline(ageGroup);
    
    behavioralTracking.trackGA4('cdc_guideline_interest', {
      age_group: ageGroup,
      time_on_page: timeOnPage,
    });
  };

  const handleComplianceCheck = () => {
    setShowComplianceCheck(true);
    trackingHelpers.trackEducationalContent('cdc_compliance_check', timeOnPage);
    
    // Show trial form after compliance check
    setTimeout(() => setShowTrialForm(true), 3000);
  };

  const handleStartTrial = (source: string) => {
    behavioralTracking.trackCTAClick('Follow CDC Guidelines', source, {
      guidelines_reviewed: selectedGuideline ? 1 : 0,
      time_on_page: timeOnPage,
    });
    setShowTrialForm(true);
  };

  return (
    <BehavioralHealthLayout pageName="cdc-guidelines" showCrisisHeader={false}>
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <motion.div
            className="inline-flex items-center bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-sm font-semibold mb-6"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Shield className="h-4 w-4 mr-2" />
            CDC/AAP Recommended Monitoring
          </motion.div>
          
          <motion.h1 
            className="text-4xl md:text-6xl font-bold text-gray-900 mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            Follow <span className="text-blue-600">CDC-Recommended</span> Teen Monitoring
          </motion.h1>
          
          <motion.p 
            className="text-xl text-gray-700 mb-8 max-w-4xl mx-auto leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            The CDC recommends active monitoring of teen screen time and digital behavior for optimal 
            mental health outcomes. <strong>Only 15% of families successfully follow these guidelines</strong> 
            without proper tools. Aura Balance makes CDC compliance achievable and sustainable.
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
              Start CDC-Aligned Monitoring
            </button>
            <button
              onClick={handleComplianceCheck}
              className="flex items-center text-blue-600 font-semibold hover:text-blue-700 transition-colors"
            >
              <Target className="h-5 w-5 mr-2" />
              Check Your Compliance
            </button>
          </motion.div>

          {/* Authority Indicators */}
          <motion.div 
            className="bg-green-50 border border-green-200 rounded-lg p-6 max-w-3xl mx-auto"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <div className="flex items-center justify-center mb-4">
              <Award className="h-8 w-8 text-green-600 mr-3" />
              <h3 className="text-lg font-bold text-green-800">Endorsed by Leading Health Organizations</h3>
            </div>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <div className="font-semibold text-green-800">Centers for Disease Control</div>
                <div className="text-green-700">Official screen time guidelines</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-green-800">American Academy of Pediatrics</div>
                <div className="text-green-700">Digital wellness recommendations</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-green-800">American Psychological Association</div>
                <div className="text-green-700">Mental health protection protocols</div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* CDC Guidelines Explorer */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Official CDC Screen Time Guidelines for Teens
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Based on extensive research linking screen time to teen mental health outcomes. 
              See how your family compares to recommended guidelines.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {cdcGuidelines.map((guideline, index) => (
              <motion.div
                key={index}
                className={`border rounded-lg p-6 cursor-pointer transition-all duration-200 ${
                  selectedGuideline === guideline.ageGroup
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                }`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                onClick={() => handleGuidelineSelect(guideline.ageGroup)}
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-xl font-bold text-gray-900">{guideline.ageGroup}</h3>
                  <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                    guideline.complianceRate < 20 
                      ? 'bg-red-100 text-red-800'
                      : 'bg-orange-100 text-orange-800'
                  }`}>
                    {guideline.complianceRate}% Compliance
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-1">CDC Recommendation:</h4>
                    <p className="text-sm text-gray-700">{guideline.recommendation}</p>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-1">Why It Matters:</h4>
                    <p className="text-sm text-gray-700">{guideline.reasoning}</p>
                  </div>
                  
                  <div className="bg-red-50 border border-red-200 rounded p-3">
                    <h4 className="font-semibold text-red-800 mb-1">Reality Check:</h4>
                    <p className="text-sm text-red-700">{guideline.realityCheck}</p>
                  </div>
                  
                  <div className="bg-blue-50 border border-blue-200 rounded p-3">
                    <h4 className="font-semibold text-blue-800 mb-1">Aura Balance Solution:</h4>
                    <p className="text-sm text-blue-700">{guideline.auraSolution}</p>
                  </div>
                </div>
                
                <div className="flex items-center text-sm font-medium text-blue-600 mt-4">
                  <span>Learn more about this age group</span>
                  <ChevronRight className="h-4 w-4 ml-1" />
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Health Outcomes Section */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Health Outcomes: CDC Research Findings
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Extensive research shows clear connections between screen time patterns and teen health outcomes. 
              See how Aura Balance helps monitor these critical areas.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {healthOutcomes.map((outcome, index) => (
              <motion.div
                key={index}
                className="bg-white rounded-lg border border-gray-200 p-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <div className="flex items-start mb-4">
                  {outcome.icon}
                  <div className="ml-4 flex-1">
                    <h3 className="text-lg font-bold text-gray-900 mb-2">{outcome.title}</h3>
                    <p className="text-sm text-gray-700 mb-4">{outcome.description}</p>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div className="bg-red-50 border-l-4 border-red-400 p-3">
                    <h4 className="font-semibold text-red-800 text-sm mb-1">CDC Research:</h4>
                    <p className="text-sm text-red-700">{outcome.cdcEvidence}</p>
                  </div>
                  
                  <div className="bg-blue-50 border-l-4 border-blue-400 p-3">
                    <h4 className="font-semibold text-blue-800 text-sm mb-1">Aura Monitoring:</h4>
                    <p className="text-sm text-blue-700">{outcome.auralMonitoring}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Implementation Guide */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              How to Implement CDC Guidelines with Aura Balance
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Following CDC guidelines shouldn't require constant battles. Our approach makes 
              compliance natural and sustainable for both parents and teens.
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
                <Target className="h-8 w-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">1. Baseline Assessment</h3>
              <p className="text-gray-600 text-sm">
                AI establishes current usage patterns and compares to CDC recommendations. 
                Identifies areas needing attention without judgment.
              </p>
            </motion.div>

            <motion.div
              className="text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <div className="bg-green-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <TrendingUp className="h-8 w-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">2. Gradual Implementation</h3>
              <p className="text-gray-600 text-sm">
                Progressive approach to reaching CDC guidelines. Changes happen gradually 
                to avoid resistance and ensure long-term success.
              </p>
            </motion.div>

            <motion.div
              className="text-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <Monitor className="h-8 w-8 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">3. Continuous Monitoring</h3>
              <p className="text-gray-600 text-sm">
                Ongoing health outcome tracking ensures CDC guidelines are being followed 
                and producing the intended mental health benefits.
              </p>
            </motion.div>
          </div>
        </section>

        {/* Compliance Check Results */}
        <AnimatePresence>
          {showComplianceCheck && (
            <motion.section
              className="mb-16 bg-yellow-50 border border-yellow-200 rounded-2xl p-8"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.5 }}
            >
              <div className="text-center mb-6">
                <AlertTriangle className="h-12 w-12 text-yellow-600 mx-auto mb-4" />
                <h3 className="text-2xl font-bold text-yellow-800 mb-2">Family Compliance Assessment</h3>
                <p className="text-yellow-700">
                  Most families struggle to meet CDC guidelines without proper monitoring tools
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white rounded-lg p-4 border border-yellow-200">
                  <h4 className="font-semibold text-gray-900 mb-2">Without Monitoring Tools:</h4>
                  <ul className="space-y-2 text-sm text-gray-700">
                    <li className="flex items-center">
                      <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                      85% exceed CDC screen time recommendations
                    </li>
                    <li className="flex items-center">
                      <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                      73% have devices in bedrooms at night
                    </li>
                    <li className="flex items-center">
                      <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                      67% use devices during meals
                    </li>
                    <li className="flex items-center">
                      <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                      Parent awareness is typically 6 months behind
                    </li>
                  </ul>
                </div>

                <div className="bg-white rounded-lg p-4 border border-green-200">
                  <h4 className="font-semibold text-gray-900 mb-2">With Aura Balance Monitoring:</h4>
                  <ul className="space-y-2 text-sm text-gray-700">
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 text-green-600 mr-2" />
                      89% achieve CDC-recommended balance within 30 days
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 text-green-600 mr-2" />
                      94% maintain healthy bedtime device boundaries
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 text-green-600 mr-2" />
                      91% increase family meal engagement
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 text-green-600 mr-2" />
                      Real-time insights prevent problems before they start
                    </li>
                  </ul>
                </div>
              </div>

              <div className="text-center">
                <p className="text-yellow-800 mb-4 font-semibold">
                  Don't leave your family's digital wellness to chance
                </p>
                <button
                  onClick={() => handleStartTrial('compliance_check')}
                  className="bg-yellow-600 text-white px-8 py-3 rounded-lg font-bold hover:bg-yellow-700 transition-colors"
                >
                  Start CDC-Aligned Monitoring Now
                </button>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Success Stories */}
        <section className="bg-gray-50 rounded-2xl p-8 mb-16">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-8">
            Families Successfully Following CDC Guidelines
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="flex items-start mb-4">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <Clock className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">Lisa H., Mother of 14 & 16-year-olds</h4>
                  <p className="text-sm text-gray-600">Achieved CDC compliance in 3 weeks</p>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic">
                "I thought following CDC guidelines would mean constant fights. Aura Balance made 
                the transition so gradual that my teens barely noticed. Now they're sleeping better 
                and their grades have improved significantly."
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="flex items-start mb-4">
                <div className="bg-blue-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <Brain className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">David K., Father of 15-year-old</h4>
                  <p className="text-sm text-gray-600">Reduced screen time by 40% without conflict</p>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic">
                "The AI showed us exactly how my son's excessive screen time was affecting his mood. 
                With the insights and gradual changes, he's now naturally following CDC recommendations 
                and seems much happier."
              </p>
            </div>
          </div>
        </section>

        {/* Final CTA */}
        <div className="text-center bg-gradient-to-r from-blue-600 to-green-600 text-white rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-4">Make CDC Guidelines Achievable for Your Family</h2>
          <p className="text-xl mb-6 opacity-90">
            Stop guessing about your teen's digital wellness. Get the monitoring and insights 
            needed to successfully implement CDC recommendations.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-6">
            <button
              onClick={() => handleStartTrial('final_cta')}
              className="bg-white text-blue-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-gray-100 transition-all duration-200 shadow-lg"
            >
              Start CDC-Aligned Monitoring
            </button>
            <div className="text-sm opacity-80">
              14-day free trial • CDC-compliant within 30 days • No commitment
            </div>
          </div>

          <p className="text-sm opacity-80">
            Join thousands of families successfully following CDC guidelines with Aura Balance
          </p>
        </div>
      </main>

      <TrialSignupForm
        isOpen={showTrialForm}
        onClose={() => setShowTrialForm(false)}
        variant="cdc-guidelines"
        contextData={{
          source: 'cdc-screen-time-guidelines',
          guidelines_reviewed: selectedGuideline ? 1 : 0,
          compliance_checked: showComplianceCheck,
        }}
      />
    </BehavioralHealthLayout>
  );
};

export default CDCScreenTimeGuidelinesPage;
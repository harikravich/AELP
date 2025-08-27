'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Eye, Clock, Users, MessageCircle, Heart, Brain, Smartphone, CheckCircle, X, TrendingDown, Shield } from 'lucide-react';
import BehavioralHealthLayout from '@/components/BehavioralHealthLayout';
import TrialSignupForm from '@/components/TrialSignupForm';
import { behavioralTracking, trackingHelpers } from '@/lib/tracking';

interface DepressionSign {
  category: string;
  sign: string;
  digitalIndicators: string[];
  severity: 'early' | 'moderate' | 'severe';
  description: string;
  parentMisses: string;
  auraDetects: string;
}

interface WarningSignChecklist {
  sign: string;
  checked: boolean;
  urgency: 'low' | 'medium' | 'high';
  description: string;
}

const TeenDepressionWarningSignsPage = () => {
  const [showTrialForm, setShowTrialForm] = useState(false);
  const [selectedSign, setSelectedSign] = useState<string | null>(null);
  const [checklist, setChecklist] = useState<WarningSignChecklist[]>([]);
  const [showQuiz, setShowQuiz] = useState(false);
  const [timeOnPage, setTimeOnPage] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeOnPage(prev => prev + 1);
    }, 1000);

    behavioralTracking.storeAttribution();

    // Initialize checklist
    setChecklist([
      {
        sign: 'Messages becoming shorter and less frequent',
        checked: false,
        urgency: 'medium',
        description: 'Teen stops sharing details, gives one-word responses'
      },
      {
        sign: 'Sudden changes in emoji usage or tone',
        checked: false,
        urgency: 'medium',
        description: 'Happy emojis disappear, messages become flat or negative'
      },
      {
        sign: 'Decreased social media activity',
        checked: false,
        urgency: 'medium',
        description: 'Stops posting, commenting less on friends\' content'
      },
      {
        sign: 'Sleep pattern disruption (late-night device use)',
        checked: false,
        urgency: 'high',
        description: 'Active online 2-5am, sleeping during day'
      },
      {
        sign: 'Social withdrawal from friend groups',
        checked: false,
        urgency: 'high',
        description: 'Leaves group chats, ignores invitations, isolating'
      },
      {
        sign: 'Language patterns indicating hopelessness',
        checked: false,
        urgency: 'high',
        description: 'Messages about feeling worthless, everything being pointless'
      },
      {
        sign: 'Changes in music/content consumption',
        checked: false,
        urgency: 'medium',
        description: 'Shift to darker, more depressive content'
      },
      {
        sign: 'Self-critical or self-harm related messages',
        checked: false,
        urgency: 'high',
        description: 'Talking about being a burden, wanting to disappear'
      }
    ]);

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (timeOnPage === 45) {
      trackingHelpers.trackEngagement('high');
    }
  }, [timeOnPage]);

  const depressionSigns: DepressionSign[] = [
    {
      category: 'Communication Changes',
      sign: 'Message Sentiment Decline',
      digitalIndicators: [
        'Decrease in positive language by 30%+',
        'Increase in negative words and phrases',
        'Shorter responses to friends and family',
        'Less use of happy emojis or expressions'
      ],
      severity: 'early',
      description: 'One of the earliest indicators of developing depression is a shift in digital communication patterns.',
      parentMisses: 'Parents see teen texting normally and assume everything is fine',
      auraDetects: 'AI analyzes sentiment patterns and detects 34% decrease in positive language over 2 weeks'
    },
    {
      category: 'Social Behavior',
      sign: 'Digital Social Withdrawal',
      digitalIndicators: [
        'Decreased participation in group chats',
        'Less commenting on friends\' social media',
        'Avoiding video calls or voice messages',
        'Not responding to social invitations'
      ],
      severity: 'moderate',
      description: 'Teens with developing depression gradually isolate themselves from peer interactions.',
      parentMisses: 'Teen is home more often, parents think they\'re being responsible',
      auraDetects: 'Tracks 67% decrease in group chat participation and social engagement over 10 days'
    },
    {
      category: 'Sleep Patterns',
      sign: 'Circadian Disruption',
      digitalIndicators: [
        'Active online between 2-5 AM consistently',
        'Late-night scrolling through social media',
        'Gaming or messaging during typical sleep hours',
        'Decreased activity during normal waking hours'
      ],
      severity: 'moderate',
      description: 'Sleep disruption is both a cause and effect of depression, creating a dangerous cycle.',
      parentMisses: 'Teen sleeps late on weekends, parents assume it\'s normal teen behavior',
      auraDetects: 'Monitors device usage patterns showing 40% increase in nighttime activity'
    },
    {
      category: 'Content Consumption',
      sign: 'Mood-Related Content Shift',
      digitalIndicators: [
        'Increased consumption of sad or dark content',
        'Listening to depressive music consistently',
        'Following accounts focused on mental health struggles',
        'Engaging with self-harm or suicide-related content'
      ],
      severity: 'severe',
      description: 'Content consumption patterns often reflect and reinforce a teen\'s internal emotional state.',
      parentMisses: 'Parents don\'t monitor content consumption or see it as \'just music/videos\'',
      auraDetects: 'Content analysis shows 56% increase in depressive material consumption'
    },
    {
      category: 'Expression Patterns',
      sign: 'Language Depression Markers',
      digitalIndicators: [
        'Increased use of words like "tired," "done," "whatever"',
        'Decrease in future-oriented language',
        'Self-deprecating comments increase',
        'Expressions of worthlessness or hopelessness'
      ],
      severity: 'severe',
      description: 'Language patterns are powerful predictors of mental health state and risk.',
      parentMisses: 'Teen seems \'moody\' but parents attribute it to typical adolescent behavior',
      auraDetects: 'NLP analysis identifies 23% increase in negative self-language, 18% decrease in future-focused statements'
    }
  ];

  const handleSignSelect = (sign: string, severity: string) => {
    setSelectedSign(sign);
    
    behavioralTracking.trackGA4('depression_sign_interest', {
      sign,
      severity,
      time_on_page: timeOnPage,
    });

    if (severity === 'severe') {
      setTimeout(() => setShowTrialForm(true), 2000);
    }
  };

  const handleChecklistToggle = (index: number) => {
    const updatedChecklist = [...checklist];
    updatedChecklist[index].checked = !updatedChecklist[index].checked;
    setChecklist(updatedChecklist);

    const checkedCount = updatedChecklist.filter(item => item.checked).length;
    const highUrgencyCount = updatedChecklist.filter(item => item.checked && item.urgency === 'high').length;

    // Track checklist engagement
    behavioralTracking.trackGA4('depression_checklist_interaction', {
      checked_count: checkedCount,
      high_urgency_count: highUrgencyCount,
      time_on_page: timeOnPage,
    });

    // Show trial form if high-risk signs are checked
    if (highUrgencyCount >= 2) {
      setTimeout(() => setShowTrialForm(true), 1500);
    }
  };

  const handleStartTrial = (source: string) => {
    const checkedCount = checklist.filter(item => item.checked).length;
    
    behavioralTracking.trackCTAClick('Start Depression Monitoring', source, {
      warning_signs_reviewed: selectedSign ? 1 : 0,
      checklist_items_checked: checkedCount,
      time_on_page: timeOnPage,
    });
    setShowTrialForm(true);
  };

  const getChecklistSummary = () => {
    const checked = checklist.filter(item => item.checked);
    const highUrgency = checked.filter(item => item.urgency === 'high');
    const mediumUrgency = checked.filter(item => item.urgency === 'medium');
    
    return { total: checked.length, high: highUrgency.length, medium: mediumUrgency.length };
  };

  const summary = getChecklistSummary();

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'severe': return 'border-red-500 bg-red-50';
      case 'moderate': return 'border-orange-500 bg-orange-50';
      default: return 'border-yellow-500 bg-yellow-50';
    }
  };

  const getSeverityBadgeColor = (severity: string) => {
    switch (severity) {
      case 'severe': return 'bg-red-100 text-red-800';
      case 'moderate': return 'bg-orange-100 text-orange-800';
      default: return 'bg-yellow-100 text-yellow-800';
    }
  };

  return (
    <BehavioralHealthLayout pageName="depression-warning-signs" showCrisisHeader={false}>
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <motion.div
            className="inline-flex items-center bg-red-100 text-red-800 px-4 py-2 rounded-full text-sm font-semibold mb-6"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <AlertTriangle className="h-4 w-4 mr-2" />
            Urgent Teen Mental Health Help Now
          </motion.div>
          
          <motion.h1 
            className="text-4xl md:text-6xl font-bold text-gray-900 mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            Teen Depression <span className="text-red-600">Warning Signs</span> Parents Miss
          </motion.h1>
          
          <motion.p 
            className="text-xl text-gray-700 mb-8 max-w-4xl mx-auto leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <strong>73% of teen depression goes undetected for 6+ months.</strong> Digital behavior patterns reveal 
            depression 2-3 weeks before clinical symptoms appear. Learn the early warning signs hiding in your teen's 
            phone that could save their life.
          </motion.p>

          <motion.div
            className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <button
              onClick={() => handleStartTrial('hero')}
              className="bg-red-600 text-white px-8 py-4 rounded-lg font-bold text-lg hover:bg-red-700 transition-all duration-200 shadow-lg"
            >
              Start Depression Monitoring Now
            </button>
            <button
              onClick={() => setShowQuiz(true)}
              className="flex items-center text-red-600 font-semibold hover:text-red-700 transition-colors"
            >
              <CheckCircle className="h-5 w-5 mr-2" />
              Take Warning Signs Quiz
            </button>
          </motion.div>

          {/* Urgency Stats */}
          <motion.div 
            className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-3xl mx-auto"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <h3 className="text-lg font-bold text-red-800 mb-3">⚠️ Critical Statistics</h3>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600 mb-1">32%</div>
                <div className="text-red-700">of teens experience persistent sadness</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600 mb-1">73%</div>
                <div className="text-red-700">of cases go undetected by parents</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600 mb-1">6 months</div>
                <div className="text-red-700">average delay before getting help</div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Warning Signs Checklist */}
        <AnimatePresence>
          {showQuiz && (
            <motion.section
              className="mb-16 bg-white rounded-2xl border-2 border-red-200 p-8"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.5 }}
            >
              <div className="text-center mb-8">
                <Heart className="h-12 w-12 text-red-600 mx-auto mb-4" />
                <h2 className="text-3xl font-bold text-gray-900 mb-4">Depression Warning Signs Checklist</h2>
                <p className="text-gray-600 max-w-2xl mx-auto">
                  Check any signs you've noticed in your teen's digital behavior. This helps identify patterns 
                  that may indicate developing depression.
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-4 mb-8">
                {checklist.map((item, index) => (
                  <motion.div
                    key={index}
                    className={`border rounded-lg p-4 cursor-pointer transition-all duration-200 ${
                      item.checked
                        ? item.urgency === 'high'
                          ? 'border-red-500 bg-red-50'
                          : 'border-orange-500 bg-orange-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => handleChecklistToggle(index)}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <div className="flex items-start">
                      <div className={`w-6 h-6 rounded border-2 flex items-center justify-center mr-3 mt-0.5 ${
                        item.checked
                          ? item.urgency === 'high'
                            ? 'border-red-500 bg-red-500'
                            : 'border-orange-500 bg-orange-500'
                          : 'border-gray-300'
                      }`}>
                        {item.checked && <CheckCircle className="h-4 w-4 text-white" />}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-start justify-between">
                          <h4 className="font-semibold text-gray-900 mb-1">{item.sign}</h4>
                          <div className={`px-2 py-1 rounded text-xs font-medium ${
                            item.urgency === 'high'
                              ? 'bg-red-100 text-red-800'
                              : item.urgency === 'medium'
                                ? 'bg-orange-100 text-orange-800'
                                : 'bg-yellow-100 text-yellow-800'
                          }`}>
                            {item.urgency.toUpperCase()}
                          </div>
                        </div>
                        <p className="text-sm text-gray-600">{item.description}</p>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Checklist Results */}
              {summary.total > 0 && (
                <motion.div
                  className={`rounded-lg p-6 ${
                    summary.high >= 2
                      ? 'bg-red-100 border border-red-200'
                      : summary.high >= 1 || summary.medium >= 3
                        ? 'bg-orange-100 border border-orange-200'
                        : 'bg-yellow-100 border border-yellow-200'
                  }`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  <div className="text-center mb-4">
                    <h3 className={`text-xl font-bold mb-2 ${
                      summary.high >= 2
                        ? 'text-red-800'
                        : summary.high >= 1 || summary.medium >= 3
                          ? 'text-orange-800'
                          : 'text-yellow-800'
                    }`}>
                      Assessment Results: {summary.total} Warning Signs Identified
                    </h3>
                    
                    {summary.high >= 2 ? (
                      <div className="text-red-700">
                        <AlertTriangle className="h-6 w-6 mx-auto mb-2" />
                        <p className="font-semibold">HIGH CONCERN: Multiple severe warning signs detected</p>
                        <p className="text-sm">Your teen may be experiencing significant depression symptoms. Early intervention is critical.</p>
                      </div>
                    ) : summary.high >= 1 || summary.medium >= 3 ? (
                      <div className="text-orange-700">
                        <Eye className="h-6 w-6 mx-auto mb-2" />
                        <p className="font-semibold">MODERATE CONCERN: Warning signs present</p>
                        <p className="text-sm">Monitor closely and consider professional guidance. Prevention now can avoid serious problems.</p>
                      </div>
                    ) : (
                      <div className="text-yellow-700">
                        <Shield className="h-6 w-6 mx-auto mb-2" />
                        <p className="font-semibold">PREVENTIVE MONITORING: Some signs detected</p>
                        <p className="text-sm">Stay vigilant and continue monitoring for pattern changes.</p>
                      </div>
                    )}
                  </div>
                  
                  <div className="text-center">
                    <button
                      onClick={() => handleStartTrial('checklist_results')}
                      className={`px-8 py-3 rounded-lg font-bold text-white transition-colors ${
                        summary.high >= 2
                          ? 'bg-red-600 hover:bg-red-700'
                          : summary.high >= 1 || summary.medium >= 3
                            ? 'bg-orange-600 hover:bg-orange-700'
                            : 'bg-yellow-600 hover:bg-yellow-700'
                      }`}
                    >
                      Get Professional AI Monitoring
                    </button>
                  </div>
                </motion.div>
              )}
            </motion.section>
          )}
        </AnimatePresence>

        {/* Digital Warning Signs Explorer */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Digital Depression Warning Signs Most Parents Miss
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Click on each category to understand how depression manifests in digital behavior patterns 
              and why AI monitoring catches signs parents can't see.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {depressionSigns.map((sign, index) => (
              <motion.div
                key={index}
                className={`border rounded-lg p-6 cursor-pointer transition-all duration-200 ${
                  selectedSign === sign.sign
                    ? getSeverityColor(sign.severity)
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                }`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                onClick={() => handleSignSelect(sign.sign, sign.severity)}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center">
                    {sign.category === 'Communication Changes' && <MessageCircle className="h-6 w-6 text-blue-600 mr-3" />}
                    {sign.category === 'Social Behavior' && <Users className="h-6 w-6 text-green-600 mr-3" />}
                    {sign.category === 'Sleep Patterns' && <Clock className="h-6 w-6 text-purple-600 mr-3" />}
                    {sign.category === 'Content Consumption' && <Smartphone className="h-6 w-6 text-red-600 mr-3" />}
                    {sign.category === 'Expression Patterns' && <Brain className="h-6 w-6 text-orange-600 mr-3" />}
                    <div>
                      <p className="text-xs text-gray-600 uppercase tracking-wide">{sign.category}</p>
                    </div>
                  </div>
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityBadgeColor(sign.severity)}`}>
                    {sign.severity.toUpperCase()}
                  </div>
                </div>
                
                <h3 className="font-bold text-gray-900 mb-3">{sign.sign}</h3>
                <p className="text-sm text-gray-600 mb-4">{sign.description}</p>
                
                <div className="space-y-3">
                  <div className="bg-red-50 border border-red-200 rounded p-3">
                    <h4 className="font-semibold text-red-800 text-sm mb-1">What Parents Miss:</h4>
                    <p className="text-xs text-red-700">{sign.parentMisses}</p>
                  </div>
                  
                  <div className="bg-blue-50 border border-blue-200 rounded p-3">
                    <h4 className="font-semibold text-blue-800 text-sm mb-1">What Aura Detects:</h4>
                    <p className="text-xs text-blue-700">{sign.auraDetects}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {selectedSign && (
            <motion.div
              className="mt-8 bg-gray-50 border border-gray-200 rounded-lg p-6 max-w-4xl mx-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex items-start mb-4">
                <AlertTriangle className="h-8 w-8 text-orange-600 mr-4 mt-1" />
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-gray-900 mb-3">
                    Digital Indicators for {selectedSign}
                  </h3>
                  <div className="grid md:grid-cols-2 gap-4">
                    {depressionSigns
                      .find(sign => sign.sign === selectedSign)
                      ?.digitalIndicators.map((indicator, i) => (
                      <div key={i} className="flex items-start">
                        <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-3 flex-shrink-0" />
                        <span className="text-sm text-gray-700">{indicator}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="text-center">
                <button
                  onClick={() => handleStartTrial('warning_sign_details')}
                  className="bg-red-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-red-700 transition-colors"
                >
                  Monitor These Patterns with AI
                </button>
              </div>
            </motion.div>
          )}
        </section>

        {/* Why Traditional Methods Fail */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Why 73% of Teen Depression Goes Undetected
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Traditional detection methods miss the subtle digital behavior changes that precede clinical depression.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <motion.div
              className="bg-red-50 rounded-lg p-6 border border-red-200"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <h3 className="text-xl font-bold text-red-800 mb-4">Traditional Approach Limitations</h3>
              <ul className="space-y-3 text-sm text-red-700">
                <li className="flex items-start">
                  <X className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>Relies on teen self-reporting (teens rarely volunteer depression symptoms)</span>
                </li>
                <li className="flex items-start">
                  <X className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>Parent observation limited to visible behaviors at home</span>
                </li>
                <li className="flex items-start">
                  <X className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>Clinical screening happens only during scheduled appointments</span>
                </li>
                <li className="flex items-start">
                  <X className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>Symptoms must reach clinical threshold before detection</span>
                </li>
                <li className="flex items-start">
                  <X className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>No insight into peer interactions or digital social health</span>
                </li>
              </ul>
            </motion.div>

            <motion.div
              className="bg-blue-50 rounded-lg p-6 border border-blue-200"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <h3 className="text-xl font-bold text-blue-800 mb-4">AI Behavioral Monitoring Advantages</h3>
              <ul className="space-y-3 text-sm text-blue-700">
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>24/7 monitoring of actual behavior patterns, not just reports</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>Detects subtle changes 2-3 weeks before clinical symptoms</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>Analyzes communication patterns invisible to human observation</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>Tracks social engagement and peer relationship changes</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                  <span>Provides early intervention opportunities before crisis develops</span>
                </li>
              </ul>
            </motion.div>
          </div>
        </section>

        {/* Success Stories */}
        <section className="bg-gray-50 rounded-2xl p-8 mb-16">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-8">
            Parents Who Caught Depression Early
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="flex items-start mb-4">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <Heart className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">Carol M., Mother of 15-year-old</h4>
                  <p className="text-sm text-gray-600">Detected early depression patterns</p>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic">
                "Aura Balance showed me that Emma's messages were becoming increasingly negative and she was 
                withdrawing from her friend groups online. I never would have noticed these patterns myself. 
                We started family conversations and got counseling support 3 weeks before her pediatrician 
                would have detected anything."
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="flex items-start mb-4">
                <div className="bg-blue-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <Brain className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">Tom R., Father of 17-year-old</h4>
                  <p className="text-sm text-gray-600">Prevented severe depression episode</p>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic">
                "The AI detected that Jake's sleep patterns were severely disrupted and his language was becoming 
                hopeless. His mood seemed fine to us, but the data showed otherwise. Early intervention prevented 
                what our therapist said would have become a major depressive episode requiring medication."
              </p>
            </div>
          </div>
        </section>

        {/* Final CTA */}
        <div className="text-center bg-gradient-to-r from-red-600 to-purple-600 text-white rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-4">Don't Wait Until It's Too Late</h2>
          <p className="text-xl mb-6 opacity-90">
            Every day you wait is another day depression patterns strengthen. Start AI monitoring 
            now and catch warning signs before they become crises.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-6">
            <button
              onClick={() => handleStartTrial('final_cta')}
              className="bg-white text-red-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-gray-100 transition-all duration-200 shadow-lg"
            >
              Start Depression Monitoring
            </button>
            <div className="text-sm opacity-80">
              14-day free trial • Detect signs 2-3 weeks early • Prevent crisis
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-4 text-sm opacity-80 max-w-2xl mx-auto">
            <div>🔍 Early detection saves lives</div>
            <div>📱 Monitor digital behavior 24/7</div>
            <div>👨‍👩‍👧‍👦 Strengthen family connections</div>
          </div>
        </div>
      </main>

      <TrialSignupForm
        isOpen={showTrialForm}
        onClose={() => setShowTrialForm(false)}
        variant="depression-prevention"
        contextData={{
          source: 'teen-depression-warning-signs',
          checklist_score: summary.total,
          high_risk_signs: summary.high,
          signs_reviewed: selectedSign ? 1 : 0,
        }}
      />
    </BehavioralHealthLayout>
  );
};

export default TeenDepressionWarningSignsPage;
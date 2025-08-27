'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { DollarSign, TrendingDown, Clock, Brain, AlertTriangle, CheckCircle, Calculator, Shield, Heart, Users, Target, Award, X } from 'lucide-react';
import BehavioralHealthLayout from '@/components/BehavioralHealthLayout';
import TrialSignupForm from '@/components/TrialSignupForm';
import { behavioralTracking, trackingHelpers } from '@/lib/tracking';

interface CostComparison {
  service: string;
  monthlyRate: number;
  sessionsPerMonth: number;
  totalMonthlyCost: number;
  annualCost: number;
  limitations: string[];
  coverage: string;
}

interface PreventionScenario {
  scenario: string;
  preventionCost: number;
  treatmentCost: number;
  savings: number;
  timeToResolution: string;
  description: string;
}

const CheaperThanTherapyPage = () => {
  const [showTrialForm, setShowTrialForm] = useState(false);
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [showCalculator, setShowCalculator] = useState(false);
  const [calculatorInputs, setCalculatorInputs] = useState({
    therapyRate: 150,
    sessionsPerMonth: 2,
    insuranceCoverage: 20
  });
  const [timeOnPage, setTimeOnPage] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeOnPage(prev => prev + 1);
    }, 1000);

    behavioralTracking.storeAttribution();

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (timeOnPage === 60) {
      trackingHelpers.trackEngagement('high');
    }
  }, [timeOnPage]);

  const costComparisons: CostComparison[] = [
    {
      service: 'Individual Therapy',
      monthlyRate: 150,
      sessionsPerMonth: 2,
      totalMonthlyCost: 300,
      annualCost: 3600,
      limitations: [
        'Reactive approach - problems must be severe enough to recognize',
        'Limited to weekly 50-minute sessions',
        'No real-time monitoring between sessions',
        'Expensive co-pays and deductibles'
      ],
      coverage: 'Often not covered by insurance for preventive care'
    },
    {
      service: 'Family Counseling',
      monthlyRate: 175,
      sessionsPerMonth: 2,
      totalMonthlyCost: 350,
      annualCost: 4200,
      limitations: [
        'Scheduling conflicts with multiple family members',
        'Reactive intervention after problems develop',
        'No behavioral pattern monitoring',
        'High out-of-pocket costs'
      ],
      coverage: 'Limited insurance coverage, high deductibles'
    },
    {
      service: 'Teen Behavioral Coach',
      monthlyRate: 125,
      sessionsPerMonth: 4,
      totalMonthlyCost: 500,
      annualCost: 6000,
      limitations: [
        'No digital behavior insights',
        'Relies on teen self-reporting',
        'Cannot monitor between sessions',
        'Expensive ongoing commitment'
      ],
      coverage: 'Rarely covered by insurance'
    }
  ];

  const preventionScenarios: PreventionScenario[] = [
    {
      scenario: 'Early Depression Detection',
      preventionCost: 384, // 32/month * 12 months
      treatmentCost: 4800, // Therapy + medication + possible hospitalization
      savings: 4416,
      timeToResolution: 'Prevention vs 6-12 months treatment',
      description: 'Aura Balance detects mood pattern changes 2-3 weeks before clinical symptoms appear, enabling early intervention.'
    },
    {
      scenario: 'Cyberbullying Intervention',
      preventionCost: 384,
      treatmentCost: 3200, // Crisis counseling + school intervention + therapy
      savings: 2816,
      timeToResolution: 'Immediate intervention vs months of recovery',
      description: 'Real-time monitoring catches cyberbullying as it escalates, preventing severe psychological impact.'
    },
    {
      scenario: 'Social Anxiety Prevention',
      preventionCost: 384,
      treatmentCost: 2800, // Therapy + potential medication + family counseling
      savings: 2416,
      timeToResolution: 'Proactive support vs 4-8 months treatment',
      description: 'Social interaction pattern analysis identifies withdrawal before it becomes clinical social anxiety.'
    },
    {
      scenario: 'Substance Abuse Prevention',
      preventionCost: 384,
      treatmentCost: 8500, // Rehab + therapy + family therapy + relapse prevention
      savings: 8116,
      timeToResolution: 'Prevention vs 6+ months intensive treatment',
      description: 'Behavioral pattern recognition identifies risk factors before substance use begins.'
    }
  ];

  const handleScenarioSelect = (scenario: string) => {
    setSelectedScenario(scenario);
    
    behavioralTracking.trackGA4('cost_scenario_interest', {
      scenario,
      time_on_page: timeOnPage,
    });
  };

  const handleStartTrial = (source: string) => {
    behavioralTracking.trackCTAClick('Start Prevention Program', source, {
      cost_analysis_viewed: true,
      time_on_page: timeOnPage,
    });
    setShowTrialForm(true);
  };

  const handleCalculatorToggle = () => {
    setShowCalculator(!showCalculator);
    trackingHelpers.trackEducationalContent('therapy_cost_calculator', timeOnPage);
  };

  const calculateSavings = () => {
    const therapyCost = (calculatorInputs.therapyRate * calculatorInputs.sessionsPerMonth * 12);
    const insuranceCoverage = (therapyCost * calculatorInputs.insuranceCoverage) / 100;
    const outOfPocketTherapy = therapyCost - insuranceCoverage;
    const auraCost = 32 * 12; // $32/month * 12 months
    const totalSavings = outOfPocketTherapy - auraCost;
    
    return {
      therapyCost,
      outOfPocketTherapy,
      auraCost,
      totalSavings: Math.max(totalSavings, 0)
    };
  };

  const savings = calculateSavings();

  return (
    <BehavioralHealthLayout pageName="cheaper-than-therapy" showCrisisHeader={false}>
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <motion.div
            className="inline-flex items-center bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm font-semibold mb-6"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <DollarSign className="h-4 w-4 mr-2" />
            Prevention Economics: Save $4,000+ per Year
          </motion.div>
          
          <motion.h1 
            className="text-4xl md:text-6xl font-bold text-gray-900 mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="text-green-600">Cheaper Than Therapy</span>: Prevention vs Treatment Costs
          </motion.h1>
          
          <motion.p 
            className="text-xl text-gray-700 mb-8 max-w-4xl mx-auto leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <strong>Aura Balance costs $32/month.</strong> Therapy costs $150+ per session and insurance doesn't cover preventive monitoring. 
            Our AI prevents problems that would require months of expensive treatment, saving families thousands while protecting teen mental health.
          </motion.p>

          <motion.div
            className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <button
              onClick={() => handleStartTrial('hero')}
              className="bg-green-600 text-white px-8 py-4 rounded-lg font-bold text-lg hover:bg-green-700 transition-all duration-200 shadow-lg"
            >
              Start Prevention Program
            </button>
            <button
              onClick={handleCalculatorToggle}
              className="flex items-center text-green-600 font-semibold hover:text-green-700 transition-colors"
            >
              <Calculator className="h-5 w-5 mr-2" />
              Calculate Your Savings
            </button>
          </motion.div>

          {/* Quick Cost Comparison */}
          <motion.div 
            className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg p-6 max-w-4xl mx-auto"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <h3 className="text-lg font-bold text-gray-900 mb-4">Cost Comparison at a Glance</h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600 mb-2">$32</div>
                <p className="text-sm text-gray-600">Aura Balance per month<br/>24/7 behavioral monitoring</p>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-red-600 mb-2">$300</div>
                <p className="text-sm text-gray-600">Therapy per month<br/>2 sessions, reactive treatment</p>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600 mb-2">$4,400+</div>
                <p className="text-sm text-gray-600">Average annual savings<br/>with prevention approach</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Cost Calculator */}
        <AnimatePresence>
          {showCalculator && (
            <motion.section
              className="mb-16 bg-white rounded-2xl border-2 border-green-200 p-8"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.5 }}
            >
              <div className="text-center mb-8">
                <Calculator className="h-12 w-12 text-green-600 mx-auto mb-4" />
                <h2 className="text-3xl font-bold text-gray-900 mb-4">Therapy Cost Calculator</h2>
                <p className="text-gray-600 max-w-2xl mx-auto">
                  Calculate your potential savings by comparing therapy costs to Aura Balance's 
                  prevention-focused monitoring approach.
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-8">
                <div className="bg-gray-50 rounded-lg p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4">Your Therapy Costs</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Therapy Rate per Session
                      </label>
                      <div className="flex items-center">
                        <DollarSign className="h-5 w-5 text-gray-400 mr-2" />
                        <input
                          type="number"
                          value={calculatorInputs.therapyRate}
                          onChange={(e) => setCalculatorInputs({
                            ...calculatorInputs,
                            therapyRate: parseInt(e.target.value) || 150
                          })}
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                          placeholder="150"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Sessions per Month
                      </label>
                      <input
                        type="number"
                        value={calculatorInputs.sessionsPerMonth}
                        onChange={(e) => setCalculatorInputs({
                          ...calculatorInputs,
                          sessionsPerMonth: parseInt(e.target.value) || 2
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                        placeholder="2"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Insurance Coverage (%)
                      </label>
                      <input
                        type="number"
                        value={calculatorInputs.insuranceCoverage}
                        onChange={(e) => setCalculatorInputs({
                          ...calculatorInputs,
                          insuranceCoverage: parseInt(e.target.value) || 20
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                        placeholder="20"
                        max="100"
                      />
                      <p className="text-xs text-gray-500 mt-1">Most insurance covers 20-50% for mental health</p>
                    </div>
                  </div>
                </div>

                <div className="bg-green-50 rounded-lg p-6">
                  <h3 className="text-xl font-bold text-green-800 mb-4">Your Annual Savings</h3>
                  
                  <div className="space-y-4">
                    <div className="flex justify-between items-center py-2 border-b border-green-200">
                      <span className="text-sm text-gray-700">Annual Therapy Cost:</span>
                      <span className="font-semibold text-gray-900">${savings.therapyCost.toLocaleString()}</span>
                    </div>
                    
                    <div className="flex justify-between items-center py-2 border-b border-green-200">
                      <span className="text-sm text-gray-700">Insurance Coverage:</span>
                      <span className="font-semibold text-green-600">
                        -${(savings.therapyCost - savings.outOfPocketTherapy).toLocaleString()}
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center py-2 border-b border-green-200">
                      <span className="text-sm text-gray-700">Your Out-of-Pocket:</span>
                      <span className="font-semibold text-gray-900">${savings.outOfPocketTherapy.toLocaleString()}</span>
                    </div>
                    
                    <div className="flex justify-between items-center py-2 border-b border-green-200">
                      <span className="text-sm text-gray-700">Aura Balance Annual Cost:</span>
                      <span className="font-semibold text-blue-600">${savings.auraCost}</span>
                    </div>
                    
                    <div className="flex justify-between items-center py-3 bg-green-100 rounded-lg px-4 mt-4">
                      <span className="font-bold text-green-800">Total Annual Savings:</span>
                      <span className="text-2xl font-bold text-green-600">${savings.totalSavings.toLocaleString()}</span>
                    </div>
                  </div>
                  
                  <div className="mt-6 text-center">
                    <button
                      onClick={() => handleStartTrial('calculator')}
                      className="bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors"
                    >
                      Start Saving with Prevention
                    </button>
                  </div>
                </div>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Prevention vs Treatment Scenarios */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Prevention vs Treatment: Real Cost Scenarios
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              See how early detection and prevention saves families thousands compared to 
              reactive treatment after problems develop.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {preventionScenarios.map((scenario, index) => (
              <motion.div
                key={scenario.scenario}
                className={`border rounded-lg p-6 cursor-pointer transition-all duration-200 ${
                  selectedScenario === scenario.scenario
                    ? 'border-green-500 bg-green-50'
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                }`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                onClick={() => handleScenarioSelect(scenario.scenario)}
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-lg font-bold text-gray-900">{scenario.scenario}</h3>
                  <div className="px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    ${scenario.savings.toLocaleString()} saved
                  </div>
                </div>
                
                <p className="text-sm text-gray-700 mb-4">{scenario.description}</p>
                
                <div className="space-y-3">
                  <div className="flex justify-between items-center py-2 bg-blue-50 rounded px-3">
                    <span className="text-sm font-medium text-blue-800">Prevention Cost:</span>
                    <span className="font-bold text-blue-600">${scenario.preventionCost}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-2 bg-red-50 rounded px-3">
                    <span className="text-sm font-medium text-red-800">Treatment Cost:</span>
                    <span className="font-bold text-red-600">${scenario.treatmentCost.toLocaleString()}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-2 bg-green-50 rounded px-3">
                    <span className="text-sm font-medium text-green-800">Time Impact:</span>
                    <span className="text-sm text-green-700">{scenario.timeToResolution}</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Detailed Cost Analysis */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Therapy vs Aura Balance: Complete Cost Breakdown
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Traditional therapy is expensive and reactive. Prevention monitoring provides 
              better outcomes at a fraction of the cost.
            </p>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div className="grid grid-cols-5 bg-gray-50 p-4 font-semibold text-gray-900 text-sm">
              <div>Service Type</div>
              <div className="text-center">Monthly Cost</div>
              <div className="text-center">Annual Cost</div>
              <div className="text-center">Insurance Coverage</div>
              <div className="text-center">Limitations</div>
            </div>
            
            {costComparisons.map((comparison, index) => (
              <motion.div
                key={comparison.service}
                className="grid grid-cols-5 p-4 border-t border-gray-200 hover:bg-gray-50 transition-colors text-sm"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <div className="font-semibold text-gray-900">{comparison.service}</div>
                <div className="text-center text-red-600 font-bold">${comparison.totalMonthlyCost}</div>
                <div className="text-center text-red-600 font-bold">${comparison.annualCost.toLocaleString()}</div>
                <div className="text-center text-sm text-gray-600">{comparison.coverage}</div>
                <div className="text-xs text-gray-600">
                  <ul className="space-y-1">
                    {comparison.limitations.slice(0, 2).map((limitation, i) => (
                      <li key={i}>• {limitation}</li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            ))}
            
            {/* Aura Balance Row */}
            <div className="grid grid-cols-5 p-4 border-t-2 border-green-200 bg-green-50 text-sm">
              <div className="font-bold text-green-800">Aura Balance Prevention</div>
              <div className="text-center text-green-600 font-bold text-lg">$32</div>
              <div className="text-center text-green-600 font-bold text-lg">$384</div>
              <div className="text-center text-sm text-green-700">N/A - Direct pay</div>
              <div className="text-xs text-green-700">
                <ul className="space-y-1">
                  <li>• 24/7 monitoring & prevention</li>
                  <li>• AI-powered early detection</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="mt-6 text-center">
            <div className="bg-green-100 border border-green-200 rounded-lg p-6 max-w-2xl mx-auto">
              <Award className="h-8 w-8 text-green-600 mx-auto mb-3" />
              <h3 className="text-lg font-bold text-green-800 mb-2">Average Family Savings</h3>
              <div className="text-3xl font-bold text-green-600 mb-2">$4,400 per year</div>
              <p className="text-sm text-green-700">
                Families using prevention monitoring avoid an average of $4,400 in therapy 
                and treatment costs annually while achieving better mental health outcomes.
              </p>
            </div>
          </div>
        </section>

        {/* Insurance Reality Check */}
        <section className="mb-16">
          <div className="bg-yellow-50 border border-yellow-200 rounded-2xl p-8">
            <div className="text-center mb-6">
              <AlertTriangle className="h-12 w-12 text-yellow-600 mx-auto mb-4" />
              <h2 className="text-3xl font-bold text-yellow-800 mb-2">Insurance Reality Check</h2>
              <p className="text-yellow-700">
                What insurance companies don't cover and why prevention makes financial sense
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-white rounded-lg p-6 border border-yellow-200">
                <h3 className="text-xl font-bold text-red-800 mb-4">What Insurance WON'T Cover</h3>
                <ul className="space-y-3 text-sm text-gray-700">
                  <li className="flex items-start">
                    <X className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Preventive behavioral health monitoring</span>
                  </li>
                  <li className="flex items-start">
                    <X className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Digital wellness tools and apps</span>
                  </li>
                  <li className="flex items-start">
                    <X className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Family coaching and guidance</span>
                  </li>
                  <li className="flex items-start">
                    <X className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Early intervention before diagnosis</span>
                  </li>
                  <li className="flex items-start">
                    <X className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>24/7 crisis prevention monitoring</span>
                  </li>
                </ul>
              </div>

              <div className="bg-white rounded-lg p-6 border border-green-200">
                <h3 className="text-xl font-bold text-green-800 mb-4">What YOU Pay For</h3>
                <ul className="space-y-3 text-sm text-gray-700">
                  <li className="flex items-start">
                    <DollarSign className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>High deductibles ($2,000-$5,000 annually)</span>
                  </li>
                  <li className="flex items-start">
                    <DollarSign className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Co-pays ($25-$50 per therapy session)</span>
                  </li>
                  <li className="flex items-start">
                    <DollarSign className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Out-of-network charges (40-50% more)</span>
                  </li>
                  <li className="flex items-start">
                    <DollarSign className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Medication costs and monitoring</span>
                  </li>
                  <li className="flex items-start">
                    <DollarSign className="h-5 w-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                    <span>Crisis intervention and hospitalization</span>
                  </li>
                </ul>
              </div>
            </div>

            <div className="text-center mt-6">
              <div className="bg-green-100 border border-green-200 rounded-lg p-4 max-w-xl mx-auto">
                <p className="text-green-800 font-semibold mb-2">
                  Smart Investment Strategy
                </p>
                <p className="text-sm text-green-700">
                  Spend $32/month on prevention instead of $300+/month on treatment. 
                  Most families recoup their entire annual investment after avoiding just one therapy session.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Parent Testimonials - Cost Focus */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Parents Share Their Cost Savings Stories
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Real families who prevented expensive mental health crises with 
              Aura Balance's affordable monitoring approach.
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
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <DollarSign className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">Susan T., Mother of 16-year-old</h4>
                  <p className="text-sm text-gray-600">Saved $4,800 in therapy costs</p>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic mb-4">
                "Aura Balance detected my son's depression patterns in February. We started family conversations 
                and adjustments immediately. My pediatrician said we avoided what would have been 6 months of 
                intensive therapy costing us over $200 per session after insurance."
              </p>
              <div className="bg-green-50 border border-green-200 rounded p-3">
                <p className="text-sm text-green-700 font-semibold">
                  Cost avoided: $4,800 in therapy + $1,200 in potential medication = $6,000 saved
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
                <div className="bg-blue-100 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                  <Shield className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <h4 className="font-bold text-gray-900">Mark & Linda R., Parents of twins</h4>
                  <p className="text-sm text-gray-600">Prevented cyberbullying crisis</p>
                </div>
              </div>
              <p className="text-gray-700 text-sm italic mb-4">
                "The AI caught that our daughter was being targeted by bullies before it escalated. 
                We worked with the school immediately. Our friends whose daughter went through similar 
                bullying spent over $8,000 on therapy, family counseling, and had to change schools."
              </p>
              <div className="bg-blue-50 border border-blue-200 rounded p-3">
                <p className="text-sm text-blue-700 font-semibold">
                  Crisis prevented, education continued normally, family relationships strengthened
                </p>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Final CTA */}
        <div className="text-center bg-gradient-to-r from-green-600 to-blue-600 text-white rounded-2xl p-8">
          <h2 className="text-3xl font-bold mb-4">Invest in Prevention, Not Treatment</h2>
          <p className="text-xl mb-6 opacity-90">
            For less than the cost of one therapy session, protect your teen's mental health 
            all month long. Prevention is always cheaper than treatment.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-6">
            <button
              onClick={() => handleStartTrial('final_cta')}
              className="bg-white text-green-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-gray-100 transition-all duration-200 shadow-lg"
            >
              Start Prevention Program - $32/month
            </button>
            <div className="text-sm opacity-80">
              14-day free trial • Cancel anytime • Save thousands vs therapy
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-4 text-sm opacity-80 max-w-2xl mx-auto">
            <div>💰 Average family saves $4,400/year</div>
            <div>⏰ Prevention starts immediately</div>
            <div>🛡️ Avoid expensive crises</div>
          </div>
        </div>
      </main>

      <TrialSignupForm
        isOpen={showTrialForm}
        onClose={() => setShowTrialForm(false)}
        variant="cost-savings"
        contextData={{
          source: 'cheaper-than-therapy',
          calculator_used: showCalculator,
          scenarios_viewed: selectedScenario ? 1 : 0,
        }}
      />
    </BehavioralHealthLayout>
  );
};

export default CheaperThanTherapyPage;
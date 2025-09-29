"use client"
import React, { useState } from 'react'
import { TrendingUp, TrendingDown, AlertCircle, CheckCircle, DollarSign, Target, Users, Palette, Share2 } from 'lucide-react'

interface ScenarioResult {
  scenario: {
    type: string
    changes: any
    timeHorizon: number
    baseline: any
    projection: any
    confidence: any
    recommendations: any[]
  }
}

export default function ScenarioModeler() {
  const [activeScenario, setActiveScenario] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ScenarioResult | null>(null)
  const [parameters, setParameters] = useState({
    budgetChange: 0,
    biddingStrategy: 'MAXIMIZE_CONVERSIONS',
    audienceSegments: [] as string[],
    creativeVariants: 3,
    channelMix: {
      google: 40,
      facebook: 30,
      direct: 20,
      organic: 10
    },
    timeHorizon: 30
  })

  const runScenario = async (type: string) => {
    setLoading(true)
    setActiveScenario(type)

    try {
      const response = await fetch('/api/scenarios/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type,
          changes: {
            budgetChange: parameters.budgetChange,
            biddingStrategy: parameters.biddingStrategy,
            audienceSegments: parameters.audienceSegments,
            creativeVariants: parameters.creativeVariants,
            channelMix: parameters.channelMix
          },
          timeHorizon: parameters.timeHorizon
        })
      })

      const data = await response.json()
      if (data.success) {
        setResult(data)
      } else {
        throw new Error(data.error)
      }
    } catch (error: any) {
      console.error('Scenario modeling failed:', error)
      alert(`Error: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  const scenarioButtons = [
    {
      id: 'budget',
      label: 'Budget Optimization',
      icon: <DollarSign className="w-5 h-5" />,
      color: 'blue',
      description: 'Model impact of budget changes'
    },
    {
      id: 'bidding',
      label: 'Bidding Strategy',
      icon: <Target className="w-5 h-5" />,
      color: 'purple',
      description: 'Compare bidding strategies'
    },
    {
      id: 'audience',
      label: 'Audience Targeting',
      icon: <Users className="w-5 h-5" />,
      color: 'green',
      description: 'Optimize audience segments'
    },
    {
      id: 'creative',
      label: 'Creative Testing',
      icon: <Palette className="w-5 h-5" />,
      color: 'pink',
      description: 'Model creative variant impact'
    },
    {
      id: 'channel',
      label: 'Channel Mix',
      icon: <Share2 className="w-5 h-5" />,
      color: 'orange',
      description: 'Optimize channel allocation'
    }
  ]

  const formatValue = (value: number, prefix = '', suffix = '') => {
    if (typeof value !== 'number') return '-'
    return `${prefix}${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}${suffix}`
  }

  return (
    <div className="space-y-6">
      {/* Scenario Parameters */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Scenario Parameters</h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Budget Change (%)
            </label>
            <input
              type="number"
              value={parameters.budgetChange}
              onChange={(e) => setParameters({ ...parameters, budgetChange: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              placeholder="e.g., 20 for 20% increase"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Bidding Strategy
            </label>
            <select
              value={parameters.biddingStrategy}
              onChange={(e) => setParameters({ ...parameters, biddingStrategy: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="MAXIMIZE_CONVERSIONS">Maximize Conversions</option>
              <option value="TARGET_CPA">Target CPA</option>
              <option value="TARGET_ROAS">Target ROAS</option>
              <option value="MAXIMIZE_CONVERSION_VALUE">Maximize Conversion Value</option>
              <option value="ENHANCED_CPC">Enhanced CPC</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Time Horizon (days)
            </label>
            <input
              type="number"
              value={parameters.timeHorizon}
              onChange={(e) => setParameters({ ...parameters, timeHorizon: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              min="7"
              max="180"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Creative Variants
            </label>
            <input
              type="number"
              value={parameters.creativeVariants}
              onChange={(e) => setParameters({ ...parameters, creativeVariants: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              min="1"
              max="10"
            />
          </div>

          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Audience Segments (comma-separated)
            </label>
            <input
              type="text"
              value={parameters.audienceSegments.join(', ')}
              onChange={(e) => setParameters({
                ...parameters,
                audienceSegments: e.target.value.split(',').map(s => s.trim()).filter(s => s)
              })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              placeholder="e.g., high_value_users, cart_abandoners, new_visitors"
            />
          </div>
        </div>

        {/* Channel Mix */}
        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Channel Mix Allocation (%)
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(parameters.channelMix).map(([channel, weight]) => (
              <div key={channel}>
                <label className="text-xs text-gray-600 capitalize">{channel}</label>
                <input
                  type="number"
                  value={weight}
                  onChange={(e) => setParameters({
                    ...parameters,
                    channelMix: { ...parameters.channelMix, [channel]: Number(e.target.value) }
                  })}
                  className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                  min="0"
                  max="100"
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Scenario Buttons */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Run Scenario Models</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {scenarioButtons.map((scenario) => (
            <button
              key={scenario.id}
              onClick={() => runScenario(scenario.id)}
              disabled={loading && activeScenario === scenario.id}
              className={`
                flex flex-col items-center justify-center p-4 rounded-lg border-2 transition-all
                ${loading && activeScenario === scenario.id
                  ? 'border-gray-300 bg-gray-50 cursor-not-allowed'
                  : `border-${scenario.color}-200 hover:border-${scenario.color}-400 hover:bg-${scenario.color}-50`
                }
              `}
            >
              <div className={`text-${scenario.color}-600 mb-2`}>
                {scenario.icon}
              </div>
              <span className="font-medium text-sm">{scenario.label}</span>
              <span className="text-xs text-gray-600 text-center mt-1">
                {scenario.description}
              </span>
              {loading && activeScenario === scenario.id && (
                <span className="text-xs text-blue-600 mt-2">Modeling...</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Results Display */}
      {result && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Scenario Results</h3>
            <button
              onClick={() => setResult(null)}
              className="text-gray-500 hover:text-gray-700"
            >
              ✕
            </button>
          </div>

          {/* Key Projections */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {result.scenario.projection.projected_conversions && (
              <div className="bg-blue-50 rounded-lg p-4">
                <p className="text-sm text-gray-600">Projected Conversions</p>
                <p className="text-2xl font-bold text-blue-600">
                  {formatValue(result.scenario.projection.projected_conversions)}
                </p>
                {result.scenario.confidence.projected_conversions && (
                  <p className="text-xs text-gray-500">
                    CI: [{formatValue(result.scenario.confidence.projected_conversions.lower)} -
                    {formatValue(result.scenario.confidence.projected_conversions.upper)}]
                  </p>
                )}
              </div>
            )}

            {result.scenario.projection.projected_revenue && (
              <div className="bg-green-50 rounded-lg p-4">
                <p className="text-sm text-gray-600">Projected Revenue</p>
                <p className="text-2xl font-bold text-green-600">
                  {formatValue(result.scenario.projection.projected_revenue, '$')}
                </p>
                {result.scenario.confidence.projected_revenue && (
                  <p className="text-xs text-gray-500">
                    CI: [{formatValue(result.scenario.confidence.projected_revenue.lower, '$')} -
                    {formatValue(result.scenario.confidence.projected_revenue.upper, '$')}]
                  </p>
                )}
              </div>
            )}

            {result.scenario.projection.projected_roas && (
              <div className="bg-purple-50 rounded-lg p-4">
                <p className="text-sm text-gray-600">Projected ROAS</p>
                <p className="text-2xl font-bold text-purple-600">
                  {formatValue(result.scenario.projection.projected_roas, '', 'x')}
                </p>
              </div>
            )}

            {result.scenario.projection.projected_cac && (
              <div className="bg-orange-50 rounded-lg p-4">
                <p className="text-sm text-gray-600">Projected CAC</p>
                <p className="text-2xl font-bold text-orange-600">
                  {formatValue(result.scenario.projection.projected_cac, '$')}
                </p>
              </div>
            )}
          </div>

          {/* Additional Metrics */}
          {result.scenario.projection.efficiency_score && (
            <div className="mb-4">
              <p className="text-sm font-medium text-gray-700">Efficiency Score</p>
              <div className="w-full bg-gray-200 rounded-full h-2.5 mt-1">
                <div
                  className="bg-blue-600 h-2.5 rounded-full"
                  style={{ width: `${result.scenario.projection.efficiency_score * 100}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {(result.scenario.projection.efficiency_score * 100).toFixed(1)}% efficient
              </p>
            </div>
          )}

          {/* Recommendations */}
          {result.scenario.recommendations && result.scenario.recommendations.length > 0 && (
            <div className="mt-6">
              <h4 className="font-medium mb-3">Recommendations</h4>
              <div className="space-y-2">
                {result.scenario.recommendations.map((rec: any, idx: number) => (
                  <div
                    key={idx}
                    className={`flex items-start gap-2 p-3 rounded-lg ${
                      rec.type === 'positive' ? 'bg-green-50' :
                      rec.type === 'warning' ? 'bg-yellow-50' : 'bg-gray-50'
                    }`}
                  >
                    {rec.type === 'positive' ? (
                      <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
                    ) : rec.type === 'warning' ? (
                      <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
                    ) : (
                      <TrendingUp className="w-5 h-5 text-gray-600 mt-0.5" />
                    )}
                    <div>
                      <p className="text-sm font-medium">{rec.message}</p>
                      {rec.priority && (
                        <span className={`text-xs px-2 py-0.5 rounded-full inline-block mt-1 ${
                          rec.priority === 'high' ? 'bg-red-100 text-red-700' :
                          rec.priority === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-gray-100 text-gray-700'
                        }`}>
                          {rec.priority} priority
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Raw Data Toggle */}
          <details className="mt-6">
            <summary className="cursor-pointer text-sm text-blue-600 hover:text-blue-800">
              View Raw Projection Data
            </summary>
            <pre className="mt-2 p-3 bg-gray-50 rounded text-xs overflow-auto">
              {JSON.stringify(result.scenario.projection, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  )
}
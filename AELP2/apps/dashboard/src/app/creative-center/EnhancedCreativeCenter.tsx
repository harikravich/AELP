"use client"
import React, { useState } from 'react'

interface GeneratedContent {
  type: string
  generated: any
  timestamp: string
}

interface MultiModalHubProps {
  onGenerate: (type: string, data: any) => void
}

const MultiModalHub: React.FC<MultiModalHubProps> = ({ onGenerate }) => {
  const [loading, setLoading] = useState<string | null>(null)
  const [generatedContent, setGeneratedContent] = useState<GeneratedContent | null>(null)
  const [formData, setFormData] = useState({
    product: '',
    campaign: '',
    audience: '',
    tone: 'professional',
    prompt: ''
  })

  const generateCreative = async (type: string) => {
    setLoading(type)
    try {
      const response = await fetch('/api/creative/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type,
          ...formData
        })
      })

      const data = await response.json()
      if (data.success) {
        setGeneratedContent(data)
        onGenerate(type, data)
      } else {
        throw new Error(data.error)
      }
    } catch (error: any) {
      console.error(`Failed to generate ${type}:`, error)
      alert(`Error: ${error.message}`)
    } finally {
      setLoading(null)
    }
  }

  const imageButtons = [
    { id: 'product-screenshot', label: 'Product Screenshots', icon: '📸' },
    { id: 'social-proof', label: 'Social Proof Graphics', icon: '⭐' },
    { id: 'data-viz', label: 'Chart/Data Visualizations', icon: '📊' },
    { id: 'image-assets', label: 'Generate Image Assets', icon: '🎨' }
  ]

  const videoButtons = [
    { id: 'demo-video', label: 'Demo Videos', icon: '🎬' },
    { id: 'testimonial', label: 'Customer Testimonials', icon: '💬' },
    { id: 'explainer', label: 'Explainer Animations', icon: '🎭' },
    { id: 'video-script-full', label: 'Create Video Script', icon: '📝' }
  ]

  const copyButtons = [
    { id: 'headlines', label: 'Headlines (30 chars)', icon: '📰' },
    { id: 'descriptions', label: 'Descriptions (90 chars)', icon: '📄' },
    { id: 'cta-extensions', label: 'CTAs & Extensions', icon: '🎯' },
    { id: 'copy-set', label: 'Generate Copy Set', icon: '📚' }
  ]

  return (
    <div className="space-y-6">
      {/* Input Form */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Creative Brief</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Product/Service</label>
            <input
              type="text"
              value={formData.product}
              onChange={(e) => setFormData({...formData, product: e.target.value})}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
              placeholder="e.g., Project Management Software"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Campaign</label>
            <input
              type="text"
              value={formData.campaign}
              onChange={(e) => setFormData({...formData, campaign: e.target.value})}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
              placeholder="e.g., Q1 2025 Launch"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Target Audience</label>
            <input
              type="text"
              value={formData.audience}
              onChange={(e) => setFormData({...formData, audience: e.target.value})}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
              placeholder="e.g., SMB owners, 25-45"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Tone</label>
            <select
              value={formData.tone}
              onChange={(e) => setFormData({...formData, tone: e.target.value})}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
            >
              <option value="professional">Professional</option>
              <option value="casual">Casual</option>
              <option value="playful">Playful</option>
              <option value="urgent">Urgent</option>
              <option value="empathetic">Empathetic</option>
            </select>
          </div>
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-700">Additional Context</label>
          <textarea
            value={formData.prompt}
            onChange={(e) => setFormData({...formData, prompt: e.target.value})}
            rows={3}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
            placeholder="Any specific requirements, features to highlight, or creative direction..."
          />
        </div>
      </div>

      {/* Image Generation Section */}
      <div>
        <h3 className="text-lg font-semibold mb-3">🖼️ Image Generation</h3>
        <div className="grid grid-cols-4 gap-3">
          {imageButtons.map(button => (
            <button
              key={button.id}
              onClick={() => generateCreative(button.id)}
              disabled={loading === button.id}
              className="flex flex-col items-center justify-center p-4 bg-white border-2 border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all duration-200 disabled:opacity-50"
            >
              <span className="text-2xl mb-2">{button.icon}</span>
              <span className="text-sm font-medium text-center">{button.label}</span>
              {loading === button.id && (
                <span className="text-xs text-blue-600 mt-1">Generating...</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Video Generation Section */}
      <div>
        <h3 className="text-lg font-semibold mb-3">🎥 Video Generation</h3>
        <div className="grid grid-cols-4 gap-3">
          {videoButtons.map(button => (
            <button
              key={button.id}
              onClick={() => generateCreative(button.id)}
              disabled={loading === button.id}
              className="flex flex-col items-center justify-center p-4 bg-white border-2 border-gray-200 rounded-lg hover:border-purple-500 hover:bg-purple-50 transition-all duration-200 disabled:opacity-50"
            >
              <span className="text-2xl mb-2">{button.icon}</span>
              <span className="text-sm font-medium text-center">{button.label}</span>
              {loading === button.id && (
                <span className="text-xs text-purple-600 mt-1">Generating...</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Copy Generation Section */}
      <div>
        <h3 className="text-lg font-semibold mb-3">✍️ Copy Generation</h3>
        <div className="grid grid-cols-4 gap-3">
          {copyButtons.map(button => (
            <button
              key={button.id}
              onClick={() => generateCreative(button.id)}
              disabled={loading === button.id}
              className="flex flex-col items-center justify-center p-4 bg-white border-2 border-gray-200 rounded-lg hover:border-green-500 hover:bg-green-50 transition-all duration-200 disabled:opacity-50"
            >
              <span className="text-2xl mb-2">{button.icon}</span>
              <span className="text-sm font-medium text-center">{button.label}</span>
              {loading === button.id && (
                <span className="text-xs text-green-600 mt-1">Generating...</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Generated Content Display */}
      {generatedContent && (
        <div className="mt-6 bg-white border-2 border-blue-200 rounded-lg p-4">
          <div className="flex justify-between items-start mb-3">
            <h3 className="text-lg font-semibold">Generated Content</h3>
            <button
              onClick={() => setGeneratedContent(null)}
              className="text-gray-500 hover:text-gray-700"
            >
              ✕
            </button>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600 mb-2">Type: {generatedContent.type}</p>
            <pre className="text-xs overflow-auto max-h-96 bg-white p-3 rounded border">
              {JSON.stringify(generatedContent.generated, null, 2)}
            </pre>
            <div className="mt-3 flex gap-2">
              <button
                onClick={() => {
                  navigator.clipboard.writeText(JSON.stringify(generatedContent.generated, null, 2))
                  alert('Copied to clipboard!')
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Copy JSON
              </button>
              <button
                onClick={() => {
                  // Queue for processing
                  fetch('/api/creative/queue', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(generatedContent)
                  })
                  alert('Queued for processing!')
                }}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
              >
                Queue for Processing
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default MultiModalHub
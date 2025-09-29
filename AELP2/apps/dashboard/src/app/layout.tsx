import './globals.css'
import React from 'react'
import { DatasetSwitcher } from '../components/DatasetSwitcher'
import { KpiSourceSwitcher } from '../components/KpiSourceSwitcher'
import Nav from '../components/Nav'
import Providers from '../components/Providers'
import { Inter } from 'next/font/google'
import { AppToaster } from '../components/ui/toaster'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })

export const metadata = {
  title: 'AELP2 | AI-Powered Marketing Intelligence',
  description: 'Enterprise-grade marketing automation platform with AI-driven insights',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans">
        <Providers>
        {process.env.PILOT_MODE === '1' && (
          <div className="w-full" style={{background:'#0ea5e9'}}>
            <div className="max-w-7xl mx-auto px-6 py-2 text-white text-xs flex items-center justify-between">
              <div>
                Pilot Mode — Ads in personal account ({process.env.GOOGLE_ADS_CUSTOMER_ID || 'unset'}); reading Aura GA4 ({process.env.GA4_PROPERTY_ID || 'unset'}) with sandbox filters. All publishes PAUSED.
              </div>
              <div className="opacity-90">Writes to prod off; HITL approvals required.</div>
            </div>
          </div>
        )}
        <div className="min-h-screen">
          {/* Premium Header */}
          <header className="sticky top-0 z-50 glass-card rounded-none border-t-0 border-x-0">
            <div className="max-w-7xl mx-auto px-6">
              <div className="flex items-center justify-between py-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/25">
                    <span className="text-white font-bold text-lg">A2</span>
                  </div>
                  <div>
                    <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                      AELP2 Dashboard
                    </h1>
                    <p className="text-xs text-white/50">Marketing Intelligence Platform</p>
                  </div>
                </div>
                <Nav />
                <div className="flex items-center gap-4">
                  <KpiSourceSwitcher />
                  <DatasetSwitcher />
                </div>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main className="max-w-7xl mx-auto px-6 py-8">
            <div className="animate-fade-in">
              {children}
            </div>
          </main>

          {/* Premium Footer */}
          <footer className="mt-20 pb-8">
            <div className="max-w-7xl mx-auto px-6">
              <div className="glass-card rounded-2xl p-6 text-center">
                <p className="text-sm text-white/60 mb-2">
                  Powered by advanced AI and machine learning algorithms
                </p>
                <div className="flex items-center justify-center gap-6 text-xs">
                  <a href="/canvas" className="text-indigo-400 hover:text-indigo-300 transition-colors">
                    Flow Canvas
                  </a>
                  <span className="text-white/20">•</span>
                  <a href="/ops/chat" className="text-indigo-400 hover:text-indigo-300 transition-colors">
                    AI Assistant
                  </a>
                  <span className="text-white/20">•</span>
                  <a href="/control" className="text-indigo-400 hover:text-indigo-300 transition-colors">
                    System Control
                  </a>
                </div>
              </div>
            </div>
          </footer>
        </div>
        <AppToaster />
        </Providers>
      </body>
    </html>
  )
}

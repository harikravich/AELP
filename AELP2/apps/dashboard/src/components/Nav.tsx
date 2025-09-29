"use client"
"use client"
import React from 'react'
import { usePathname } from 'next/navigation'
import { 
  BarChart3, 
  FlaskConical, 
  DollarSign, 
  Palette, 
  GraduationCap, 
  Route, 
  Gavel, 
  Settings2, 
  MessageSquareCode 
} from 'lucide-react'

const links = [
  { href: '/', label: 'Overview', icon: BarChart3 },
  { href: '/qs', label: 'QS/IS', icon: Settings2 },
  { href: '/ops/chat', label: 'Chat', icon: MessageSquareCode },
  { href: '/creative-center', label: 'Creatives', icon: Palette },
  { href: '/landing', label: 'Pages', icon: Route },
  { href: '/spend-planner', label: 'Spend', icon: DollarSign },
  { href: '/channels', label: 'Channels', icon: FlaskConical },
  { href: '/auctions-monitor', label: 'Auctions', icon: Gavel },
  { href: '/rl-insights', label: 'RL Lab', icon: GraduationCap },
  { href: '/approvals', label: 'Approvals', icon: Settings2 },
]

import { useSession, signIn, signOut } from 'next-auth/react'

export default function Nav(){
  const path = usePathname() || ''
  const { data: session, status } = useSession()
  return (
    <nav className="flex items-center gap-1 flex-1">
      {links.map(l=>{
        const active = path.startsWith(l.href)
        const Icon = l.icon
        return (
          <a 
            key={l.href}
            href={l.href}
            className={`
              group relative inline-flex items-center gap-2 px-4 py-2.5 rounded-xl
              font-medium text-sm transition-all duration-300
              ${active 
                ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg shadow-indigo-500/25' 
                : 'text-white/70 hover:text-white hover:bg-white/10'
              }
            `}
          >
            <Icon className={`w-4 h-4 transition-transform duration-300 ${active ? 'scale-110' : 'group-hover:scale-110'}`} />
            <span className="hidden lg:inline">{l.label}</span>
            {active && (
              <span className="absolute inset-x-0 -bottom-px h-px bg-gradient-to-r from-transparent via-white to-transparent opacity-50" />
            )}
          </a>
        )
      })}
      <div className="ml-auto pl-2">
        {status === 'authenticated' ? (
          <button onClick={()=>signOut()} className="text-white/70 hover:text-white text-sm">Sign out</button>
        ) : (
          <button onClick={()=>signIn()} className="text-white/70 hover:text-white text-sm">Sign in</button>
        )}
      </div>
    </nav>
  )
}

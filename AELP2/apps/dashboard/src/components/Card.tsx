import React from 'react'

type Props = {
  title?: string
  subtitle?: string
  footer?: React.ReactNode
  actions?: React.ReactNode
  className?: string
  children?: React.ReactNode
}

export default function Card({ title, subtitle, footer, actions, className = '', children }: Props) {
  return (
    <div className={`card p-4 ${className}`}>
      {(title || actions) && (
        <div className="flex items-center justify-between mb-2">
          <div>
            {title && <h2 className="text-lg font-medium leading-tight">{title}</h2>}
            {subtitle && <div className="text-xs text-slate-500 mt-0.5">{subtitle}</div>}
          </div>
          {actions && <div className="ml-2">{actions}</div>}
        </div>
      )}
      {children}
      {footer && <div className="mt-3 text-xs text-slate-500">{footer}</div>}
    </div>
  )
}


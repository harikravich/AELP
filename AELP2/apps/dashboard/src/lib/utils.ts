import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: any[]) {
  return twMerge(clsx(inputs))
}

export function toPlain(val: any): any {
  if (val == null) return val
  // BigQuery timestamp/datetime often comes as object with { value }
  if (typeof val === 'object') {
    if ('value' in val && typeof (val as any).value !== 'object') return (val as any).value
    if (val instanceof Date) return val.toISOString()
  }
  return val
}

export function fmtWhen(val: any): string {
  const raw = toPlain(val)
  try {
    const s = typeof raw === 'string' ? raw : String(raw)
    // Try to format ISO-like strings; else truncate
    if (/^\d{4}-\d{2}-\d{2}T\d{2}:/i.test(s)) return s.replace('T', ' ').slice(0, 16)
    if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s
    // Fallback for Date strings
    return s.slice(0, 16)
  } catch {
    return String(raw)
  }
}

export function fmtUSD(n: any, digits: number = 0): string {
  const v = Number(n || 0)
  return v.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: digits, minimumFractionDigits: digits })
}

export function fmtInt(n: any): string {
  const v = Number(n || 0)
  return v.toLocaleString('en-US', { maximumFractionDigits: 0 })
}

export function fmtPct(n: any, digits: number = 1): string {
  const v = Number(n || 0)
  return (v * 100).toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits }) + '%'
}

export function fmtFloat(n: any, digits: number = 2): string {
  const v = Number(n || 0)
  return v.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

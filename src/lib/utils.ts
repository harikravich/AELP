import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { format, formatDistanceToNow } from 'date-fns';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(amount: number, currency = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount);
}

export function formatNumber(num: number, decimals = 0): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(num);
}

export function formatPercentage(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatMetric(value: number, metric: string): string {
  switch (metric) {
    case 'ctr':
    case 'conversionRate':
    case 'explorationRate':
      return formatPercentage(value);
    case 'spend':
    case 'revenue':
    case 'cpc':
    case 'cpm':
    case 'cpa':
      return formatCurrency(value);
    case 'roas':
      return `${value.toFixed(2)}x`;
    case 'impressions':
    case 'clicks':
    case 'conversions':
      return formatNumber(value);
    default:
      return formatNumber(value, 2);
  }
}

export function formatTimeAgo(date: Date): string {
  return formatDistanceToNow(date, { addSuffix: true });
}

export function formatDateTime(date: Date): string {
  return format(date, 'MMM dd, yyyy HH:mm');
}

export function getStatusColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'training':
    case 'active':
    case 'deployed':
      return 'text-blue-600 bg-blue-50';
    case 'completed':
    case 'success':
      return 'text-green-600 bg-green-50';
    case 'paused':
    case 'warning':
      return 'text-yellow-600 bg-yellow-50';
    case 'failed':
    case 'error':
    case 'critical':
      return 'text-red-600 bg-red-50';
    default:
      return 'text-gray-600 bg-gray-50';
  }
}

export function getSeverityColor(severity: string): string {
  switch (severity.toLowerCase()) {
    case 'low':
      return 'text-green-600 bg-green-50';
    case 'medium':
      return 'text-yellow-600 bg-yellow-50';
    case 'high':
      return 'text-orange-600 bg-orange-50';
    case 'critical':
      return 'text-red-600 bg-red-50';
    default:
      return 'text-gray-600 bg-gray-50';
  }
}

export function calculateGrowth(current: number, previous: number): number {
  if (previous === 0) return current > 0 ? 100 : 0;
  return ((current - previous) / previous) * 100;
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let lastCall = 0;
  return (...args: Parameters<T>) => {
    const now = Date.now();
    if (now - lastCall >= delay) {
      lastCall = now;
      func(...args);
    }
  };
}

export function generateChartColors(count: number): string[] {
  const baseColors = [
    '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6366f1'
  ];
  
  if (count <= baseColors.length) {
    return baseColors.slice(0, count);
  }
  
  const colors = [...baseColors];
  while (colors.length < count) {
    colors.push(...baseColors);
  }
  
  return colors.slice(0, count);
}

export function downloadCSV(data: any[], filename: string): void {
  if (!data.length) return;
  
  const headers = Object.keys(data[0]);
  const csvContent = [
    headers.join(','),
    ...data.map(row => 
      headers.map(header => {
        const value = row[header];
        return typeof value === 'string' && value.includes(',') 
          ? `"${value}"` 
          : value;
      }).join(',')
    )
  ].join('\n');
  
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.csv`;
  link.click();
  window.URL.revokeObjectURL(url);
}

export function getTimeRangePreset(preset: string): { start: Date; end: Date } {
  const now = new Date();
  const start = new Date();
  
  switch (preset) {
    case '1h':
      start.setHours(now.getHours() - 1);
      break;
    case '24h':
      start.setDate(now.getDate() - 1);
      break;
    case '7d':
      start.setDate(now.getDate() - 7);
      break;
    case '30d':
      start.setDate(now.getDate() - 30);
      break;
    default:
      start.setDate(now.getDate() - 7);
  }
  
  return { start, end: now };
}

export function isStatisticallySignificant(pValue: number, alpha = 0.05): boolean {
  return pValue < alpha;
}

export function calculateConfidenceInterval(
  mean: number,
  stdError: number,
  confidence = 0.95
): [number, number] {
  const z = confidence === 0.95 ? 1.96 : confidence === 0.99 ? 2.58 : 1.64;
  const margin = z * stdError;
  return [mean - margin, mean + margin];
}
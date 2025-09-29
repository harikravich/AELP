import { cookies } from 'next/headers'

export type DatasetMode = 'sandbox' | 'prod'

// Default sandbox to the training dataset if not explicitly provided to avoid hitting a non-existent dataset
export const SANDBOX_DATASET = process.env.BIGQUERY_SANDBOX_DATASET || (process.env.BIGQUERY_TRAINING_DATASET as string)
export const PROD_DATASET = process.env.BIGQUERY_TRAINING_DATASET as string
export const DATASET_COOKIE = 'aelp-dataset'

export function getDatasetFromCookie(): { dataset: string, mode: DatasetMode } {
  const c = cookies()
  // Default to prod so new users see real data without setting a cookie
  const cookieVal = c.get(DATASET_COOKIE)?.value
  const mode = cookieVal === 'sandbox' ? 'sandbox' : 'prod'
  const dataset = mode === 'prod' ? PROD_DATASET : SANDBOX_DATASET
  return { dataset, mode }
}

export function resolveDatasetForAction(action: 'read' | 'write'): { dataset: string, mode: DatasetMode, allowed: boolean, reason?: string } {
  const { dataset, mode } = getDatasetFromCookie()
  if (mode === 'prod' && action === 'write') {
    return { dataset, mode, allowed: false, reason: 'Writes are blocked on prod (safety rule)' }
  }
  return { dataset, mode, allowed: true }
}

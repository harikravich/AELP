export const CONFIG = {
  API_BASE: (import.meta.env.VITE_API_BASE_URL as string) || '',
  DATASET_MODE: ((import.meta.env.VITE_DATASET_MODE as string) === 'prod' ? 'prod' : 'sandbox') as 'sandbox' | 'prod',
}


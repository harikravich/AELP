export type ModuleSlug = 'insight_preview' | 'scam_check'

export interface ModuleSpec {
  slug: ModuleSlug
  title: string
  consentText: string
}

export const MODULES: Record<ModuleSlug, ModuleSpec> = {
  insight_preview: {
    slug: 'insight_preview',
    title: 'Insight Preview',
    consentText: 'I consent to a preview using public/allowed signals and understand no private data is stored.'
  },
  scam_check: {
    slug: 'scam_check',
    title: 'Scam Link Check',
    consentText: 'I consent to a basic link risk check based on the provided URL.'
  }
}


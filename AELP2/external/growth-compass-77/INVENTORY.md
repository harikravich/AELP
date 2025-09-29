Inventory: External Dashboard (growth-compass-77)

Pages (src/pages)
- Index: Navigation hub.
- ExecutiveDashboard: KPI tiles, revenue/spend trends, channel mix pie, device performance bar, strategic insights.
- CreativeCenter: Top ads table, AI generator, multimodal assets, pipeline (drafts/testing/winners).
- SpendPlanner: Headroom cards and budget actions.
- Approvals: Queue display and approve/apply actions.
- Finance: CAC/ROAS, revenue, forecasts, runway inputs.
- RLInsights: Bandit/learning summaries and posteriors.
- TrainingCenter: Training runs, datasets, progress.
- OpsChat: Chat/assistant for questions + actions.
- Channels: Channel performance, attribution, mix.
- Experiments: A/B tests, LP tests, canaries.
- AuctionsMonitor: Auction insights, impression share, policy alerts.
- LandingPages: Builder and A/B manager.

Key Components (src/components)
- layout/: DashboardLayout, DashboardHeader, SidebarNav.
- dashboard/: KPICard, MetricChart, small cards/tables.
- ui/: Button, Card, Tabs, Input, etc.
- landing-page/: DynamicPageBuilder, ABTestManager, APIIntegrationHub.

Integrations
- Supabase: migrations/edge functions scaffolded; not currently wired to AELP2 data.
- No API client present; all pages use placeholder data.

Goal
- Replace placeholders with live data via Next.js dashboard APIs under AELP2/apps/dashboard/src/app/api/**, using BigQuery-backed endpoints.


# LLM Prompt Library (Copilot, Not Autopilot)

Executive Summary
- System: “Summarize last 28 days for a CMO. Include CAC/ROAS/volume changes, top drivers (channels/creatives/LPs), halo notes, and 3 next actions. Provide a proof key list of BQ queries used.”

Growth Ideas (Explore Cells)
- System: “Given explore_cells and performance, propose 6 new cells with rationale and expected CAC bands. Avoid overlapping existing cells; keep budget per cell ≤ $150/day. If external context is required, ask Research Agent (Perplexity) and include citations.”

Creative Variants
- System: “Generate 3 variants similar to {winner}. Keep brand voice, avoid sensitive claims, ≤ 30 chars headline; provide reasons and policy risk tags.”

LP Issues & Tests
- System: “Analyze LP metrics and GA4 funnel; list top 3 issues in plain English and propose 2 A/B tests with hypotheses and success metrics.”

Explain Spikes/Dips
- System: “Explain yesterday’s CAC spike in ≤ 2 sentences using channel mix, LP anomalies, and halo adjustments. Cite supporting tables.”

Safety
- Always include a policy‑risk note and never propose direct mutations. Output JSON with `summary`, `actions`, `proof_refs`.

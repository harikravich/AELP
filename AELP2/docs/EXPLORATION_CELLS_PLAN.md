# Exploration Cells Plan (Initial 24)

Budget policy: 10–15% daily budget for exploration; $50–$150/day per cell. Promote/kill rules: posterior P(CAC ≤ target) ≥ 0.8 and SRM OK → promote; else continue or kill.

Key = angle | audience | channel | lp | offer

## Use Cases & Angles
- Parental Controls / Balance: Balance‑Safety; Safe Gaming; Digital Wellness
- Fraud/Scam Protection: Scam Shield; Real‑Time Alerts
- Identity & Credit Guard: Identity Monitoring; Credit Lock; Teen Identity
- VPN/Safe Browsing: Family VPN; Safe Browsing
- Password Manager: Family Passwords; Shared Vault
- Family Locator/Alerts: Real‑time Check‑ins; Geo‑fences

## Initial 24 Cells
1) Balance‑Safety | Parents iOS | Search‑Brand | /balance‑v2 | 14‑day trial
2) Balance‑Safety | Parents iOS | Search‑NB | /balance‑v2 | 14‑day trial
3) Balance‑Safety | Parents Teens | YouTube Shorts | /balance‑v2 | 14‑day trial
4) Safe Gaming | Parents iOS | YouTube Shorts | /balance‑v2 | 14‑day trial
5) Digital Wellness | Parents Android | Search‑NB | /balance‑v2 | 14‑day trial
6) Scam Shield | Parents 25–54 | Search‑NB | /fraud‑v1 | 14‑day trial
7) Scam Shield | Parents 25–54 | PMax | /fraud‑v1 | 14‑day trial
8) Real‑Time Alerts | Parents iOS | YouTube In‑Stream | /fraud‑v1 | 14‑day trial
9) Identity Monitoring | Parents 25–54 | Search‑NB | /identity‑v1 | 14‑day trial
10) Credit Lock | Parents iOS | Search‑Brand | /identity‑v1 | 14‑day trial
11) Teen Identity | Parents Teens | YouTube Shorts | /identity‑v1 | 14‑day trial
12) Family VPN | Parents 25–54 | PMax | /vpn‑v1 | 14‑day trial
13) Safe Browsing | Parents iOS | Search‑NB | /vpn‑v1 | 14‑day trial
14) Family Passwords | Parents 25–54 | Search‑NB | /passwords‑v1 | 14‑day trial
15) Shared Vault | Parents iOS | Discovery | /passwords‑v1 | 14‑day trial
16) Locator Check‑ins | Parents Teens | YouTube Shorts | /locator‑v1 | 14‑day trial
17) Geo‑fences | Parents 25–54 | Search‑NB | /locator‑v1 | 14‑day trial
18) Balance‑Safety | Parents iOS (CA) | Search‑NB | /balance‑v2 | 14‑day trial
19) Scam Shield | Parents Android | Search‑NB | /fraud‑v1 | 14‑day trial
20) Identity Monitoring | Parents iOS (TX) | Search‑NB | /identity‑v1 | 14‑day trial
21) Family VPN | Parents Android | YouTube Shorts | /vpn‑v1 | 14‑day trial
22) Family Passwords | Parents 25–54 | PMax | /passwords‑v1 | 14‑day trial
23) Locator Check‑ins | Parents 25–54 | PMax | /locator‑v1 | 14‑day trial
24) Safe Browsing | Parents iOS | Discovery | /vpn‑v1 | 14‑day trial

## Guardrails & Metrics
- CAC targets by channel (initial): Search ≤ $80; YouTube ≤ $95; PMax ≤ $85; Discovery ≤ $90. Adjust by segment as we learn.
- Promote/kill cadence: evaluate every 48–72h (or earlier if power achieved); enforce per‑change ≤5% and daily ≤10% caps.
- Halo: weekly GeoLift reads; adjust cell ROI before ramping.

## Data Writing
- Insert cells into `${DATASET}.explore_cells` as they are queued; bandit_service updates posteriors daily to `${DATASET}.bandit_posteriors`.


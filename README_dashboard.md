# GAELP Training Orchestrator

A comprehensive training orchestrator for ad campaign agents that manages the complete simulation-to-real-world learning progression.

## Features

### 🚀 Real-time Monitoring
- Live training progress tracking with WebSocket updates
- Real-time campaign performance metrics
- System health monitoring and alerts
- Resource utilization tracking

### 📊 Comprehensive Analytics
- Interactive training curves (reward, loss, policy entropy)
- Campaign performance analysis (ROAS, CTR, conversion rates)
- A/B testing results and statistical significance
- Agent behavior analysis and exploration patterns

### 🏆 Performance Tracking
- Dynamic leaderboards with multiple ranking metrics
- Agent performance comparisons
- Top performing creatives and strategies
- Historical trend analysis

### 🛡️ Safety & Compliance
- Safety event monitoring and alerting
- Policy violation tracking
- Automated safety reviews
- Budget and spend controls

### 🔧 Reproducibility Tools
- One-click experiment reproduction
- Training configuration export
- Environment version tracking
- Collaboration features

## Tech Stack

- **Frontend**: Next.js 14, React 18, TypeScript
- **Styling**: Tailwind CSS, Framer Motion
- **Data Visualization**: Recharts, D3.js
- **State Management**: Zustand
- **Data Fetching**: React Query
- **Backend**: Next.js API Routes
- **Database**: Google BigQuery
- **Real-time**: WebSockets
- **Deployment**: Docker, Google Cloud Run

## Quick Start

### Prerequisites
- Node.js 18+
- Google Cloud Project with BigQuery enabled
- Service Account with BigQuery permissions

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd aelp-dashboard
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env.local
# Edit .env.local with your configuration
```

4. Run the development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Environment Variables

```env
# Required
GOOGLE_CLOUD_PROJECT_ID=your-project-id
BIGQUERY_DATASET=gaelp_data
NEXTAUTH_SECRET=your-secret

# Optional
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
REDIS_URL=redis://localhost:6379
```

## Project Structure

```
src/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   ├── training/          # Training dashboard pages
│   ├── campaigns/         # Campaign dashboard pages
│   └── leaderboards/      # Leaderboard pages
├── components/            # React components
│   ├── dashboard/         # Dashboard-specific components
│   ├── charts/           # Chart components
│   ├── layout/           # Layout components
│   └── ui/               # Reusable UI components
├── hooks/                 # Custom React hooks
├── lib/                   # Utility functions and services
└── types/                 # TypeScript type definitions
```

## Dashboard Features

### Overview Dashboard
- Key performance indicators
- Real-time updates
- System status
- Quick navigation

### Training Dashboard
- Agent selection and filtering
- Training progress visualization
- Resource utilization monitoring
- Episode progress tracking

### Campaign Dashboard
- Platform-specific performance
- Spend and revenue tracking
- A/B test results
- Campaign optimization insights

### Leaderboards
- Multi-metric rankings
- Performance comparisons
- Top performing agents and creatives
- Historical performance trends

## Data Integration

### BigQuery Schema
The dashboard expects data in BigQuery with the following tables:
- `agents` - Agent metadata and status
- `training_metrics` - Training progress data
- `campaign_metrics` - Campaign performance data
- `safety_events` - Safety monitoring events
- `ab_test_results` - A/B testing results

### API Endpoints
- `GET /api/overview/stats` - Dashboard overview statistics
- `GET /api/training/metrics` - Training progress data
- `GET /api/campaigns/metrics` - Campaign performance data
- `GET /api/safety/events` - Safety events
- `WebSocket /api/ws` - Real-time updates

## Deployment

### Docker
```bash
# Build the image
docker build -t gaelp-dashboard .

# Run the container
docker run -p 3000:3000 --env-file .env gaelp-dashboard
```

### Google Cloud Run
```bash
# Build and deploy
gcloud run deploy gaelp-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Development

### Running Tests
```bash
npm test
```

### Building for Production
```bash
npm run build
npm start
```

### Code Quality
```bash
npm run lint
npm run type-check
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Security

- All API endpoints include rate limiting
- Authentication via NextAuth.js
- CORS protection
- Input validation and sanitization
- Secure headers configuration

## Performance

- Server-side rendering for fast initial loads
- Progressive loading and pagination
- Efficient data caching with React Query
- Image optimization
- Bundle splitting and code optimization

## Monitoring

- Real-time performance metrics
- Error tracking with Sentry (optional)
- Comprehensive logging
- Health check endpoints

## License

MIT License - see LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Contact the GAELP team
- Check the documentation wiki
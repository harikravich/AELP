# Welcome to your Lovable project

## Project info

**URL**: https://lovable.dev/projects/d27421b3-622d-4c5c-96a4-d3baf9178cb3

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/d27421b3-622d-4c5c-96a4-d3baf9178cb3) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## Integration with AELP2

This dashboard is wired to the AELP2 Next.js API suite and BigQuery datasets. To use with real data:

1) Run the Next.js app: `cd AELP2/apps/dashboard && npm i && npm run dev` (requires GOOGLE_CLOUD_PROJECT and BQ access). Set dataset via `POST /api/dataset?mode=sandbox|prod`.
2) Configure this app: create `.env` with `VITE_API_BASE_URL=` (empty for same-origin) and `VITE_DATASET_MODE=sandbox`. The dev server proxies `/api` to `http://localhost:3000`.
3) Install and run: `npm ci --include=dev && npm run dev` (or `npm run build && npm run preview`).
4) Optional smoke test: `API_BASE=http://localhost:3000 npm run smoke:api`.

Wired pages: Executive (KPIs, headroom), Creative Center (creatives + Ads preview, enqueue), Spend Planner (headroom/MMM, approvals), Approvals (queue + apply), Channels (attribution), RL Insights (decisions), Training (ops status), Finance (CAC/ROAS + identity triggers).

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/d27421b3-622d-4c5c-96a4-d3baf9178cb3) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/tips-tricks/custom-domain#step-by-step-guide)

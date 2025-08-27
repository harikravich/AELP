import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { TopBar } from '@/components/layout/TopBar';
import { LeaderboardDashboard } from '@/components/dashboard/LeaderboardDashboard';

export default function LeaderboardsPage() {
  return (
    <DashboardLayout>
      <TopBar
        title="Leaderboards"
        showTimeRange={true}
        showFilters={false}
        showExport={true}
      />
      <LeaderboardDashboard />
    </DashboardLayout>
  );
}
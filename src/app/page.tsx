import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { TopBar } from '@/components/layout/TopBar';
import { OverviewDashboard } from '@/components/dashboard/OverviewDashboard';

export default function HomePage() {
  return (
    <DashboardLayout>
      <TopBar
        title="Overview"
        showTimeRange={true}
        showFilters={true}
        showExport={true}
      />
      <OverviewDashboard />
    </DashboardLayout>
  );
}
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { TopBar } from '@/components/layout/TopBar';
import { CampaignsDashboard } from '@/components/dashboard/CampaignsDashboard';

export default function CampaignsPage() {
  return (
    <DashboardLayout>
      <TopBar
        title="Campaign Performance"
        showTimeRange={true}
        showFilters={true}
        showExport={true}
      />
      <CampaignsDashboard />
    </DashboardLayout>
  );
}
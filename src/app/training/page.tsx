import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { TopBar } from '@/components/layout/TopBar';
import { TrainingDashboard } from '@/components/dashboard/TrainingDashboard';

export default function TrainingPage() {
  return (
    <DashboardLayout>
      <TopBar
        title="Training Dashboard"
        showTimeRange={true}
        showFilters={true}
        showExport={true}
      />
      <TrainingDashboard />
    </DashboardLayout>
  );
}
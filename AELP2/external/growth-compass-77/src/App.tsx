import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useEffect } from 'react'
import { api } from './integrations/aelp-api/client'
import { CONFIG } from './lib/config'
import Index from "./pages/Index";
import CreativeCenter from "./pages/CreativeCenter";
import ExecutiveDashboard from "./pages/ExecutiveDashboard";
import SpendPlanner from "./pages/SpendPlanner";
import Approvals from "./pages/Approvals";
import CreativePlanner from "./pages/CreativePlanner";
import Finance from "./pages/Finance";
import RLInsights from "./pages/RLInsights";
import TrainingCenter from "./pages/TrainingCenter";
import OpsChat from "./pages/OpsChat";
import Channels from "./pages/Channels";
import Experiments from "./pages/Experiments";
import AuctionsMonitor from "./pages/AuctionsMonitor";
import LandingPages from "./pages/LandingPages";
import NotFound from "./pages/NotFound";
import Backstage from "./pages/Backstage";
import Audiences from "./pages/Audiences";
import Canvas from "./pages/Canvas";
import QS from "./pages/QS";
import Affiliates from "./pages/Affiliates";

const queryClient = new QueryClient();

function InitDataset() {
  useEffect(() => {
    api.dataset.set(CONFIG.DATASET_MODE).catch(() => void 0)
  }, [])
  return null
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <InitDataset />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/creative-center" element={<CreativeCenter />} />
          <Route path="/creative-planner" element={<CreativePlanner />} />
          <Route path="/executive" element={<ExecutiveDashboard />} />
          <Route path="/spend-planner" element={<SpendPlanner />} />
          <Route path="/qs" element={<QS />} />
          <Route path="/affiliates" element={<Affiliates />} />
          <Route path="/approvals" element={<Approvals />} />
          <Route path="/finance" element={<Finance />} />
          <Route path="/rl-insights" element={<RLInsights />} />
          <Route path="/training" element={<TrainingCenter />} />
          <Route path="/chat" element={<OpsChat />} />
          <Route path="/channels" element={<Channels />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/auctions" element={<AuctionsMonitor />} />
          <Route path="/backstage" element={<Backstage />} />
          <Route path="/audiences" element={<Audiences />} />
          <Route path="/canvas" element={<Canvas />} />
          <Route path="/landing-pages" element={<LandingPages />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;

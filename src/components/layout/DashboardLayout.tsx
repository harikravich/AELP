'use client';

import { useState } from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { 
  BarChart3, 
  Activity, 
  Target, 
  Brain, 
  Trophy, 
  Settings, 
  Bell, 
  User,
  Menu,
  X,
  ChevronDown
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { useDashboard } from '@/hooks/useDashboard';

interface NavigationItem {
  name: string;
  href: string;
  icon: React.ReactNode;
  badge?: number;
  children?: NavigationItem[];
}

const navigation: NavigationItem[] = [
  {
    name: 'Overview',
    href: '/',
    icon: <BarChart3 size={20} />,
  },
  {
    name: 'Training',
    href: '/training',
    icon: <Activity size={20} />,
    children: [
      { name: 'Progress', href: '/training/progress', icon: <Activity size={16} /> },
      { name: 'Agents', href: '/training/agents', icon: <Brain size={16} /> },
      { name: 'Resources', href: '/training/resources', icon: <Settings size={16} /> },
    ],
  },
  {
    name: 'Campaigns',
    href: '/campaigns',
    icon: <Target size={20} />,
    children: [
      { name: 'Performance', href: '/campaigns/performance', icon: <BarChart3 size={16} /> },
      { name: 'A/B Tests', href: '/campaigns/ab-tests', icon: <Target size={16} /> },
      { name: 'Analytics', href: '/campaigns/analytics', icon: <Brain size={16} /> },
    ],
  },
  {
    name: 'Leaderboards',
    href: '/leaderboards',
    icon: <Trophy size={20} />,
  },
  {
    name: 'Safety',
    href: '/safety',
    icon: <Bell size={20} />,
    badge: 3,
  },
];

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [expandedItems, setExpandedItems] = useState<string[]>([]);
  const pathname = usePathname();
  const { activeView, setActiveView } = useDashboard();

  const toggleExpanded = (itemName: string) => {
    setExpandedItems(prev => 
      prev.includes(itemName) 
        ? prev.filter(name => name !== itemName)
        : [...prev, itemName]
    );
  };

  const isActiveLink = (href: string) => {
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
  };

  const NavItem = ({ item, level = 0 }: { item: NavigationItem; level?: number }) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedItems.includes(item.name);
    const isActive = isActiveLink(item.href);

    return (
      <div>
        <Link
          href={item.href}
          className={`
            flex items-center gap-3 px-3 py-2 text-sm font-medium rounded-md transition-colors
            ${level > 0 ? 'ml-6' : ''}
            ${isActive 
              ? 'bg-primary-50 text-primary-700 border-r-2 border-primary-600' 
              : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
            }
          `}
          onClick={() => setSidebarOpen(false)}
        >
          {item.icon}
          <span className="flex-1">{item.name}</span>
          {item.badge && (
            <Badge variant="error" size="sm">
              {item.badge}
            </Badge>
          )}
          {hasChildren && (
            <button
              onClick={(e) => {
                e.preventDefault();
                toggleExpanded(item.name);
              }}
              className="p-1 hover:bg-gray-200 rounded"
            >
              <ChevronDown 
                size={16} 
                className={`transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              />
            </button>
          )}
        </Link>
        
        {hasChildren && isExpanded && (
          <div className="mt-1 space-y-1">
            {item.children!.map((child) => (
              <NavItem key={child.name} item={child} level={level + 1} />
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out
        lg:translate-x-0 lg:static lg:inset-0
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <BarChart3 size={20} className="text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900">GAELP</span>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden p-1 hover:bg-gray-100 rounded"
          >
            <X size={20} />
          </button>
        </div>

        <nav className="flex-1 px-4 py-4 space-y-1 overflow-y-auto">
          {navigation.map((item) => (
            <NavItem key={item.name} item={item} />
          ))}
        </nav>

        <div className="p-4 border-t border-gray-200">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center">
              <User size={16} />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-gray-900 truncate">
                John Doe
              </div>
              <div className="text-xs text-gray-500 truncate">
                Researcher
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden lg:ml-0">
        {/* Top Header */}
        <header className="bg-white shadow-sm border-b border-gray-200 lg:hidden">
          <div className="flex items-center justify-between h-16 px-4">
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 hover:bg-gray-100 rounded-md"
            >
              <Menu size={20} />
            </button>
            <h1 className="text-lg font-semibold text-gray-900">
              Dashboard
            </h1>
            <div className="w-8" /> {/* Spacer */}
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto">
          <div className="p-6">
            {children}
          </div>
        </main>
      </div>

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-gray-600 bg-opacity-50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
}
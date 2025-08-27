/**
 * Shared types and interfaces for advertising platform MCP connectors
 */

// Base interfaces
export interface BaseConfig {
  apiKey: string;
  accessToken: string;
  refreshToken?: string;
  clientId?: string;
  clientSecret?: string;
  accountId: string;
  rateLimitPerSecond: number;
  retryAttempts: number;
  timeout: number;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
  rateLimitRemaining?: number;
  rateLimitReset?: number;
}

// Campaign interfaces
export interface CampaignObjective {
  type: 'BRAND_AWARENESS' | 'REACH' | 'TRAFFIC' | 'ENGAGEMENT' | 'APP_INSTALLS' | 
        'VIDEO_VIEWS' | 'LEAD_GENERATION' | 'MESSAGES' | 'CONVERSIONS' | 'SALES';
}

export interface Budget {
  amount: number;
  currency: 'USD' | 'EUR' | 'GBP' | 'CAD' | 'AUD';
  type: 'DAILY' | 'LIFETIME';
}

export interface Targeting {
  demographics: {
    ageMin?: number;
    ageMax?: number;
    genders?: ('male' | 'female' | 'all')[];
  };
  locations: {
    countries?: string[];
    regions?: string[];
    cities?: string[];
    postalCodes?: string[];
  };
  interests?: string[];
  behaviors?: string[];
  customAudiences?: string[];
  lookalikeSources?: string[];
  devices?: ('mobile' | 'desktop' | 'tablet')[];
  languages?: string[];
}

export interface Creative {
  id?: string;
  name: string;
  type: 'IMAGE' | 'VIDEO' | 'CAROUSEL' | 'COLLECTION' | 'TEXT';
  assets: {
    images?: {
      url: string;
      altText?: string;
    }[];
    videos?: {
      url: string;
      thumbnail?: string;
    }[];
    text?: {
      headline: string;
      description: string;
      callToAction?: string;
    };
  };
  destinationUrl?: string;
}

export interface Campaign {
  id?: string;
  name: string;
  status: 'ACTIVE' | 'PAUSED' | 'DELETED' | 'PENDING_REVIEW' | 'DISAPPROVED';
  objective: CampaignObjective;
  budget: Budget;
  targeting: Targeting;
  creatives: Creative[];
  startDate?: string;
  endDate?: string;
  bidStrategy?: {
    type: 'AUTOMATIC' | 'MANUAL_CPC' | 'MANUAL_CPM' | 'TARGET_CPA' | 'TARGET_ROAS';
    amount?: number;
  };
  platformSpecific?: any; // Platform-specific configurations
}

// Performance metrics
export interface PerformanceMetrics {
  impressions: number;
  clicks: number;
  conversions: number;
  spend: number;
  ctr: number; // Click-through rate
  cpc: number; // Cost per click
  cpm: number; // Cost per mille
  cpa?: number; // Cost per acquisition
  roas?: number; // Return on ad spend
  frequency?: number;
  reach?: number;
  dateRange: {
    start: string;
    end: string;
  };
}

export interface CampaignPerformance extends PerformanceMetrics {
  campaignId: string;
  campaignName: string;
  breakdown?: {
    byAge?: { [ageGroup: string]: PerformanceMetrics };
    byGender?: { [gender: string]: PerformanceMetrics };
    byLocation?: { [location: string]: PerformanceMetrics };
    byDevice?: { [device: string]: PerformanceMetrics };
    byCreative?: { [creativeId: string]: PerformanceMetrics };
  };
}

// Audience interfaces
export interface Audience {
  id?: string;
  name: string;
  type: 'CUSTOM' | 'LOOKALIKE' | 'SAVED';
  description?: string;
  size?: number;
  source?: {
    type: 'PIXEL' | 'CUSTOMER_LIST' | 'MOBILE_APP' | 'WEBSITE_VISITORS' | 'ENGAGEMENT';
    details: any;
  };
  retentionDays?: number;
}

// Error handling
export interface RateLimitInfo {
  remaining: number;
  resetTime: number;
  limit: number;
}

export interface ConnectorError extends Error {
  code: string;
  statusCode?: number;
  rateLimitInfo?: RateLimitInfo;
  retryAfter?: number;
  platform: 'META' | 'GOOGLE';
  details?: any;
}

// Validation schemas
export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface ContentPolicyValidation extends ValidationResult {
  violatedPolicies?: string[];
  suggestedChanges: string[];
}

// Spending controls
export interface SpendingLimits {
  dailyLimit: number;
  monthlyLimit: number;
  campaignLimit: number;
  currency: string;
  alertThresholds: {
    warning: number; // Percentage of limit
    critical: number; // Percentage of limit
  };
}

export interface SpendingStatus {
  currentSpend: {
    daily: number;
    monthly: number;
    total: number;
  };
  remainingBudget: {
    daily: number;
    monthly: number;
    total: number;
  };
  alerts: {
    level: 'INFO' | 'WARNING' | 'CRITICAL';
    message: string;
    timestamp: string;
  }[];
}
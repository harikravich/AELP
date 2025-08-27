/**
 * Base connector class for advertising platform integrations
 */

import { 
  BaseConfig, 
  ApiResponse, 
  Campaign, 
  CampaignPerformance, 
  Creative, 
  Audience, 
  ConnectorError,
  SpendingLimits,
  SpendingStatus,
  ContentPolicyValidation 
} from './types.js';
import { RateLimiter, retryWithBackoff, Logger, validateConfig } from './utils.js';

export abstract class BaseAdConnector {
  protected config: BaseConfig;
  protected rateLimiter: RateLimiter;
  protected logger: Logger;
  protected spendingLimits?: SpendingLimits;
  
  constructor(config: BaseConfig, platform: string) {
    const validation = validateConfig(config, [
      'apiKey', 'accessToken', 'accountId', 'rateLimitPerSecond'
    ]);
    
    if (!validation.valid) {
      throw new Error(`Invalid configuration: ${validation.errors.join(', ')}`);
    }
    
    this.config = config;
    this.rateLimiter = new RateLimiter(config.rateLimitPerSecond);
    this.logger = new Logger(`${platform}-connector`);
  }
  
  // Abstract methods that must be implemented by platform-specific connectors
  abstract testConnection(): Promise<ApiResponse<boolean>>;
  
  // Campaign management
  abstract createCampaign(campaign: Campaign): Promise<ApiResponse<Campaign>>;
  abstract updateCampaign(campaignId: string, updates: Partial<Campaign>): Promise<ApiResponse<Campaign>>;
  abstract getCampaign(campaignId: string): Promise<ApiResponse<Campaign>>;
  abstract listCampaigns(filters?: any): Promise<ApiResponse<Campaign[]>>;
  abstract deleteCampaign(campaignId: string): Promise<ApiResponse<boolean>>;
  abstract pauseCampaign(campaignId: string): Promise<ApiResponse<boolean>>;
  abstract resumeCampaign(campaignId: string): Promise<ApiResponse<boolean>>;
  
  // Creative management
  abstract uploadCreative(creative: Creative): Promise<ApiResponse<Creative>>;
  abstract updateCreative(creativeId: string, updates: Partial<Creative>): Promise<ApiResponse<Creative>>;
  abstract getCreative(creativeId: string): Promise<ApiResponse<Creative>>;
  abstract listCreatives(filters?: any): Promise<ApiResponse<Creative[]>>;
  abstract deleteCreative(creativeId: string): Promise<ApiResponse<boolean>>;
  
  // Performance monitoring
  abstract getCampaignPerformance(
    campaignId: string, 
    dateRange: { start: string; end: string },
    metrics?: string[]
  ): Promise<ApiResponse<CampaignPerformance>>;
  
  abstract getAccountPerformance(
    dateRange: { start: string; end: string },
    breakdown?: string
  ): Promise<ApiResponse<CampaignPerformance[]>>;
  
  // Audience management
  abstract createAudience(audience: Audience): Promise<ApiResponse<Audience>>;
  abstract getAudience(audienceId: string): Promise<ApiResponse<Audience>>;
  abstract listAudiences(filters?: any): Promise<ApiResponse<Audience[]>>;
  abstract deleteAudience(audienceId: string): Promise<ApiResponse<boolean>>;
  
  // Content policy validation
  abstract validateContent(creative: Creative): Promise<ApiResponse<ContentPolicyValidation>>;
  
  // Protected utility methods
  protected async makeApiRequest<T>(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    url: string,
    data?: any,
    headers?: Record<string, string>
  ): Promise<ApiResponse<T>> {
    await this.rateLimiter.waitIfNeeded();
    
    return retryWithBackoff(async () => {
      this.logger.debug(`Making ${method} request to ${url}`);
      
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.accessToken}`,
          ...headers
        },
        body: data ? JSON.stringify(data) : undefined,
        signal: AbortSignal.timeout(this.config.timeout || 30000)
      });
      
      const responseData = await response.json();
      
      if (!response.ok) {
        throw this.createError(
          responseData.error?.message || `HTTP ${response.status}`,
          responseData.error?.code || response.status.toString(),
          response.status,
          responseData
        );
      }
      
      return {
        success: true,
        data: responseData,
        rateLimitRemaining: this.rateLimiter.getRemainingRequests()
      };
    }, this.config.retryAttempts || 3);
  }
  
  protected createError(
    message: string, 
    code: string, 
    statusCode?: number, 
    details?: any
  ): ConnectorError {
    const error = new Error(message) as ConnectorError;
    error.name = 'ConnectorError';
    error.code = code;
    error.statusCode = statusCode;
    error.platform = this.getPlatform();
    
    if (details) {
      error.details = details;
    }
    
    // Handle rate limiting
    if (statusCode === 429) {
      error.retryAfter = details?.retryAfter || 60;
      error.rateLimitInfo = {
        remaining: 0,
        resetTime: Date.now() + ((error.retryAfter || 60) * 1000),
        limit: this.config.rateLimitPerSecond
      };
    }
    
    return error;
  }
  
  protected abstract getPlatform(): 'META' | 'GOOGLE';
  
  // Spending control methods
  setSpendingLimits(limits: SpendingLimits): void {
    this.spendingLimits = limits;
    this.logger.info('Spending limits updated', limits);
  }
  
  async getSpendingStatus(): Promise<SpendingStatus | null> {
    if (!this.spendingLimits) {
      return null;
    }
    
    try {
      // Get current month's date range
      const now = new Date();
      const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);
      const dayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      
      // Get performance data for spending calculation
      const monthlyPerformance = await this.getAccountPerformance({
        start: monthStart.toISOString().split('T')[0],
        end: now.toISOString().split('T')[0]
      });
      
      const dailyPerformance = await this.getAccountPerformance({
        start: dayStart.toISOString().split('T')[0],
        end: now.toISOString().split('T')[0]
      });
      
      if (!monthlyPerformance.success || !dailyPerformance.success) {
        throw new Error('Failed to retrieve spending data');
      }
      
      const monthlySpend = monthlyPerformance.data?.reduce((sum, campaign) => sum + campaign.spend, 0) || 0;
      const dailySpend = dailyPerformance.data?.reduce((sum, campaign) => sum + campaign.spend, 0) || 0;
      
      const status: SpendingStatus = {
        currentSpend: {
          daily: dailySpend,
          monthly: monthlySpend,
          total: monthlySpend // Simplified - could track all-time spend
        },
        remainingBudget: {
          daily: Math.max(0, this.spendingLimits.dailyLimit - dailySpend),
          monthly: Math.max(0, this.spendingLimits.monthlyLimit - monthlySpend),
          total: Math.max(0, this.spendingLimits.monthlyLimit - monthlySpend)
        },
        alerts: []
      };
      
      // Generate alerts
      const dailyUtilization = (dailySpend / this.spendingLimits.dailyLimit) * 100;
      const monthlyUtilization = (monthlySpend / this.spendingLimits.monthlyLimit) * 100;
      
      if (dailyUtilization >= this.spendingLimits.alertThresholds.critical) {
        status.alerts.push({
          level: 'CRITICAL',
          message: `Daily spending is at ${dailyUtilization.toFixed(1)}% of limit`,
          timestamp: new Date().toISOString()
        });
      } else if (dailyUtilization >= this.spendingLimits.alertThresholds.warning) {
        status.alerts.push({
          level: 'WARNING',
          message: `Daily spending is at ${dailyUtilization.toFixed(1)}% of limit`,
          timestamp: new Date().toISOString()
        });
      }
      
      if (monthlyUtilization >= this.spendingLimits.alertThresholds.critical) {
        status.alerts.push({
          level: 'CRITICAL',
          message: `Monthly spending is at ${monthlyUtilization.toFixed(1)}% of limit`,
          timestamp: new Date().toISOString()
        });
      } else if (monthlyUtilization >= this.spendingLimits.alertThresholds.warning) {
        status.alerts.push({
          level: 'WARNING',
          message: `Monthly spending is at ${monthlyUtilization.toFixed(1)}% of limit`,
          timestamp: new Date().toISOString()
        });
      }
      
      return status;
    } catch (error) {
      this.logger.error('Failed to get spending status', error);
      return null;
    }
  }
  
  protected async checkSpendingLimits(additionalSpend: number): Promise<boolean> {
    if (!this.spendingLimits) {
      return true; // No limits set
    }
    
    const status = await this.getSpendingStatus();
    if (!status) {
      this.logger.warn('Could not verify spending limits - proceeding with caution');
      return true;
    }
    
    // Check if additional spend would exceed limits
    if (status.currentSpend.daily + additionalSpend > this.spendingLimits.dailyLimit) {
      this.logger.error('Operation would exceed daily spending limit');
      return false;
    }
    
    if (status.currentSpend.monthly + additionalSpend > this.spendingLimits.monthlyLimit) {
      this.logger.error('Operation would exceed monthly spending limit');
      return false;
    }
    
    return true;
  }
  
  // Health check methods
  async healthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    checks: {
      connection: boolean;
      rateLimits: boolean;
      spending: boolean;
    };
    details: any;
  }> {
    const checks = {
      connection: false,
      rateLimits: false,
      spending: false
    };
    
    const details: any = {};
    
    // Test connection
    try {
      const connectionTest = await this.testConnection();
      checks.connection = connectionTest.success;
      details.connection = connectionTest.success ? 'OK' : connectionTest.error?.message;
    } catch (error) {
      details.connection = (error as Error).message;
    }
    
    // Check rate limits
    const remainingRequests = this.rateLimiter.getRemainingRequests();
    checks.rateLimits = remainingRequests > 0;
    details.rateLimits = `${remainingRequests} requests remaining`;
    
    // Check spending status
    try {
      const spendingStatus = await this.getSpendingStatus();
      checks.spending = spendingStatus ? spendingStatus.alerts.length === 0 : true;
      details.spending = spendingStatus || 'No spending limits configured';
    } catch (error) {
      details.spending = (error as Error).message;
    }
    
    // Determine overall status
    const healthyChecks = Object.values(checks).filter(Boolean).length;
    const status = healthyChecks === 3 ? 'healthy' : 
                   healthyChecks >= 2 ? 'degraded' : 'unhealthy';
    
    return { status, checks, details };
  }
}
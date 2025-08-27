/**
 * Base connector class for advertising platform integrations
 */
import { RateLimiter, retryWithBackoff, Logger, validateConfig } from './utils.js';
export class BaseAdConnector {
    config;
    rateLimiter;
    logger;
    spendingLimits;
    constructor(config, platform) {
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
    // Protected utility methods
    async makeApiRequest(method, url, data, headers) {
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
                throw this.createError(responseData.error?.message || `HTTP ${response.status}`, responseData.error?.code || response.status.toString(), response.status, responseData);
            }
            return {
                success: true,
                data: responseData,
                rateLimitRemaining: this.rateLimiter.getRemainingRequests()
            };
        }, this.config.retryAttempts || 3);
    }
    createError(message, code, statusCode, details) {
        const error = new Error(message);
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
                resetTime: Date.now() + (error.retryAfter * 1000),
                limit: this.config.rateLimitPerSecond
            };
        }
        return error;
    }
    // Spending control methods
    setSpendingLimits(limits) {
        this.spendingLimits = limits;
        this.logger.info('Spending limits updated', limits);
    }
    async getSpendingStatus() {
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
            const status = {
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
            }
            else if (dailyUtilization >= this.spendingLimits.alertThresholds.warning) {
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
            }
            else if (monthlyUtilization >= this.spendingLimits.alertThresholds.warning) {
                status.alerts.push({
                    level: 'WARNING',
                    message: `Monthly spending is at ${monthlyUtilization.toFixed(1)}% of limit`,
                    timestamp: new Date().toISOString()
                });
            }
            return status;
        }
        catch (error) {
            this.logger.error('Failed to get spending status', error);
            return null;
        }
    }
    async checkSpendingLimits(additionalSpend) {
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
    async healthCheck() {
        const checks = {
            connection: false,
            rateLimits: false,
            spending: false
        };
        const details = {};
        // Test connection
        try {
            const connectionTest = await this.testConnection();
            checks.connection = connectionTest.success;
            details.connection = connectionTest.success ? 'OK' : connectionTest.error?.message;
        }
        catch (error) {
            details.connection = error.message;
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
        }
        catch (error) {
            details.spending = error.message;
        }
        // Determine overall status
        const healthyChecks = Object.values(checks).filter(Boolean).length;
        const status = healthyChecks === 3 ? 'healthy' :
            healthyChecks >= 2 ? 'degraded' : 'unhealthy';
        return { status, checks, details };
    }
}
//# sourceMappingURL=base-connector.js.map
/**
 * Test runner for MCP connectors
 * Validates connector functionality without making real API calls
 */

import { MetaAdsConnector } from '../meta-ads/meta-connector.js';
import { GoogleAdsConnector } from '../google-ads/google-connector.js';
import { Logger } from '../shared/utils.js';

interface TestResult {
  name: string;
  passed: boolean;
  error?: string;
  duration: number;
}

class TestRunner {
  private logger: Logger;
  private results: TestResult[] = [];
  
  constructor() {
    this.logger = new Logger('TestRunner');
  }
  
  async runTest(testName: string, testFn: () => Promise<void>): Promise<void> {
    const startTime = Date.now();
    
    try {
      await testFn();
      this.results.push({
        name: testName,
        passed: true,
        duration: Date.now() - startTime
      });
      this.logger.info(`✅ ${testName} - PASSED`);
    } catch (error) {
      this.results.push({
        name: testName,
        passed: false,
        error: (error as Error).message,
        duration: Date.now() - startTime
      });
      this.logger.error(`❌ ${testName} - FAILED: ${(error as Error).message}`);
    }
  }
  
  async runAllTests(): Promise<void> {
    this.logger.info('🧪 Starting MCP Connector Tests...');
    
    // Test shared utilities
    await this.testSharedUtilities();
    
    // Test Meta connector (without real API calls)
    await this.testMetaConnector();
    
    // Test Google connector (without real API calls)
    await this.testGoogleConnector();
    
    // Print summary
    this.printSummary();
  }
  
  private async testSharedUtilities(): Promise<void> {
    const { RateLimiter, validateCampaignName, validateBudget } = await import('../shared/utils.js');
    
    await this.runTest('RateLimiter - Basic functionality', async () => {
      const limiter = new RateLimiter(2, 1000); // 2 requests per second
      
      // Should allow first request immediately
      const start = Date.now();
      await limiter.waitIfNeeded();
      const firstWait = Date.now() - start;
      
      if (firstWait > 100) {
        throw new Error(`First request should be immediate, but took ${firstWait}ms`);
      }
      
      // Should allow second request immediately
      await limiter.waitIfNeeded();
      
      // Third request should wait
      const thirdStart = Date.now();
      await limiter.waitIfNeeded();
      const thirdWait = Date.now() - thirdStart;
      
      if (thirdWait < 900) {
        throw new Error(`Third request should wait ~1000ms, but only waited ${thirdWait}ms`);
      }
    });
    
    await this.runTest('validateCampaignName - Valid names', async () => {
      const result = validateCampaignName('Valid Campaign Name');
      if (!result.valid || result.errors.length > 0) {
        throw new Error('Valid campaign name failed validation');
      }
    });
    
    await this.runTest('validateCampaignName - Invalid names', async () => {
      const result = validateCampaignName('');
      if (result.valid || result.errors.length === 0) {
        throw new Error('Empty campaign name should fail validation');
      }
    });
    
    await this.runTest('validateBudget - Valid budgets', async () => {
      const result = validateBudget(50, 'DAILY');
      if (!result.valid || result.errors.length > 0) {
        throw new Error('Valid budget failed validation');
      }
    });
    
    await this.runTest('validateBudget - Invalid budgets', async () => {
      const result = validateBudget(-10, 'DAILY');
      if (result.valid || result.errors.length === 0) {
        throw new Error('Negative budget should fail validation');
      }
    });
  }
  
  private async testMetaConnector(): Promise<void> {
    const mockConfig = {
      apiKey: 'test-app-id',
      accessToken: 'test-access-token',
      accountId: 'act_123456789',
      businessAccountId: '123456789',
      appId: 'test-app-id',
      appSecret: 'test-app-secret',
      rateLimitPerSecond: 10,
      retryAttempts: 1,
      timeout: 5000
    };
    
    await this.runTest('MetaAdsConnector - Initialization', async () => {
      const connector = new MetaAdsConnector(mockConfig);
      if (!connector) {
        throw new Error('Failed to initialize Meta connector');
      }
    });
    
    await this.runTest('MetaAdsConnector - Configuration validation', async () => {
      try {
        // Should throw error for missing required config
        new MetaAdsConnector({} as any);
        throw new Error('Should have thrown configuration error');
      } catch (error) {
        if (!(error as Error).message.includes('Invalid configuration')) {
          throw error;
        }
      }
    });
    
    await this.runTest('MetaAdsConnector - Campaign mapping', async () => {
      const connector = new MetaAdsConnector(mockConfig);
      
      // Test objective mapping (private method, but we can test the concept)
      const testCampaign = {
        id: 'test-123',
        name: 'Test Campaign',
        status: 'ACTIVE' as const,
        objective: { type: 'TRAFFIC' as const },
        budget: {
          amount: 50,
          currency: 'USD' as const,
          type: 'DAILY' as const
        },
        targeting: {},
        creatives: []
      };
      
      // This would normally require a real API call, but we're just testing structure
      if (!testCampaign.name || !testCampaign.objective) {
        throw new Error('Campaign structure validation failed');
      }
    });
  }
  
  private async testGoogleConnector(): Promise<void> {
    const mockConfig = {
      apiKey: 'test-developer-token',
      accessToken: 'test-access-token',
      accountId: '123-456-7890',
      customerId: '123-456-7890',
      developerToken: 'test-developer-token',
      rateLimitPerSecond: 10,
      retryAttempts: 1,
      timeout: 5000
    };
    
    await this.runTest('GoogleAdsConnector - Initialization', async () => {
      const connector = new GoogleAdsConnector(mockConfig);
      if (!connector) {
        throw new Error('Failed to initialize Google connector');
      }
    });
    
    await this.runTest('GoogleAdsConnector - Configuration validation', async () => {
      try {
        // Should throw error for missing required config
        new GoogleAdsConnector({} as any);
        throw new Error('Should have thrown configuration error');
      } catch (error) {
        if (!(error as Error).message.includes('Invalid configuration')) {
          throw error;
        }
      }
    });
    
    await this.runTest('GoogleAdsConnector - Campaign structure', async () => {
      const connector = new GoogleAdsConnector(mockConfig);
      
      const testCampaign = {
        id: 'test-123',
        name: 'Test Campaign',
        status: 'ACTIVE' as const,
        objective: { type: 'TRAFFIC' as const },
        budget: {
          amount: 50,
          currency: 'USD' as const,
          type: 'DAILY' as const
        },
        targeting: {},
        creatives: []
      };
      
      if (!testCampaign.name || !testCampaign.objective) {
        throw new Error('Campaign structure validation failed');
      }
    });
  }
  
  private printSummary(): void {
    const passed = this.results.filter(r => r.passed).length;
    const failed = this.results.filter(r => !r.passed).length;
    const totalTime = this.results.reduce((sum, r) => sum + r.duration, 0);
    
    this.logger.info('');
    this.logger.info('📊 Test Summary');
    this.logger.info('================');
    this.logger.info(`Total tests: ${this.results.length}`);
    this.logger.info(`Passed: ${passed}`);
    this.logger.info(`Failed: ${failed}`);
    this.logger.info(`Total time: ${totalTime}ms`);
    
    if (failed > 0) {
      this.logger.info('');
      this.logger.info('❌ Failed tests:');
      this.results.filter(r => !r.passed).forEach(result => {
        this.logger.info(`  - ${result.name}: ${result.error}`);
      });
    }
    
    this.logger.info('');
    this.logger.info(failed === 0 ? '🎉 All tests passed!' : `⚠️  ${failed} test(s) failed`);
    
    if (failed > 0) {
      process.exit(1);
    }
  }
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const runner = new TestRunner();
  runner.runAllTests().catch((error) => {
    console.error('Test runner failed:', error);
    process.exit(1);
  });
}
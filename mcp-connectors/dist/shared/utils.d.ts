/**
 * Shared utilities for MCP advertising connectors
 */
import { ValidationResult } from './types.js';
export declare class RateLimiter {
    private requests;
    private maxRequests;
    private windowMs;
    constructor(maxRequests: number, windowMs?: number);
    waitIfNeeded(): Promise<void>;
    getRemainingRequests(): number;
}
export declare function retryWithBackoff<T>(fn: () => Promise<T>, maxRetries?: number, baseDelay?: number): Promise<T>;
export declare function validateCampaignName(name: string): ValidationResult;
export declare function validateBudget(amount: number, type: 'DAILY' | 'LIFETIME'): ValidationResult;
export declare function validateTargeting(targeting: any): ValidationResult;
export declare function sanitizeInput(input: string): string;
export declare function maskSensitiveData(data: any): any;
export declare function buildQueryString(params: Record<string, any>): string;
export declare function formatDate(date: Date): string;
export declare function parseDate(dateString: string): Date;
export declare function getDateRange(days: number): {
    start: string;
    end: string;
};
export declare class Logger {
    private context;
    constructor(context: string);
    info(message: string, data?: any): void;
    warn(message: string, data?: any): void;
    error(message: string, error?: Error | any): void;
    debug(message: string, data?: any): void;
}
export declare function validateConfig(config: any, requiredFields: string[]): ValidationResult;
//# sourceMappingURL=utils.d.ts.map
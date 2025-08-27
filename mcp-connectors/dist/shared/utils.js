/**
 * Shared utilities for MCP advertising connectors
 */
// Rate limiting utilities
export class RateLimiter {
    requests = [];
    maxRequests;
    windowMs;
    constructor(maxRequests, windowMs = 1000) {
        this.maxRequests = maxRequests;
        this.windowMs = windowMs;
    }
    async waitIfNeeded() {
        const now = Date.now();
        // Remove old requests outside the window
        this.requests = this.requests.filter(time => now - time < this.windowMs);
        if (this.requests.length >= this.maxRequests) {
            const oldestRequest = Math.min(...this.requests);
            const waitTime = this.windowMs - (now - oldestRequest);
            if (waitTime > 0) {
                await new Promise(resolve => setTimeout(resolve, waitTime));
            }
        }
        this.requests.push(now);
    }
    getRemainingRequests() {
        const now = Date.now();
        this.requests = this.requests.filter(time => now - time < this.windowMs);
        return Math.max(0, this.maxRequests - this.requests.length);
    }
}
// Retry utilities
export async function retryWithBackoff(fn, maxRetries = 3, baseDelay = 1000) {
    let lastError;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            return await fn();
        }
        catch (error) {
            lastError = error;
            if (attempt === maxRetries) {
                throw lastError;
            }
            // Check if error is retryable
            if (error instanceof ConnectorError) {
                if (error.statusCode === 429) {
                    // Rate limited - wait longer
                    const delay = error.retryAfter ? error.retryAfter * 1000 : baseDelay * Math.pow(2, attempt);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                }
                if (error.statusCode && error.statusCode >= 400 && error.statusCode < 500) {
                    // Client error - don't retry
                    throw error;
                }
            }
            // Exponential backoff for other errors
            const delay = baseDelay * Math.pow(2, attempt);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    throw lastError;
}
// Validation utilities
export function validateCampaignName(name) {
    const errors = [];
    const warnings = [];
    if (!name || name.trim().length === 0) {
        errors.push('Campaign name is required');
    }
    else {
        if (name.length > 100) {
            errors.push('Campaign name must be 100 characters or less');
        }
        if (name.length < 5) {
            warnings.push('Campaign name should be at least 5 characters for clarity');
        }
        // Check for special characters that might cause issues
        const invalidChars = /[<>\"'&]/;
        if (invalidChars.test(name)) {
            warnings.push('Campaign name contains special characters that might cause display issues');
        }
    }
    return {
        valid: errors.length === 0,
        errors,
        warnings
    };
}
export function validateBudget(amount, type) {
    const errors = [];
    const warnings = [];
    if (amount <= 0) {
        errors.push('Budget amount must be greater than 0');
    }
    else {
        if (type === 'DAILY' && amount < 1) {
            warnings.push('Daily budget below $1 may result in limited delivery');
        }
        if (type === 'LIFETIME' && amount < 10) {
            warnings.push('Lifetime budget below $10 may result in limited delivery');
        }
        if (amount > 100000) {
            warnings.push('Large budget amount - ensure this is intentional');
        }
    }
    return {
        valid: errors.length === 0,
        errors,
        warnings
    };
}
export function validateTargeting(targeting) {
    const errors = [];
    const warnings = [];
    // Age validation
    if (targeting.demographics?.ageMin && targeting.demographics?.ageMax) {
        if (targeting.demographics.ageMin > targeting.demographics.ageMax) {
            errors.push('Minimum age cannot be greater than maximum age');
        }
        if (targeting.demographics.ageMax - targeting.demographics.ageMin > 40) {
            warnings.push('Large age range may reduce targeting effectiveness');
        }
    }
    // Location validation
    if (!targeting.locations?.countries?.length &&
        !targeting.locations?.regions?.length &&
        !targeting.locations?.cities?.length) {
        errors.push('At least one location target is required');
    }
    // Audience size estimation (simplified)
    const estimatedReach = estimateAudienceSize(targeting);
    if (estimatedReach < 1000) {
        warnings.push('Targeting may be too narrow - consider expanding audience');
    }
    else if (estimatedReach > 50000000) {
        warnings.push('Targeting may be too broad - consider narrowing audience');
    }
    return {
        valid: errors.length === 0,
        errors,
        warnings
    };
}
function estimateAudienceSize(targeting) {
    // Simplified audience size estimation
    let baseSize = 1000000; // Start with 1M as base
    // Age range factor
    if (targeting.demographics?.ageMin && targeting.demographics?.ageMax) {
        const ageRange = targeting.demographics.ageMax - targeting.demographics.ageMin;
        baseSize *= (ageRange / 50); // Normalize to 50-year range
    }
    // Gender factor
    if (targeting.demographics?.genders?.length === 1) {
        baseSize *= 0.5; // Roughly half for single gender
    }
    // Interest targeting factor
    if (targeting.interests?.length > 0) {
        baseSize *= Math.max(0.1, 1 - (targeting.interests.length * 0.1));
    }
    return Math.max(100, Math.floor(baseSize));
}
// Security utilities
export function sanitizeInput(input) {
    return input
        .replace(/[<>\"'&]/g, '') // Remove potentially dangerous characters
        .trim()
        .substring(0, 1000); // Limit length
}
export function maskSensitiveData(data) {
    const masked = { ...data };
    // Mask common sensitive fields
    const sensitiveFields = ['accessToken', 'refreshToken', 'apiKey', 'clientSecret', 'password'];
    for (const field of sensitiveFields) {
        if (masked[field]) {
            masked[field] = '*'.repeat(8);
        }
    }
    return masked;
}
// URL utilities
export function buildQueryString(params) {
    const searchParams = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== null) {
            if (Array.isArray(value)) {
                value.forEach(v => searchParams.append(key, String(v)));
            }
            else {
                searchParams.append(key, String(value));
            }
        }
    }
    return searchParams.toString();
}
// Date utilities
export function formatDate(date) {
    return date.toISOString().split('T')[0];
}
export function parseDate(dateString) {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) {
        throw new Error(`Invalid date format: ${dateString}`);
    }
    return date;
}
export function getDateRange(days) {
    const end = new Date();
    const start = new Date();
    start.setDate(end.getDate() - days);
    return {
        start: formatDate(start),
        end: formatDate(end)
    };
}
// Logging utilities
export class Logger {
    context;
    constructor(context) {
        this.context = context;
    }
    info(message, data) {
        console.log(`[${this.context}] INFO: ${message}`, data ? JSON.stringify(maskSensitiveData(data)) : '');
    }
    warn(message, data) {
        console.warn(`[${this.context}] WARN: ${message}`, data ? JSON.stringify(maskSensitiveData(data)) : '');
    }
    error(message, error) {
        console.error(`[${this.context}] ERROR: ${message}`, error?.stack || error);
    }
    debug(message, data) {
        if (process.env.DEBUG === 'true') {
            console.debug(`[${this.context}] DEBUG: ${message}`, data ? JSON.stringify(maskSensitiveData(data)) : '');
        }
    }
}
// Configuration validation
export function validateConfig(config, requiredFields) {
    const errors = [];
    for (const field of requiredFields) {
        if (!config[field]) {
            errors.push(`Missing required configuration field: ${field}`);
        }
    }
    // Validate rate limiting settings
    if (config.rateLimitPerSecond && (config.rateLimitPerSecond < 1 || config.rateLimitPerSecond > 100)) {
        errors.push('rateLimitPerSecond must be between 1 and 100');
    }
    // Validate retry settings
    if (config.retryAttempts && (config.retryAttempts < 0 || config.retryAttempts > 10)) {
        errors.push('retryAttempts must be between 0 and 10');
    }
    // Validate timeout
    if (config.timeout && (config.timeout < 1000 || config.timeout > 60000)) {
        errors.push('timeout must be between 1000ms and 60000ms');
    }
    return {
        valid: errors.length === 0,
        errors,
        warnings: []
    };
}
//# sourceMappingURL=utils.js.map
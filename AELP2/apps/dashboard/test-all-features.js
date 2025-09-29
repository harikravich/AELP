#!/usr/bin/env node

/**
 * AELP2 Dashboard Feature Test Suite
 * Tests all the fixed UI features end-to-end
 */

const http = require('http');
const https = require('https');

const API_BASE = 'http://localhost:8080/api';

const tests = {
  // Creative Generation Tests
  'Creative Generation API': async () => {
    const types = [
      'rsa-text', 'display-banner', 'video-script',
      'product-screenshot', 'social-proof', 'data-viz',
      'image-assets', 'demo-video', 'testimonial',
      'explainer', 'headlines', 'descriptions'
    ];

    for (const type of types) {
      const response = await fetch(`${API_BASE}/creative/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type,
          product: 'Test Product',
          campaign: 'Test Campaign',
          audience: 'Test Audience',
          tone: 'professional'
        })
      });

      if (!response.ok) {
        throw new Error(`Creative generation failed for ${type}: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(`Creative generation returned error for ${type}`);
      }
      console.log(`✓ Creative generation works for: ${type}`);
    }
  },

  // Executive Report Test
  'Executive Report Download': async () => {
    const response = await fetch(
      `${API_BASE}/reports/executive?startDate=2025-08-01&endDate=2025-09-01`
    );

    if (!response.ok) {
      throw new Error(`Executive report generation failed: ${response.status}`);
    }

    const contentType = response.headers.get('content-type');
    if (!contentType?.includes('text/html')) {
      throw new Error(`Executive report wrong content type: ${contentType}`);
    }

    console.log('✓ Executive report download works');
  },

  // Training System Test
  'Training System Connection': async () => {
    const response = await fetch(`${API_BASE}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modelType: 'rl',
        episodes: 10,
        batchSize: 32,
        learningRate: 0.001,
        useRecSim: true,
        useAuctionGym: true
      })
    });

    // We expect this might fail if the orchestrator isn't set up,
    // but the endpoint should at least exist
    if (response.status === 404) {
      throw new Error('Training endpoint not found');
    }

    console.log('✓ Training endpoint exists and responds');
  },

  // Scenario Modeling Test
  'Scenario Modeling': async () => {
    const scenarios = ['budget', 'bidding', 'audience', 'creative', 'channel'];

    for (const type of scenarios) {
      const response = await fetch(`${API_BASE}/scenarios/model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type,
          changes: {
            budgetChange: 20,
            biddingStrategy: 'MAXIMIZE_CONVERSIONS',
            audienceSegments: ['test_segment'],
            creativeVariants: 3,
            channelMix: { google: 50, facebook: 30, direct: 20 }
          },
          timeHorizon: 30
        })
      });

      if (!response.ok) {
        throw new Error(`Scenario modeling failed for ${type}: ${response.status}`);
      }

      const data = await response.json();
      if (!data.scenario) {
        throw new Error(`Scenario modeling returned no data for ${type}`);
      }
      console.log(`✓ Scenario modeling works for: ${type}`);
    }
  },

  // Queue Processor Test
  'Queue Processing': async () => {
    // Add a job to queue
    const response = await fetch(`${API_BASE}/queue/processor`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        queueName: 'queue:creative:generate',
        job: {
          type: 'test',
          data: { test: true }
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Queue processor failed: ${response.status}`);
    }

    const data = await response.json();
    if (!data.jobId) {
      throw new Error('Queue processor returned no job ID');
    }

    // Check queue status
    const statusResponse = await fetch(`${API_BASE}/queue/processor?jobId=${data.jobId}`);
    if (!statusResponse.ok) {
      throw new Error(`Queue status check failed: ${statusResponse.status}`);
    }

    console.log('✓ Queue processor works');
  },

  // Google Ads Publishing Test (mock)
  'Creative Publishing API': async () => {
    const response = await fetch(`${API_BASE}/creative/publish`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        campaignId: 'test_campaign',
        adGroupId: 'test_adgroup',
        headlines: ['Test Headline 1', 'Test Headline 2'],
        descriptions: ['Test Description 1', 'Test Description 2'],
        finalUrls: ['https://example.com'],
        action: 'create'
      })
    });

    // This will likely fail without real Google Ads credentials,
    // but we're checking the endpoint exists
    if (response.status === 404) {
      throw new Error('Publishing endpoint not found');
    }

    console.log('✓ Publishing endpoint exists');
  }
};

// Run all tests
async function runTests() {
  console.log('🚀 Starting AELP2 Dashboard Feature Tests\n');
  console.log('=' .repeat(50));

  let passed = 0;
  let failed = 0;
  const failures = [];

  for (const [name, test] of Object.entries(tests)) {
    process.stdout.write(`Testing ${name}... `);
    try {
      await test();
      passed++;
    } catch (error) {
      console.log(`✗ Failed: ${error.message}`);
      failed++;
      failures.push({ name, error: error.message });
    }
  }

  console.log('\n' + '=' .repeat(50));
  console.log('\n📊 Test Results:');
  console.log(`   ✅ Passed: ${passed}`);
  console.log(`   ❌ Failed: ${failed}`);
  console.log(`   Total: ${passed + failed}`);

  if (failures.length > 0) {
    console.log('\n❌ Failed Tests:');
    failures.forEach(({ name, error }) => {
      console.log(`   - ${name}: ${error}`);
    });
  }

  console.log('\n' + '=' .repeat(50));

  if (failed === 0) {
    console.log('\n🎉 All tests passed! Dashboard features are working correctly.');
  } else {
    console.log(`\n⚠️  ${failed} tests failed. Please review and fix the issues.`);
  }

  process.exit(failed > 0 ? 1 : 0);
}

// Check if server is running
function checkServer() {
  return new Promise((resolve) => {
    http.get('http://localhost:8080', (res) => {
      resolve(res.statusCode === 200 || res.statusCode === 404);
    }).on('error', () => {
      resolve(false);
    });
  });
}

// Main execution
async function main() {
  console.log('Checking if Next.js server is running on localhost:8080...');

  const serverRunning = await checkServer();
  if (!serverRunning) {
    console.error('❌ Server not running! Please start the server with: npm run dev');
    console.error('   Run from: /home/hariravichandran/AELP/AELP2/apps/dashboard');
    process.exit(1);
  }

  console.log('✓ Server is running\n');

  // Add fetch polyfill for Node.js < 18
  if (!global.fetch) {
    const https = require('https');
    const http = require('http');

    global.fetch = (url, options = {}) => {
      return new Promise((resolve, reject) => {
        const parsedUrl = new URL(url);
        const client = parsedUrl.protocol === 'https:' ? https : http;

        const req = client.request(parsedUrl, {
          method: options.method || 'GET',
          headers: options.headers || {}
        }, (res) => {
          let data = '';
          res.on('data', chunk => data += chunk);
          res.on('end', () => {
            resolve({
              ok: res.statusCode >= 200 && res.statusCode < 300,
              status: res.statusCode,
              headers: {
                get: (name) => res.headers[name.toLowerCase()]
              },
              json: () => Promise.resolve(JSON.parse(data)),
              text: () => Promise.resolve(data)
            });
          });
        });

        req.on('error', reject);
        if (options.body) {
          req.write(options.body);
        }
        req.end();
      });
    };
  }

  await runTests();
}

main().catch(console.error);
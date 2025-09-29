import { NextRequest, NextResponse } from 'next/server'
import { createClient } from 'redis'
import { BigQuery } from '@google-cloud/bigquery'

// Redis client for queue management
const redis = createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379',
})

redis.on('error', (err) => console.error('Redis Client Error', err))

// BigQuery client
const bigquery = new BigQuery({
  projectId: process.env.GOOGLE_CLOUD_PROJECT,
})

// Queue names
const QUEUES = {
  CREATIVE: 'queue:creative:generate',
  PUBLISH: 'queue:creative:publish',
  TRAINING: 'queue:training:jobs',
  REPORT: 'queue:report:generate',
  OPTIMIZATION: 'queue:optimization:jobs',
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const { queueName, job } = body

    if (!queueName || !job) {
      return NextResponse.json(
        { error: 'Queue name and job data are required' },
        { status: 400 }
      )
    }

    // Connect to Redis
    if (!redis.isOpen) {
      await redis.connect()
    }

    // Add job to queue with timestamp and ID
    const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const jobData = {
      id: jobId,
      ...job,
      status: 'pending',
      createdAt: new Date().toISOString(),
      attempts: 0,
    }

    // Push to Redis queue
    await redis.lPush(queueName, JSON.stringify(jobData))

    // Log to BigQuery
    await logJobCreation(queueName, jobData)

    // Start processing if not already running
    processQueue(queueName)

    return NextResponse.json({
      success: true,
      jobId,
      queueName,
      position: await redis.lLen(queueName),
    })
  } catch (error: any) {
    console.error('Queue processor error:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to process queue job' },
      { status: 500 }
    )
  }
}

export async function GET(req: NextRequest) {
  try {
    const searchParams = req.nextUrl.searchParams
    const queueName = searchParams.get('queue')
    const jobId = searchParams.get('jobId')

    if (!redis.isOpen) {
      await redis.connect()
    }

    if (jobId) {
      // Get specific job status
      const job = await getJobStatus(jobId)
      return NextResponse.json(job)
    }

    if (queueName) {
      // Get queue status
      const length = await redis.lLen(queueName)
      const jobs = await redis.lRange(queueName, 0, 9) // Get first 10 jobs
      return NextResponse.json({
        queue: queueName,
        length,
        jobs: jobs.map((j) => JSON.parse(j)),
      })
    }

    // Get all queues status
    const status: any = {}
    for (const [name, key] of Object.entries(QUEUES)) {
      status[name] = {
        length: await redis.lLen(key),
      }
    }

    return NextResponse.json(status)
  } catch (error: any) {
    console.error('Queue status error:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to get queue status' },
      { status: 500 }
    )
  }
}

// Process jobs from queue
async function processQueue(queueName: string) {
  if (!redis.isOpen) {
    await redis.connect()
  }

  // Check if processor is already running
  const lockKey = `lock:${queueName}`
  const lock = await redis.setNX(lockKey, '1')
  if (!lock) {
    console.log(`Processor already running for ${queueName}`)
    return
  }

  // Set lock expiry (5 minutes)
  await redis.expire(lockKey, 300)

  try {
    while (true) {
      // Get next job from queue
      const jobStr = await redis.rPop(queueName)
      if (!jobStr) {
        break // No more jobs
      }

      const job = JSON.parse(jobStr)
      console.log(`Processing job ${job.id} from ${queueName}`)

      try {
        // Process based on queue type
        let result
        switch (queueName) {
          case QUEUES.CREATIVE:
            result = await processCreativeJob(job)
            break
          case QUEUES.PUBLISH:
            result = await processPublishJob(job)
            break
          case QUEUES.TRAINING:
            result = await processTrainingJob(job)
            break
          case QUEUES.REPORT:
            result = await processReportJob(job)
            break
          case QUEUES.OPTIMIZATION:
            result = await processOptimizationJob(job)
            break
          default:
            throw new Error(`Unknown queue: ${queueName}`)
        }

        // Update job status
        await updateJobStatus(job.id, 'completed', result)
      } catch (error: any) {
        console.error(`Job ${job.id} failed:`, error)
        job.attempts++

        if (job.attempts < 3) {
          // Retry job
          await redis.lPush(queueName, JSON.stringify(job))
        } else {
          // Move to dead letter queue
          await redis.lPush(`${queueName}:failed`, JSON.stringify({
            ...job,
            error: error.message,
            failedAt: new Date().toISOString(),
          }))
          await updateJobStatus(job.id, 'failed', { error: error.message })
        }
      }
    }
  } finally {
    // Release lock
    await redis.del(lockKey)
  }
}

async function processCreativeJob(job: any) {
  // Generate creative using Anthropic API
  const response = await fetch(`${process.env.NEXT_PUBLIC_URL}/api/creative/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(job.data),
  })
  return await response.json()
}

async function processPublishJob(job: any) {
  // Publish to Google Ads
  const response = await fetch(`${process.env.NEXT_PUBLIC_URL}/api/creative/publish`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(job.data),
  })
  return await response.json()
}

async function processTrainingJob(job: any) {
  // Start training
  const response = await fetch(`${process.env.NEXT_PUBLIC_URL}/api/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(job.data),
  })
  return await response.json()
}

async function processReportJob(job: any) {
  // Generate report
  const response = await fetch(`${process.env.NEXT_PUBLIC_URL}/api/reports/executive`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  })
  return { generated: true, size: response.headers.get('content-length') }
}

async function processOptimizationJob(job: any) {
  // Run optimization
  // This would call the actual optimization logic
  return {
    optimized: true,
    improvements: job.data.expectedImprovements || 0.15,
  }
}

async function logJobCreation(queueName: string, job: any) {
  try {
    const dataset = bigquery.dataset(process.env.BIGQUERY_TRAINING_DATASET!)
    const table = dataset.table('queue_jobs')

    await table.insert({
      job_id: job.id,
      queue_name: queueName,
      job_data: JSON.stringify(job),
      status: 'pending',
      created_at: job.createdAt,
    })
  } catch (error) {
    console.error('Failed to log job creation:', error)
  }
}

async function updateJobStatus(jobId: string, status: string, result?: any) {
  try {
    const dataset = bigquery.dataset(process.env.BIGQUERY_TRAINING_DATASET!)
    const table = dataset.table('queue_jobs')

    // Update job status in BigQuery
    const query = `
      UPDATE \`${process.env.GOOGLE_CLOUD_PROJECT}.${process.env.BIGQUERY_TRAINING_DATASET}.queue_jobs\`
      SET
        status = @status,
        result = @result,
        updated_at = CURRENT_TIMESTAMP()
      WHERE job_id = @jobId
    `

    const options = {
      query,
      params: {
        status,
        result: result ? JSON.stringify(result) : null,
        jobId,
      },
    }

    await bigquery.query(options)

    // Store result in Redis for quick retrieval
    if (redis.isOpen) {
      await redis.setEx(
        `job:${jobId}`,
        3600, // 1 hour TTL
        JSON.stringify({ status, result, updatedAt: new Date().toISOString() })
      )
    }
  } catch (error) {
    console.error('Failed to update job status:', error)
  }
}

async function getJobStatus(jobId: string) {
  // Try Redis first
  if (redis.isOpen) {
    const cached = await redis.get(`job:${jobId}`)
    if (cached) {
      return JSON.parse(cached)
    }
  }

  // Fall back to BigQuery
  const query = `
    SELECT *
    FROM \`${process.env.GOOGLE_CLOUD_PROJECT}.${process.env.BIGQUERY_TRAINING_DATASET}.queue_jobs\`
    WHERE job_id = @jobId
    LIMIT 1
  `

  const [rows] = await bigquery.query({
    query,
    params: { jobId },
  })

  return rows[0] || { error: 'Job not found' }
}
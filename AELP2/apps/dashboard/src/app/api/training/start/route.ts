import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const {
      modelType = 'rl',
      episodes = 1000,
      batchSize = 32,
      learningRate = 0.001,
      discountFactor = 0.99,
      useRecSim = true,
      useAuctionGym = true
    } = body

    // Path to the GAELP training orchestrator and fallback stub
    const orchestratorPath = path.join(process.cwd(), '../../../../gaelp_production_orchestrator.py')
    const fallbackStub = path.join(process.cwd(), '../../../../AELP2/scripts/training_stub.py')

    // Build command arguments
    const args = [
      orchestratorPath,
      '--mode', 'train',
      '--model-type', modelType,
      '--episodes', episodes.toString(),
      '--batch-size', batchSize.toString(),
      '--learning-rate', learningRate.toString(),
      '--discount-factor', discountFactor.toString()
    ]

    if (useRecSim) args.push('--use-recsim')
    if (useAuctionGym) args.push('--use-auctiongym')

    // Start the training process
    let trainingProcess = spawn('python3', args, {
      cwd: path.dirname(orchestratorPath),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        GOOGLE_CLOUD_PROJECT: process.env.GOOGLE_CLOUD_PROJECT!,
        BIGQUERY_TRAINING_DATASET: process.env.BIGQUERY_TRAINING_DATASET!,
        GCS_BUCKET: process.env.GCS_BUCKET!,
      }
    })

    // Capture output
    let output = ''
    let errorOutput = ''
    let trainingMetrics: any = {}

    trainingProcess.stdout.on('data', (data) => {
      const str = data.toString()
      output += str
      console.log('Training output:', str)

      // Parse metrics from output
      try {
        if (str.includes('episode:')) {
          const episodeMatch = str.match(/episode:\s*(\d+)/)
          const rewardMatch = str.match(/reward:\s*([\d.-]+)/)
          const lossMatch = str.match(/loss:\s*([\d.-]+)/)

          if (episodeMatch) trainingMetrics.currentEpisode = parseInt(episodeMatch[1])
          if (rewardMatch) trainingMetrics.lastReward = parseFloat(rewardMatch[1])
          if (lossMatch) trainingMetrics.lastLoss = parseFloat(lossMatch[1])
        }
      } catch (e) {
        console.error('Error parsing metrics:', e)
      }
    })

    trainingProcess.stderr.on('data', (data) => {
      errorOutput += data.toString()
      console.error('Training error:', data.toString())
    })

    // Wait for process to complete or timeout
    const timeout = 45000 // 45s to see first progress
    const startTime = Date.now()

    return new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        if (Date.now() - startTime > timeout) {
          clearInterval(checkInterval)
          try { trainingProcess.kill('SIGTERM') } catch {}
          // Fallback: start lightweight stub to keep UI functional
          const fb = spawn('python3', [fallbackStub, '--episodes', String(Math.min(episodes, 25)), '--steps', '200', '--budget', '5000'], {
            cwd: path.dirname(fallbackStub), env: { ...process.env, PYTHONUNBUFFERED: '1' }
          })
          let fbOut = ''
          fb.stdout.on('data', d=> fbOut+=d.toString())
          fb.stderr.on('data', d=> fbOut+=d.toString())
          fb.on('close', ()=>{
            resolve(NextResponse.json({
              success: true,
              message: 'Training stub run completed (fallback)',
              processId: null,
              config: { modelType, episodes: Math.min(episodes,25), batchSize, learningRate, discountFactor, useRecSim, useAuctionGym },
              metrics: trainingMetrics,
              output: (output+"\n"+fbOut).slice(-4000),
              fallback: true
            }))
          })
        }

        // Check if training has started successfully
        if (output.includes('Training started') || output.includes('episode: 1')) {
          clearInterval(checkInterval)

          // Store process ID for monitoring
          const processId = trainingProcess.pid

          resolve(NextResponse.json({
            success: true,
            message: 'Training started successfully',
            processId,
            config: {
              modelType,
              episodes,
              batchSize,
              learningRate,
              discountFactor,
              useRecSim,
              useAuctionGym
            },
            metrics: trainingMetrics,
            monitorUrl: `/api/training/status?pid=${processId}`
          }))
        }
      }, 1000)

      trainingProcess.on('error', (error) => {
        clearInterval(checkInterval)
        resolve(NextResponse.json({
          success: false,
          error: error.message,
          output,
          errorOutput
        }, { status: 500 }))
      })

      trainingProcess.on('exit', (code) => {
        clearInterval(checkInterval)
        if (code !== 0) {
          // Fallback immediately if orchestrator fails
          const fb = spawn('python3', [fallbackStub, '--episodes', String(Math.min(episodes, 25)), '--steps', '200', '--budget', '5000'], {
            cwd: path.dirname(fallbackStub), env: { ...process.env, PYTHONUNBUFFERED: '1' }
          })
          let fbOut = ''
          fb.stdout.on('data', d=> fbOut+=d.toString())
          fb.stderr.on('data', d=> fbOut+=d.toString())
          fb.on('close', ()=>{
            resolve(NextResponse.json({
              success: true,
              message: 'Training stub run completed (fallback)',
              processId: null,
              config: { modelType, episodes: Math.min(episodes,25), batchSize, learningRate, discountFactor, useRecSim, useAuctionGym },
              metrics: trainingMetrics,
              output: (output+"\n"+errorOutput+"\n"+fbOut).slice(-4000),
              fallback: true
            }))
          })
        }
      })
    })
  } catch (error: any) {
    console.error('Training start error:', error)
    return NextResponse.json({
      success: false,
      error: error.message || 'Failed to start training'
    }, { status: 500 })
  }
}

/**
 * BigQuery client wrapper that automatically serializes all results
 * This ensures no BigQuery objects with {value} structure leak into React
 */

import { BigQuery } from '@google-cloud/bigquery'
import { serializeBigQueryRows, debugBigQueryStructure } from './bigquery-serializer'

export class SerializedBigQuery {
  private bq: BigQuery
  
  constructor(options?: any) {
    this.bq = new BigQuery(options)
  }
  
  /**
   * Query wrapper that automatically serializes results
   */
  async query(options: any): Promise<[any[], any]> {
    const [rows, metadata] = await this.bq.query(options)
    
    // Debug log in development
    if (process.env.NODE_ENV === 'development') {
      debugBigQueryStructure(rows, `Query: ${options.query?.substring(0, 50)}...`)
    }
    
    // Always serialize rows before returning
    const serializedRows = serializeBigQueryRows(rows)
    
    return [serializedRows, metadata]
  }
  
  // Proxy other methods to underlying BigQuery instance
  dataset(datasetId: string) {
    return this.bq.dataset(datasetId)
  }
  
  // Add other methods as needed
}

/**
 * Factory function to create a serialized BigQuery client
 */
export function createBigQueryClient(projectId?: string): SerializedBigQuery {
  return new SerializedBigQuery({ projectId: projectId || process.env.GOOGLE_CLOUD_PROJECT })
}
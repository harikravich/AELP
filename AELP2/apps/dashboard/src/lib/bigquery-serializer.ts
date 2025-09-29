/**
 * Comprehensive BigQuery object serialization utility
 * Handles all BigQuery timestamp and nested objects
 */

export function serializeBigQueryValue(value: any): any {
  // Handle null/undefined
  if (value == null) return value;
  
  // Handle primitives
  if (typeof value !== 'object') return value;
  
  // Handle BigQuery objects with { value } structure
  if ('value' in value && Object.keys(value).length === 1) {
    return serializeBigQueryValue(value.value);
  }
  
  // Handle Date objects
  if (value instanceof Date) {
    return value.toISOString();
  }
  
  // Handle arrays
  if (Array.isArray(value)) {
    return value.map(item => serializeBigQueryValue(item));
  }
  
  // Handle regular objects recursively
  const serialized: any = {};
  for (const key in value) {
    if (value.hasOwnProperty(key)) {
      serialized[key] = serializeBigQueryValue(value[key]);
    }
  }
  
  return serialized;
}

export function serializeBigQueryRows(rows: any[]): any[] {
  if (!Array.isArray(rows)) return [];
  
  return rows.map(row => {
    if (typeof row !== 'object' || row == null) return row;
    
    const serialized: any = {};
    for (const key in row) {
      if (row.hasOwnProperty(key)) {
        serialized[key] = serializeBigQueryValue(row[key]);
      }
    }
    return serialized;
  });
}

// Helper to safely serialize any BigQuery result
export function serializeBigQueryResult(result: any): any {
  if (result == null) return result;
  
  // Handle array results (most common)
  if (Array.isArray(result)) {
    return serializeBigQueryRows(result);
  }
  
  // Handle single object results
  if (typeof result === 'object') {
    return serializeBigQueryValue(result);
  }
  
  return result;
}

// Debug helper to log structure before serialization
export function debugBigQueryStructure(data: any, label: string = 'Data'): void {
  console.log(`[BQ Debug] ${label}:`, {
    type: typeof data,
    isArray: Array.isArray(data),
    keys: data && typeof data === 'object' ? Object.keys(data).slice(0, 5) : null,
    sample: data && Array.isArray(data) && data.length > 0 ? {
      firstRow: data[0],
      firstRowKeys: Object.keys(data[0] || {}),
      firstRowTypes: Object.entries(data[0] || {}).reduce((acc, [k, v]) => {
        acc[k] = typeof v === 'object' && v !== null ? 
          (v.hasOwnProperty('value') ? 'BQ_OBJECT' : 'object') : 
          typeof v;
        return acc;
      }, {} as any)
    } : null
  });
}
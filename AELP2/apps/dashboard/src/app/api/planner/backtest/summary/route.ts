import { NextResponse } from 'next/server';
import fs from 'node:fs/promises';

function reportsDir() {
  return process.env.AELP_REPORTS_DIR || `${process.cwd()}/AELP2/reports`;
}

export async function GET() {
  const p = reportsDir();
  try {
    const [sumBuf, ciBuf] = await Promise.all([
      fs.readFile(`${p}/auction_backtest_summary.json`).catch(() => Buffer.from('{}')),
      fs.readFile(`${p}/auction_backtest_precision_ci.json`).catch(() => Buffer.from('{}')),
    ]);
    const summary = JSON.parse(sumBuf.toString() || '{}');
    const ci = JSON.parse(ciBuf.toString() || '{}');
    return NextResponse.json({ summary, ci });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || 'failed' }, { status: 500 });
  }
}


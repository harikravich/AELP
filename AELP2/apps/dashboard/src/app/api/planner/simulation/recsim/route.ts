import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';

export async function GET() {
  try {
    const p = process.env.AELP_REPORTS_DIR || process.cwd() + '/AELP2/reports';
    const buf = await fs.readFile(`${p}/recsim_offline_simulation.json`);
    return new NextResponse(buf, { status: 200, headers: { 'content-type': 'application/json' } });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || 'not found' }, { status: 404 });
  }
}


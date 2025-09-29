import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useEffect, useState } from "react";

type Row = {
  date: string;
  campaign_id: string;
  name: string | null;
  impressions: number;
  clicks: number;
  spend: number;
  conversions: number;
  impression_share: number | null;
  lost_is_rank: number | null;
  lost_is_budget: number | null;
  ctr: number;
  cvr: number;
  cac: number | null;
  brand: boolean;
  alerts: { type: string; severity: string }[];
  suggestions: string[];
};

export default function QS() {
  const [rows, setRows] = useState<Row[]>([]);
  const [summary, setSummary] = useState<any>();
  const [issues, setIssues] = useState<any>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | undefined>();
  const [days, setDays] = useState<string>('14')
  const [group, setGroup] = useState<'campaign'|'daily'>('campaign')
  const [brand, setBrand] = useState<'all'|'brand'|'nonbrand'>('all')
  const [channel, setChannel] = useState<'ALL'|'SEARCH'>('SEARCH')
  const [minSpend, setMinSpend] = useState<string>('0')

  const load = () => {
    setLoading(true)
    const base = (import.meta as any).env.VITE_API_BASE_URL || ''
    const params = new URLSearchParams({
      days, group, brand, channel, min_spend: String(Number(minSpend)||0), excl_today: '1'
    })
    fetch(`${base}/api/bq/ads/qs-is?` + params.toString(), { credentials:'include' })
      .then((r) => r.json())
      .then((d) => {
        if (d.error) setError(d.error);
        else {
          setRows(d.rows || []);
          setSummary(d.summary || undefined)
          setIssues(d.issues || undefined)
        }
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }
  useEffect(() => {
    load()
  }, []);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <h1 className="text-3xl font-bold text-foreground">Auction Health (QS/IS)</h1>
          <div className="flex items-center gap-2">
            <Select value={days} onValueChange={setDays}>
              <SelectTrigger className="w-[110px]"><SelectValue placeholder="Days"/></SelectTrigger>
              <SelectContent>
                <SelectItem value="7">7 days</SelectItem>
                <SelectItem value="14">14 days</SelectItem>
                <SelectItem value="28">28 days</SelectItem>
                <SelectItem value="60">60 days</SelectItem>
                <SelectItem value="90">90 days</SelectItem>
              </SelectContent>
            </Select>
            <Select value={group} onValueChange={(v:any)=>setGroup(v)}>
              <SelectTrigger className="w-[130px]"><SelectValue placeholder="Group"/></SelectTrigger>
              <SelectContent>
                <SelectItem value="campaign">By campaign</SelectItem>
                <SelectItem value="daily">Daily rows</SelectItem>
              </SelectContent>
            </Select>
            <Select value={channel} onValueChange={(v:any)=>setChannel(v)}>
              <SelectTrigger className="w-[120px]"><SelectValue placeholder="Channel"/></SelectTrigger>
              <SelectContent>
                <SelectItem value="SEARCH">Search</SelectItem>
                <SelectItem value="ALL">All</SelectItem>
              </SelectContent>
            </Select>
            <Select value={brand} onValueChange={(v:any)=>setBrand(v)}>
              <SelectTrigger className="w-[140px]"><SelectValue placeholder="Brand"/></SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="brand">Brand only</SelectItem>
                <SelectItem value="nonbrand">Non‑brand</SelectItem>
              </SelectContent>
            </Select>
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Min spend</span>
              <Input value={minSpend} onChange={e=>setMinSpend(e.target.value)} className="w-24" placeholder="$" />
            </div>
            <Button onClick={load} variant="default">Apply</Button>
          </div>
        </div>
        {summary && (
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            <Card className="p-4"><div className="text-xs text-muted-foreground">Total Spend</div><div className="text-2xl font-semibold">${Math.round(summary.total_spend||0).toLocaleString('en-US')}</div></Card>
            <Card className="p-4"><div className="text-xs text-muted-foreground">Conversions</div><div className="text-2xl font-semibold">{Math.round(summary.total_conversions||0).toLocaleString('en-US')}</div></Card>
            <Card className="p-4"><div className="text-xs text-muted-foreground">Avg CAC</div><div className="text-2xl font-semibold">{summary.avg_cac!=null?`$${Math.round(summary.avg_cac).toLocaleString('en-US')}`:'—'}</div></Card>
            <Card className="p-4"><div className="text-xs text-muted-foreground">Zero‑conv Spend</div><div className="text-2xl font-semibold">{summary.zero_conv_spend_share!=null? (summary.zero_conv_spend_share*100).toFixed(1)+'%':'—'}</div></Card>
            <Card className="p-4"><div className="text-xs text-muted-foreground">Rank‑limited Spend</div><div className="text-2xl font-semibold">{summary.rank_limited_spend_share!=null? (summary.rank_limited_spend_share*100).toFixed(1)+'%':'—'}</div></Card>
          </div>
        )}

        {issues && (
          <Card className="p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Top Opportunities</h2>
              <Badge variant="outline">{summary?.days || days} days • {group} • {channel} • {brand}</Badge>
            </div>
            <Tabs defaultValue="rank" className="w-full">
              <TabsList>
                <TabsTrigger value="rank">Fix Rank‑Limited</TabsTrigger>
                <TabsTrigger value="brand">Defend Brand</TabsTrigger>
                <TabsTrigger value="waste">Trim Waste</TabsTrigger>
                <TabsTrigger value="ctr">Improve CTR</TabsTrigger>
              </TabsList>
              <TabsContent value="rank">
                <IssueTable rows={(issues.rank_limited||[])} label="Rank‑limited (lost is rank > 60%)" />
              </TabsContent>
              <TabsContent value="brand">
                <IssueTable rows={(issues.brand_is_low||[])} label="Brand IS < 80%" />
              </TabsContent>
              <TabsContent value="waste">
                <IssueTable rows={(issues.zero_conv_spend||[])} label="Zero‑conversion spend" />
              </TabsContent>
              <TabsContent value="ctr">
                <IssueTable rows={(issues.low_ctr||[])} label="Low CTR (<2%)" />
              </TabsContent>
            </Tabs>
          </Card>
        )}

        <Card className="p-4 border">
          {loading ? (
            <div>Loading…</div>
          ) : error ? (
            <div className="text-destructive">{error}</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left border-b">
                    {group==='daily' && <th className="py-2">Date</th>}
                    <th>Campaign</th>
                    <th>Spend</th>
                    <th>Impr</th>
                    <th>CTR</th>
                    <th>CVR</th>
                    <th>CAC</th>
                    <th>IS</th>
                    <th>Lost IS (Rank)</th>
                    <th>Alerts</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(0, 300).map((r: any, i) => (
                    <tr key={i} className="border-b">
                      {group==='daily' && <td className="py-1">{r.date?.slice(0, 10)}</td>}
                      <td>{r.name || r.campaign_id}{r.brand ? ' · Brand' : ''}{group==='campaign' && r.days ? <span className="text-xs text-muted-foreground"> • {r.days}d</span>: null}</td>
                      <td>${Math.round(r.spend).toLocaleString('en-US')}</td>
                      <td>{r.impressions.toLocaleString('en-US')}</td>
                      <td>{(r.ctr * 100).toFixed(2)}%</td>
                      <td>{(r.cvr * 100).toFixed(2)}%</td>
                      <td>{r.cac != null ? `$${Math.round(r.cac).toLocaleString('en-US')}` : '—'}</td>
                      <td>{r.impression_share != null ? (r.impression_share * 100).toFixed(1) + '%' : '—'}</td>
                      <td>{r.lost_is_rank != null ? (r.lost_is_rank * 100).toFixed(1) + '%' : '—'}</td>
                      <td>
                        {r.alerts?.map((a, idx) => (
                          <span
                            key={idx}
                            className={`inline-block px-2 py-0.5 mr-1 rounded ${a.severity === 'high' ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-800'}`}
                          >
                            {a.type}
                          </span>
                        ))}
                        {r.suggestions?.length ? (
                          <div className="text-xs text-muted-foreground mt-1">{r.suggestions[0]}</div>
                        ) : null}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </DashboardLayout>
  );
}

function IssueTable({ rows, label }: { rows: Row[]; label: string }) {
  return (
    <div className="mt-3">
      <div className="text-sm text-muted-foreground mb-2">{label}</div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b">
              <th>Campaign</th>
              <th>Spend</th>
              <th>IS</th>
              <th>Lost IS (Rank)</th>
              <th>CAC</th>
              <th>Suggestion</th>
            </tr>
          </thead>
          <tbody>
            {(rows||[]).slice(0, 20).map((r: any, i: number) => (
              <tr key={i} className="border-b">
                <td>{r.name || r.campaign_id}{r.brand ? ' · Brand' : ''}</td>
                <td>${Math.round(r.spend).toLocaleString('en-US')}</td>
                <td>{r.impression_share != null ? (r.impression_share * 100).toFixed(1) + '%' : '—'}</td>
                <td>{r.lost_is_rank != null ? (r.lost_is_rank * 100).toFixed(1) + '%' : '—'}</td>
                <td>{r.cac != null ? `$${Math.round(r.cac).toLocaleString('en-US')}` : '—'}</td>
                <td className="text-xs text-muted-foreground">{r.suggestions?.[0] || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

import pandas as pd
u=pd.read_parquet('artifacts/marketing/unified_ctr.parquet')
u['date']=pd.to_datetime(u['date'])
maxd=u['date'].max()
u=u.sort_values(['ad_id','date'])
u['ctr_u']=u.get('link_ctr_unique', u.get('link_ctr', u['ctr']))
u['ctr_u_7d']=u.groupby('ad_id')['ctr_u'].transform(lambda s: s.rolling(window=7, min_periods=1).mean())
cols=['date','ad_id','impressions']
if 'unique_inline_link_clicks' in u.columns:
    cols+=['unique_inline_link_clicks']
cur=u[(u['date']==maxd)&(u['impressions']>0)][cols+['ctr_u','ctr_u_7d']].copy()
if 'unique_inline_link_clicks' in cur.columns:
    cur=cur.rename(columns={'unique_inline_link_clicks':'unique_link_clicks','ctr_u':'actual_ctr_u'})
else:
    cur=cur.rename(columns={'ctr_u':'actual_ctr_u'})
cur.to_parquet('artifacts/predictions/current_running_scored_unique.parquet', index=False)
print('today unique rows:', len(cur), 'max date:', maxd.strftime('%Y-%m-%d'))
print(cur.sort_values('actual_ctr_u', ascending=False).head(10).to_string(index=False))

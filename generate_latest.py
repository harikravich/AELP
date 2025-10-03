import pandas as pd
from pathlib import Path
p='artifacts/features/marketing_ctr_enhanced.parquet'
df=pd.read_parquet(p)
df2=df.sort_values(['ad_id','date']).drop_duplicates('ad_id', keep='last')
Path('artifacts/features').mkdir(parents=True, exist_ok=True)
df2.to_parquet('artifacts/features/marketing_ctr_latest_enhanced.parquet', index=False)
print('latest rows', len(df2))

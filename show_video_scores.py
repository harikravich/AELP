import pandas as pd
vs=pd.read_parquet('artifacts/predictions/veo_videos_scores.parquet')
vc=pd.read_parquet('artifacts/predictions/veo_videos_ctr.parquet')
print('Classifier p(click>0):')
print(vs.sort_values('pred_ctr', ascending=False).to_string(index=False))
print()
print('CTR estimate:')
print(vc.sort_values('pred_ctr', ascending=False).to_string(index=False))

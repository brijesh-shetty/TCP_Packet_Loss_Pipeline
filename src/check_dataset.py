import pandas as pd
import numpy as np

df = pd.read_csv('enhanced_dataset.csv')

print('=== YOUR DATASET STATISTICS ===')
print(f'Total rows        : {len(df):,}')
print(f'Total features    : {df.shape[1]}')
loss_count = int(df['label'].sum())
no_loss = len(df) - loss_count
print(f'Loss events       : {loss_count:,}  ({df["label"].mean()*100:.2f}%)')
print(f'No-loss events    : {no_loss:,}  ({100-df["label"].mean()*100:.2f}%)')
print()

if 'cc_algo' in df.columns:
    print('=== By CC Algorithm ===')
    g = df.groupby('cc_algo')['label'].agg(['count','sum','mean'])
    g.columns = ['total','loss_count','loss_rate']
    print(g.to_string())
    print()

if 'bandwidth_mbit' in df.columns:
    bw_vals = sorted(df['bandwidth_mbit'].dropna().unique().tolist())
    print('BW values (Mbit)  :', bw_vals)
if 'delay_ms' in df.columns:
    dl_vals = sorted(df['delay_ms'].dropna().unique().tolist())
    print('Delay values (ms) :', dl_vals)
if 'duration_s' in df.columns:
    dr_vals = sorted(df['duration_s'].dropna().unique().tolist())
    print('Duration (s)      :', dr_vals)

print()
print('=== BASE PAPER COMPARISON ===')
print('Base paper experiments : 5 BW x 5 delay x 3 BDP x 2 CC x 3 scenarios = 450 experiments')
print('Base paper duration    : 300s per experiment')
print('Base paper dataset     : ~500,000+ rows (much larger)')
print('Base paper loss rate   : ~1%  (very imbalanced, natural loss only)')
print()
print('YOUR experiments       :', end=' ')
n_exp = 150 * 2  # cubic + reno folders
print(f'{n_exp} (150 cubic + 150 reno)')
print('YOUR duration          : 100s and 300s variants')
print('YOUR dataset           : {:,} rows'.format(len(df)))
print('YOUR loss rate         : {:.2f}%  (multi-signal consensus)'.format(df['label'].mean()*100))
print()
print('KEY DIFFERENCE: Base paper uses 6 background flows to create congestion.')
print('Your data ALSO has 6 background flows (output_backup1/text/cubic/6bg_flows/)')
print('Loss rate differs because of LABELING METHOD, not the experiments.')

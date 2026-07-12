from pathlib import Path
import subprocess

models = Path('models')
niches = ['fitness', 'food']

print('=== MODEL FILES ===')
for n in niches:
    files = ['xgb', 'rf', 'lgb', 'ridge', 'weights', 'features', 'imputer', 'explainer']
    missing = [f for f in files if not (models / f'{f}_{n}.pkl').exists()]
    status = 'ALL OK' if not missing else ('MISSING: ' + str(missing))
    print(f'  {n}: {status}')

print()
print('=== model_train.py CHANGES ===')
src = Path('model_train.py').read_text(encoding='utf-8')
print('  PCA target:          ', 'PCA virality target' in src)
meta_block = src.split('METADATA_FEATURES')[1].split(']')[0]
print('  posted_hour removed: ', 'posted_hour' not in meta_block)
print('  chan_like_rate added:', 'log1p_chan_like_rate' in src)

print()
print('=== NICHE SCRIPTS (need to match model_train.py) ===')
for fname in ['model_train_fitness.py', 'model_train_food.py']:
    p = Path(fname)
    if not p.exists():
        print(f'  {fname}: MISSING - needs rebuild')
        continue
    s = p.read_text(encoding='utf-8')
    pca = 'PCA virality target' in s
    mb = s.split('METADATA_FEATURES')[1].split(']')[0] if 'METADATA_FEATURES' in s else ''
    timing_gone = 'posted_hour' not in mb
    print(f'  {fname}: PCA={pca}, timing_removed={timing_gone}')

print()
print('=== API SERVER ===')
r = subprocess.run(['python', '-c', 'import urllib.request; print(urllib.request.urlopen("http://localhost:8000/health", timeout=2).read().decode())'],
                   capture_output=True, text=True, timeout=5)
print(' ', r.stdout.strip() if r.returncode == 0 else 'DOWN - needs restart')

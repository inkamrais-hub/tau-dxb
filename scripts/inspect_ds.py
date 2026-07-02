from datasets import load_dataset
import os
os.environ['HF_HOME'] = '/hy-tmp/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
ds = load_dataset('leeduckgo/cantonese-life-scenarios-corpus', cache_dir='/hy-tmp/hf_cache', trust_remote_code=True)
print('Splits:', list(ds.keys()))
for split_name in ds.keys():
    split = ds[split_name]
    print(f'\n{split_name}: {len(split)} rows')
    if len(split) > 0:
        keys = list(split[0].keys())
        print(f'  Fields: {keys}')
        row = split[0]
        for k in keys:
            v = row[k]
            v_str = str(v)
            if len(v_str) > 120:
                v_str = v_str[:120] + '...'
            print(f'  {k}: {v_str}')

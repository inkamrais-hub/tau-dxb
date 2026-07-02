from datasets import load_dataset
try:
    ds = load_dataset('mozilla-foundation/common_voice_17_0', 'zh-HK', split='test', trust_remote_code=True)
    print('Success!', len(ds))
    print(list(ds[0].keys()))
except Exception as e:
    print(f'Error: {e}')

import json, os

preds = {}
with open('/root/dimsum/outputs/integrated/submission.jsonl') as f:
    for line in f:
        d = json.loads(line)
        preds[d['key']] = d['predict']

data = []
with open('/root/dimsum/data/test.jsonl') as f:
    for line in f:
        d = json.loads(line)
        fname = d['audio_path'].split('/')[-1].replace('.wav', '')
        data.append({
            'utt_id': d['utt_id'],
            'audio_path': d['audio_path'],
            'ref_text': d['ref_text'],
            'pred_text': preds.get(fname, ''),
            'tags': d.get('tags', []),
        })

with open('/root/dimsum/outputs/integrated/submission_fmt.jsonl', 'w') as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

print(f'Written {len(data)} entries')

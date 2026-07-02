import json

data = []
with open('/root/dimsum/outputs/integrated/submission.jsonl') as f:
    for line in f:
        d = json.loads(line)
        data.append({'utt_id': d['key'], 'ref_text': d['ground_truth'], 'pred_text': d['predict']})

with open('/root/dimsum/outputs/integrated/submission_fmt.jsonl', 'w') as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

print(f'Converted {len(data)} entries')

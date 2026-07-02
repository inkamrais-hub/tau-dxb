"""Check HF repo for available files."""
import urllib.request
import json

# Try HF API first
url = "https://hf-mirror.com/api/datasets/leeduckgo/cantonese-life-scenarios-corpus"
try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
        print("=== HF API Response ===")
        for f in data.get('siblings', []):
            print(f.get('rfilename', ''))
except Exception as e:
    print(f"API failed: {e}")
    # Fallback: parse the tree page
    url2 = "https://hf-mirror.com/datasets/leeduckgo/cantonese-life-scenarios-corpus/tree/main"
    try:
        req = urllib.request.Request(url2)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode()
            import re
            files = re.findall(r'href="/datasets/leeduckgo/cantonese-life-scenarios-corpus/resolve/main/([^"]+)"', html)
            print("=== Parsed from tree page ===")
            for f in sorted(set(files)):
                print(f)
    except Exception as e2:
        print(f"Tree page also failed: {e2}")

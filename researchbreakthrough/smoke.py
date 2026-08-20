import json, os, platform, urllib.request
print(json.dumps({'python': platform.python_version(), 'cpu_count': os.cpu_count()}))
with urllib.request.urlopen('https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv', timeout=20) as r:
    b = r.read(128)
print('NETWORK_OK', len(b), b[:40])

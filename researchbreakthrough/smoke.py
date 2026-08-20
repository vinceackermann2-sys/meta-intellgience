import json, os, platform, urllib.request
# synchronization marker: full real-data gates are now enabled on the base workflow
print(json.dumps({'python': platform.python_version(), 'cpu_count': os.cpu_count()}))
with urllib.request.urlopen('https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv', timeout=20) as r:
    b = r.read(128)
print('NETWORK_OK', len(b), b[:40])

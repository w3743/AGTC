import subprocess, sys
session = sys.argv[1]
code = sys.stdin.read()
p = subprocess.run([r'D:\lceda-pro\lceda-pro.exe','invoke','--session',session,'--code',code,'--timeout','60000'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=75)
print('RC',p.returncode)
sys.stdout.buffer.write(p.stdout.encode('utf-8','replace'))
sys.stderr.buffer.write(p.stderr.encode('utf-8','replace'))



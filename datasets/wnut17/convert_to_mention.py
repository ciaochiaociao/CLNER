import sys, re

for line in sys.stdin:
#line = line.strip()
#if line:
    line = re.sub(r'(?<=\t[B|I]-).*', 'mention', line)
    sys.stdout.write(line)

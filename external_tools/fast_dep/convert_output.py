lines = open('run__20517/out_put').readlines()
f = open('wsj10_tr', 'w')

correct = 0
total = 0

def process(i):
    global total, correct
    sent = lines[i].strip().split()[:-1]
    gold = lines[i + 1].strip().split()[1:]
    rule = lines[i + 2].strip().split()[1:]
    for s, g, r in zip(sent, gold, rule):
        total += 1
        if g == r:
            correct+=1
        idx, *w = s.split(':')
        w = ':'.join(w)
        *w, p, _ = w.split('/')
        w = '/'.join(w)
        g = int(g.split('-')[0])
        r = int(r.split('-')[0])
        if g == len(sent):
            g = 0
        else:
            g += 1
        if r == len(sent):
            r = 0
        else:
            r += 1
        f.write(f'{int(idx)+1}\t{w}\t-\t{p}\t-\t-\t{g}\t-\t-\t{r}\n')
    f.write('\n')

for i in range(0, len(lines), 5):
    process(i)
print(correct/total)
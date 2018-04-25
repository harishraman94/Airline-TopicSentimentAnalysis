w2id = {}
wf1 = open('..\\output\\voca.txt')
for l in wf1.readlines():
    ws = l.split()
    if ws[0] not in w2id:
        w2id[ws[0]] = ws[1]

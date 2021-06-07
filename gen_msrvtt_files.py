import pickle

initial = pickle.load(open('/scratch/shared/beegfs/albanie/shared-datasets/MSRVTT/high-quality/structured-symlinks/raw-captions.pkl', 'rb'))

idx_to_vid = {}
idx_to_sent = {}

raw_captions = {}

step = len(initial) / 12

for k in initial:
    key = int(k.split('video')[-1])
    id = key//step
    if k == 'video9268':
        print(id)
 
    if id not in raw_captions:
        raw_captions[id] = {}
    raw_captions[id][k] = initial[k]

print(raw_captions[11].keys())
'''
for nn in range(12):
    with open('./msrvtt_data/MSRVTT_' + str(nn) + '_test.csv', 'w') as f:
        f.write('key,vid_key,video_id,sentence\n')
        i = 0
        for k in raw_captions[nn]:
            for s in raw_captions[nn][k]:
                f.write('ret' + str(i) + ',msr' + k.split('video')[-1] + ',' + k + ',' + " ".join(s) + '\n')
                idx_to_vid[i] = k
                idx_to_sent[i] = " ".join(s)
                i += 1

    pickle.dump(idx_to_vid, open('./msrvtt_data/idx_to_vid' + str(nn) + '.pkl', 'wb'))
    pickle.dump(idx_to_sent, open('./msrvtt_data/idx_to_sent' + str(nn) + '.pkl', 'wb'))
'''

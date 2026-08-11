#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def filter_answerable(input_file,output_file):
    data=json.load(open(input_file,encoding='utf-8')); kept=[]; n=0
    for article in data['data']:
        a={'title':article.get('title',''),'paragraphs':[]}
        for para in article['paragraphs']:
            q=[x for x in para['qas'] if not x.get('is_impossible',False) and x.get('answers')]
            if q: a['paragraphs'].append({'context':para['context'],'qas':q}); n+=len(q)
        if a['paragraphs']: kept.append(a)
    Path(output_file).parent.mkdir(parents=True,exist_ok=True); json.dump({'version':data.get('version','2.0'),'data':kept},open(output_file,'w')); print(f'Wrote {n} answerable questions to {output_file}')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('input_file'); p.add_argument('output_file'); a=p.parse_args(); filter_answerable(a.input_file,a.output_file)

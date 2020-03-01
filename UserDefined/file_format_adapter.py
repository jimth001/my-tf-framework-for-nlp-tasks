from TextPreprocessing.file_api import read_file_lines,write_file_lines
base_path='../data/family_relationship/'
input='informal.txt'
target='formal.txt'

def combine(inf_path,fm_path,out_path):
    inf=read_file_lines(inf_path)
    fm=read_file_lines(fm_path)
    r=['input\ttarget']
    for i,f in zip(inf,fm):
        r.append(i+'\t'+f)
    write_file_lines(out_path,lines=r)

combine(base_path+'train/informal',base_path+'train/formal',base_path+'train.tsv')
combine(base_path+'tune/informal',base_path+'tune/formal.ref0',base_path+'dev.tsv')

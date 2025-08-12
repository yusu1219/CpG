import pandas as pd
import functions as F
chr=1
start=10000
end = 10935
cpg_sites_file='./data/cpg_standard.bed.gz'
print('setting is already. Data loading...')
data = pd.read_csv('./data/APL.mhap.gz',sep='\t',header=None,dtype={3:'str'},low_memory=False)
print('data is already!')
print(F._cal_cpel(cpg_sites_file, data ,chr ,start ,end ,step=500 ,compress=False ,vis=False))


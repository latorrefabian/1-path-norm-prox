import os
import pdb
from tqdm import tqdm

for f in tqdm(os.listdir('log')):
    job_num = os.path.splitext(f)[0]
    try:
        job_num = int(job_num)
    except:
        continue

    if job_num < 550800:
        os.remove(os.path.join('log', f))

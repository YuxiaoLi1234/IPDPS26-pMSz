import os, re, glob
import numpy as np
import matplotlib.pyplot as plt

def parse_file_groupwise(filepath):

    groups = []
    current = []

    with open(filepath, 'r') as f:
        for ln in f:
            if ln.strip().startswith("iteStep"):
                if current: 
                    groups.append(current)
                    current = []
            if "iteration" in ln:
                nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', ln)
                
                if len(nums) >= 1:
                    # a, b, c, d, it = map(float, nums[-5:])
                    print(nums)
                    comp = float(nums[-1])
                    comm = 0.0
                    current.append((comp, comm, int(round(it))))
        if current:
            groups.append(current)

    if not groups:
        return None, None, None

    
    totals = [sum(c+m for c,m,_ in g) for g in groups]
    idx = int(np.argmin(totals))

    comp = [c for c,_,_ in groups[idx]]
    comm = [m for _,m,_ in groups[idx]]
    labels = [it for *_,it in groups[idx]]
    return comp, comm, labels

def get_rank_from_filename(filename):
    m = re.search(r"rank(\d+)_", filename)
    return int(m.group(1)) if m else None


data_dir = "." # your data file path
files = glob.glob(os.path.join(data_dir, "../gannt_results/naive_MSz/NYX_sz3_rank_1.txt"))

all_data = {}
for file in files:
    rank = get_rank_from_filename(os.path.basename(file))
    if rank is None:
        continue
    comp, comm, labels = parse_file_groupwise(file)
    if comp is not None:
        all_data[rank] = (comp, comm, labels)


ranks = sorted(all_data.keys())
fig, ax = plt.subplots(figsize=(10,3))
bar_w = 0.3

comp_color = "#abc8e9"  
comm_color = "#ece190" 

previous=0
for i, rk in enumerate(ranks):
    comp, comm, labels = all_data[rk]
    n = len(comp)

    x = i + (np.arange(n) - (n-1)/2.0) * (bar_w+0.05)
    
    x = (np.arange(n)) * 0.5
    ax.bar(x, comp, bar_w, color=comp_color, label="Local Correction" if i==0 else None)
    ax.bar(x, comm, bar_w, bottom=comp, color=comm_color, label="Data Communication" if i==0 else None)
    cnt = 0
    for xi, c, m, lab in zip(x, comp, comm, labels):
        x = cnt
    
        draw_lab = lab
        if(cnt == 0): 
            previous = lab
        else: 
            draw_lab = lab - previous
            previous = lab
        # ax.text(xi, c+m, f"{cnt}", ha='center', va='bottom', fontsize=14)
        cnt+=1

ax.set_xticks(np.arange(n) * 0.5)
ax.set_xticklabels([str(r) for r in range(n)])
ax.tick_params(axis='x', labelsize=14) 
ax.tick_params(axis='y', labelsize=14)

ax.set_xlabel("Number of iterations",  fontsize=14, weight="light")
ax.set_ylabel("Time (seconds)",  fontsize=14, weight="light")
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("../figures/figure9b.svg")
plt.show()


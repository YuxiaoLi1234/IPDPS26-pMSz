import re, glob
import matplotlib.pyplot as plt
import numpy as np


comp_color = "#abc8e9"  
comm_color = "#ece190" 

def parse_file(filename):
    comp_list = []
    comm_list = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    in_step = False
    curr_comp = 0.0
    curr_comm = 0.0
    labe = 0
    for line in lines:
        if line.startswith('iteStep: 1'):
            if in_step:
                comp_list.append(curr_comp)
                comm_list.append(curr_comm)
            in_step = True
            curr_comp = 0.0
            curr_comm = 0.0
            comm_ite =0
            continue

        if in_step and 'version 2 iteration:' in line:
            parts = line.strip().split(':')[-1].strip().split(',')
            print(parts)
            if len(parts) < 4:
                continue
            vals = [float(p.strip()) for p in parts]
            curr_comp += vals[0] + vals[2]
            curr_comm += vals[1] + vals[3]
            comm_ite +=1
            labe = int(vals[-1])
    # 
    if in_step:
        comp_list.append(curr_comp)
        comm_list.append(curr_comm)

    return comp_list, comm_list, labe, comm_ite


txt_files = sorted(
    glob.glob("path/to/your/Global_performance_with_ite_cuda_sz3_perlin_*_*_*_rank*_.txt"),
    key=lambda x: int(re.search(r"_rank(\d+)_\.txt", x).group(1))
)

print(txt_files)
all_comps = []
all_comms = []
labels = []
ites = []
commu_ites = []
for fname in txt_files:
    match = re.search(r"_rank(\d+)_\.txt", fname)
    if not match:
        continue
    rank = int(match.group(1))
    comp, comm, ite, comm_ite = parse_file(fname)
    
    if comp and comm:
        all_comps.append(comp)
        all_comms.append(comm)
        labels.append(f"{rank}")
        ites.append(ite)
        commu_ites.append(comm_ite)

sorted_data = sorted(zip(labels, all_comps, all_comms), key=lambda x: int(x[0]))
labels = [x[0] for x in sorted_data]

print(labels)
iterations = ites

sorted_data = sorted(zip(labels, all_comps, all_comms, ites, commu_ites), key=lambda x: int(x[0]))

# 解包
labels     = [x[0] for x in sorted_data]

all_comps  = [ [v  for v in x[1]] for x in sorted_data ]  # comp / iteration
all_comms  = [ [v  for v in x[2]] for x in sorted_data ]  # comm / iteration


all_comps  = [ x[1] for x in sorted_data ]  # comp / iteration
all_comms  = [ x[2] for x in sorted_data ]  # comm / iteration

all_comms[0] = 0


avg_comps = [np.min(comp) for comp in all_comps]
avg_comms = [np.min(comm) for comm in all_comms]
total_time = [c + m for c, m in zip(avg_comps, avg_comms)]

x = np.arange(len(labels))
bar_width = 0.6

plt.figure(figsize=(10,4))
bar_position = (positions_comp + positions_comm) / 2

plt.bar(x, avg_comps, width=bar_width, label='Computation', color=comp_color)
plt.bar(x, avg_comms, width=bar_width, bottom=avg_comps, label='Data Communication', color="#f1aaaf")

for i, (t, it, comm_it) in enumerate(zip(total_time, ites, commu_ites + [commu_ites[-1]] * (len(labels) - len(commu_ites)))):
    label_text = f"{it}/{comm_it}"
    plt.text(x[i], t + 0.001, label_text, ha='center', va='bottom', fontsize=14)

plt.xticks(x, labels, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Number of GPUs', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("fig10b.svg", dpi=300)
plt.show()



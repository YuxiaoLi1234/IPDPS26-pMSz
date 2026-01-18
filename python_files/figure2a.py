import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import glob, re

comp_color = "#abc8e9"  
comm_color = "#ece190"  

def parse_file(filename):
    comp_list_c = []
    comp_list_r = []
    comm_list = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    in_step = False
    curr_comp_c = 0.0
    curr_comp_r = 0.0
    curr_comm = 0.0
    labe = 0
    for line in lines:
        if line.startswith('iteStep: 1'):
            if in_step:
                comp_list_c.append(curr_comp_c)
                comp_list_r.append(curr_comp_r)
                comm_list.append(curr_comm)
            in_step = True
            curr_comm = 0.0
            curr_comp_c = 0.0
            curr_comp_r = 0.0
            comm_ite =0
            continue

        if in_step and 'version 2 iteration:' in line:
            parts = line.strip().split(':')[-1].strip().split(',')
            if len(parts) != 9:
                continue
            vals = [float(p.strip()) for p in parts]
            curr_comp_c += vals[0]+ vals[4]
            curr_comp_r += vals[1]  + vals[5] + vals[6]+ vals[2] 
            curr_comm += vals[3]
            comm_ite +=1
            labe = int(vals[-1])
    if in_step:
        comp_list_c.append(curr_comp_c)
        comp_list_r.append(curr_comp_r)
        comm_list.append(curr_comm)

    return comp_list_c,comp_list_r, comm_list, labe, comm_ite

txt_files = sorted(*
    glob.glob("../stat_result/naive_MSz/perlin_*_sz3_*.txt"),
    key=lambda x: int(re.search(r"sz3_(\d+)\.txt", x).group(1))
)


all_comps_r = []
all_comps_c = []
all_comms = []
labels = []
ites = []
commu_ites = []
for fname in txt_files:
    match = re.search(r'sz3_(\d+)', fname)
    if not match:
        continue
    rank = int(match.group(1))
    comp_c, comp_r, comm, ite, comm_ite = parse_file(fname)
    
    if comp_c and comm:
        all_comps_c.append(comp_c)
        all_comps_r.append(comp_r)
        all_comms.append(comm)
        labels.append(f"{rank}")
        ites.append(ite)
        commu_ites.append(comm_ite)
sorted_data = sorted(zip(labels, all_comps_c, all_comps_r, all_comms), key=lambda x: int(x[0]))
labels = [x[0] for x in sorted_data]
print(labels)
iterations = ites
sorted_data = sorted(zip(labels, all_comps_c, all_comps_r, all_comms, ites, commu_ites), key=lambda x: int(x[0]))

labels     = [x[0] for x in sorted_data]
all_comps_c  = [ [v  for v in x[1][-29:]] for x in sorted_data ]  # comp / iteration
all_comps_r  = [ [v  for v in x[2][-29:]] for x in sorted_data ]  # comp / iteration
all_comms  = [ [v for v in x[3]][-29:] for x in sorted_data ]  # comm / iteration


all_comps_c  = [ x[1][-29:] for x in sorted_data ]  # comp / iteration
all_comps_r = [ x[2][-29:] for x in sorted_data ]  # comp / iteration
all_comms  = [ x[3][-29:] for x in sorted_data ]  # comm / iteration
all_comms[0] = 0


avg_comps_c = [np.mean(comp) for comp in all_comps_c]
avg_comps_r = [np.mean(comp) for comp in all_comps_r]
avg_comms = [np.mean(comm) for comm in all_comms]
total_time = [c + m + k for c, m, k in zip(avg_comps_c, avg_comms, avg_comps_r)]

positions_comp_c = np.arange(len(labels)) * 2.0
positions_comp_r = positions_comp_c + 0.8
positions_comm = positions_comp_r + 0.8

num_ranks = [int(l) for l in labels]
all_comps_c  = [ [v for v in x[1][-29:]] for x in sorted_data ]  # comp / iteration
all_comps_r  = [ [v  for v in x[2][-29:]] for x in sorted_data ]  # comp / iteration
all_comms  = [ [v for v in x[3][-29:]] for x in sorted_data ]
avg_comps_c =  [np.mean(comp) for i, comp in enumerate(all_comps_c)]
avg_comps_r =  [np.mean(comp) for i, comp in enumerate(all_comps_r)]
avg_comms =  [np.mean(comm) for i, comm in enumerate(all_comms)]
avg_comps = [np.mean(comp) + np.mean(all_comps_r[i]) + np.mean(avg_comms[i]) for i, comp in enumerate(all_comps_c)]


efficiency = [avg_comps[0] / (t) for i, t in enumerate(avg_comps)]  # 相对于第一个rank的效率

print(efficiency[-1])
x = np.arange(len(num_ranks))
fig, ax1 = plt.subplots(figsize=(10,4))

position_comm = [avg_comps_c[i] + avg_comps_r[i] for i in range(len(avg_comps_c))]
# --- Bar Chart: Comp Time ---
bar = ax1.bar(x, avg_comps_c, color=comp_color, width=0.6, label="Extract and Correct Wrong Extrema")
bar = ax1.bar(x, avg_comps_r, bottom = avg_comps_c, color="#85ab70", width=0.6, label="Integral Path Computation")
bar = ax1.bar(x, avg_comms, bottom = position_comm, color="#f1aaaf", width=0.6, label="Data Communication")
ax1.set_ylabel("Time (seconds)", color='black', fontsize=14)
ax1.set_ylim(0, max(avg_comps) + 0.03)
ax1.tick_params(axis='y', labelcolor='black', labelsize=14)

# 
for i in range(len(avg_comps)):
    ax1.text(x[i], avg_comps[i] + 0.001, f"{avg_comps[i]:.4f}", ha='center', fontsize=14)

# 
ax1.set_xticks(x)
ax1.set_xticklabels([str(r) for r in num_ranks])
ax1.tick_params(axis='x', labelsize=14)
ax1.set_xlabel("Number of GPUs", fontsize=14)

# --- Line Plot: Efficiency ---
ax2 = ax1.twinx()
ax2.plot(x, efficiency, color='orange', marker='o', linewidth=2,label="Parallel Efficiency")
ax2.set_ylabel("Parallel Efficiency", color='black', fontsize=14)
ax2.tick_params(axis='y', labelcolor='black', labelsize=14)
ax2.set_ylim(0, 1.2)

for i in range(len(efficiency)):
    ax2.text(x[i], efficiency[i] + 0.06, f"{efficiency[i] * 100:.2f}%", ha='center', color='black', fontsize=14)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2,
           loc="upper right", fontsize=12, frameon=False)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("../figures/figure2a.svg")
plt.show()

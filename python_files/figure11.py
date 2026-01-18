import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import glob, re

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
    labe=0
    for line in lines:
        if line.startswith('iteStep: 1'):
            if in_step:
                comp_list.append(curr_comp)
                comm_list.append(curr_comm)
            in_step = True
            curr_comp = 0.0
            curr_comm = 0.0
            continue

        if in_step and 'version 2 iteration:' in line:
            parts = line.strip().split(':')[-1].strip().split(',')
            if len(parts) < 4:
                continue
            vals = [float(p.strip()) for p in parts]
            curr_comp += vals[0] + vals[2]
            curr_comm += vals[1] + vals[3]
            labe = int(vals[-1])
            
    if in_step:
        comp_list.append(curr_comp)
        comm_list.append(curr_comm)

    return comp_list, comm_list, labe


txt_files = sorted(
    glob.glob("../stat_results/strong_scale/perlin_1024_1024_1024_sz3_rank_*.txt"),
    key=lambda x: int(re.search(r"rank_(\d+)\.txt", x).group(1))
)

print(txt_files)
all_comps = []
all_comms = []
labels = []
ites = []
for fname in txt_files:
    match = re.search(r'rank_(\d+)', fname)
    if not match:
        continue
    rank = int(match.group(1))
    comp, comm, labe = parse_file(fname)
    
    if comp and comm:
        all_comps.append(comp)
        all_comms.append(comm)
        labels.append(f"{rank}")
        ites.append(labe)
sorted_data = sorted(zip(labels, all_comps, all_comms), key=lambda x: int(x[0]))
labels = [x[0] for x in sorted_data]

labels     = [x[0] for x in sorted_data]

all_comps  = [ [v / x[3] for v in x[1]][-29:] for x in sorted_data ]  # comp / iteration
all_comms  = [ [v / x[3] for v in x[2]][-29:] for x in sorted_data ]  # comm / iteration

all_comms[0] = 0
plt.figure(figsize=(12, 6))

positions_comp = np.arange(len(labels)) * 2.0
positions_comm = positions_comp + 0.8


plt.boxplot(all_comps, positions=positions_comp, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor="lightblue"), showfliers=True)


plt.boxplot(all_comms, positions=positions_comm, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor="lightgreen"), showfliers=True)


middle_positions = (positions_comp + positions_comm) / 2
plt.xticks(middle_positions, labels)
plt.xlabel("Rank")
plt.ylabel("Time (s)")
plt.title("Strong Scaling: Comp (blue) vs Comm (green)")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("fixed_strong_scaling_boxplot.png", dpi=300)
plt.show()

avg_comps = [np.mean(comp) for comp in all_comps]
avg_comms = [np.mean(comm) for comm in all_comms]
total_time = [c + m for c, m in zip(avg_comps, avg_comms)]


x = np.arange(len(labels))
bar_width = 0.6


plt.figure(figsize=(12, 6))


plt.bar(x, avg_comps, width=bar_width, label='Comp', color='lightblue')


plt.bar(x, avg_comms, width=bar_width, bottom=avg_comps, label='Comm', color='lightgreen')


plt.xticks(x, labels, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Number of Ranks', fontsize=14)
plt.ylabel('Avg Time (s)', fontsize=14)
plt.title('Strong Scaling: Stacked Comp and Comm Time ')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("strong_scaling_stacked_barchart_with_iters.png", dpi=300)
plt.show()


num_ranks = [int(l) for l in labels]
all_comps  = [ [v for v in x[1][-5:]] for x in sorted_data ]  # comp / iteration
all_comms  = [ [v for v in x[2][-5:]] for x in sorted_data ]
all_comms[0]=0
avg_comps =  [np.mean(comp) + np.mean(all_comms[i]) for i, comp in enumerate(all_comps)]
efficiency = [avg_comps[0] / (t * num_ranks[i]) for i,t in enumerate(avg_comps)]  # 

x = np.arange(len(num_ranks))

fig, ax1 = plt.subplots(figsize=(8, 3))

# --- Bar Chart: Comp Time ---
bar = ax1.bar(x, avg_comps, color=comp_color, width=0.6, label="Computation Time")
ax1.set_ylabel("Time (seconds)", color='black', fontsize=14)
ax1.set_ylim(0, max(avg_comps) + 0.5)
ax1.tick_params(axis='y', labelcolor='black',  labelsize=14)


for i in range(len(avg_comps)):
    ax1.text(x[i], avg_comps[i] + 0.002, f"{avg_comps[i]:.2f}", ha='center', fontsize=14)


ax1.set_xticks(x)
ax1.set_xticklabels([str(r) for r in num_ranks])
ax1.tick_params(axis='x', labelcolor='black',  labelsize=14)
ax1.set_xlabel("Number of GPUs",  fontsize=14)

# --- Line Plot: Efficiency ---
ax2 = ax1.twinx()
ax2.plot(x, efficiency, color='orange', marker='o', linewidth=2,label="Parallel Efficiency")
ax2.set_ylabel("Parallel Efficiency", color='black',  fontsize=14)
ax2.tick_params(axis='y', labelcolor='black',  labelsize=14)

ax2.set_ylim(0, 1.2)

# Efficiency
for i in range(len(efficiency)):
    ax2.text(x[i], efficiency[i] + 0.06, f"{efficiency[i]*100:.2f}"+"%", ha='center', color='black', fontsize=14)
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2,
           loc="upper right", fontsize=12, frameon=False)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("../figures/figure11.svg")
plt.show()


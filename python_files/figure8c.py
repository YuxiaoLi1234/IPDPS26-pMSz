import re, glob, matplotlib.pyplot as plt
import numpy as np

def parse_file(filename):
    
    segments = []
    in_step = False
    prev_end = 0.0

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("iteStep: 1"):
                in_step = True
                prev_end = 0.0
                segments = []
                continue

            if in_step and "version 2 iteration:" in line:
                parts = line.split(":")[-1].split(",")
                vals = [float(p.strip()) for p in parts if p.strip() != ""]
                if len(vals) < 4:
                    continue

                a0, a1, a2, a3= vals[:4]

             
                seg0 = max(a0 - prev_end, 0.0)
                seg1 = max(a1 - a0,      0.0)
                seg2 = max(a2 - a1,      0.0)
                seg3 = max(a3 - a2, 0.0)
               
                segments.extend([
                    ("comp", seg0 + seg2),
                   
                    ("comm", seg1 + seg3),
                    
                ])

                prev_end = a3  # 更新基线

    return segments


txt_files = sorted(glob.glob("../gannt_results/pMSz/perlin_*_sz3_rank_*.txt"))
all_data = []
for f in txt_files:
    match = re.search(r'rank_(\d+)_', f)
    rank = int(match.group(1)) if match else len(all_data)
    segs = parse_file(f)
    all_data.append((rank, segs))

all_data = sorted(all_data, key=lambda x: x[0])


fig, ax = plt.subplots(figsize=(10, 3))
y_height = 0.6

for i, (rank, segments) in enumerate(all_data):
    t = 0.0
    for j, (stype, duration) in enumerate(segments):
        if duration <= 0:
            continue
        if stype == "comp":
            color = "#80b1d3"  
            label = "Local Correction" if (i == 0 and j == 0) else ""
        else:
            color = "#f1aaaf" 
            label = "Data Communication" if (i == 0 and j == 1) else ""
        ax.barh(y=i, width=duration, left=t, height=y_height,
                color=color, edgecolor='none', label=label)
        t += duration

ax.set_yticks(range(len(all_data)))
ax.set_yticklabels([f"{r}" for r, _ in all_data], fontsize=10)
ax.set_xlabel("Time (second)", fontsize=14)
ax.set_ylabel("GPU ID", fontsize=14)
ax.legend(loc='upper right', fontsize=10, frameon=False)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("../figures/figure8c.svg", dpi=300)
plt.show()

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
                if len(vals) < 7:
                    continue

                a0, a1, a2, a3, a4, a5, a6, a7= vals[:-1]

               
                seg0 = max(a0 - prev_end, 0.0)
                seg1 = max(a1 - a0,      0.0)
                seg2 = max(a2 - a1,      0.0)
                seg3 = max(a3 - a2, 0.0)
                seg4 = max(a4 - a3,      0.0)
                seg5 = max(a5 - a4,      0.0)
                seg6 = max(a6 - a5,      0.0)
                seg7 = max(a7 - a6,      0.0)
                
               
                segments.extend([
                    ("comp", seg0),
                    ("rcomp", seg1 +seg5 +seg6+ seg2 ),
                    ("comm", seg3 +seg7
                     
                    ),
                    # ("comp", seg2),
                    # ("comm", seg3),
                ])

                prev_end = a7 

    return segments

txt_files = sorted(glob.glob("../data/gannt_chart/global/Global_naive_MSz_performance_with_ite_cuda_sz3_perlin_*rank_*.txt"))
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
            color = "#80b1d3"  # 蓝色：Local Correction
            label = "Local Correction" if (i == 0 and j == 0) else ""
        elif stype == "rcomp":
            color = "#85ab70"  # 蓝色：Local Correction
            label = "Local Correction" if (i == 0 and j == 0) else ""
        else:
            color = "#f1aaaf"  # 粉色：Data Communication
            label = "Data Communication" if (i == 0 and j == 1) else ""
        ax.barh(y=i, width=duration, left=t, height=y_height,
                color=color, edgecolor='none', label=label)
        t += duration

# === 美化 ===
ax.set_yticks(range(len(all_data)))
ax.set_yticklabels([f"{r}" for r, _ in all_data], fontsize=10)
ax.set_xlabel("Time (second)", fontsize=14)
ax.set_ylabel("GPU ID", fontsize=14)
ax.legend(loc='upper right', fontsize=10, frameon=False)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("fig8a.svg", dpi=300)
plt.show()

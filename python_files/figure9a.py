import matplotlib.pyplot as plt
import numpy as np
def barcharts(data, type1):

    iterations = data[:, 0]
    update_directions = data[:, 1]
    find_false_critical_point = data[:, 2]
    fix_false_critical_point = data[:, 3]
    mss_computation = data[:, 4]
    find_false_regular_point = data[:, 5]
    fix_false_regular_point = data[:, 6]
    labels = data[:, 7]
    
    critical = find_false_critical_point + fix_false_critical_point
    others = update_directions + mss_computation + find_false_regular_point + fix_false_regular_point
    
    width = 0.8
    fig, ax = plt.subplots(figsize=(11,3))
    
    p1 = ax.bar(iterations, critical, width, label='C-loops', color='#bec0da')
    p2 = ax.bar(iterations, others, width, bottom=critical, label='R-loops', color='#efb7c0')
    
    ax.set_xlabel('Number of iterations', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.legend()
    
    plt.savefig("../figures/figure9a.svg")
    plt.xticks(fontsize=14, weight="light")
    plt.yticks(fontsize=14, weight="light")
    plt.show()

data_file = "../stat_result/MSz/NYX_sz3_rank_1.txt"
data = []
result = 0
with open(data_file, 'r') as file:
    for line in file:
        if line.startswith("iteration:"):
            parts = line.split()
            iteration = int(parts[1].strip(':'))
            values = [float(v.replace(',', ''))/1000 for v in parts[2:8]]
            result+=np.sum(values)
            label = int(parts[8].replace(',', ''))
            data.append((iteration, *values, label))
    

data = np.array(data)

barcharts(data,"cuda")
data = [i[1:-1] for i in data]
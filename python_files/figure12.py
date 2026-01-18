import pandas as pd
import matplotlib.pyplot as plt
import re

filenames = ["dark_density_normalized", "Nyx"]
compressor_id = "sz3"
for compressor_id in ["sz3", "zfp"]:
    for filename in filenames:

        file_path = "../stat_result/pMSz/"+filename+"sz3_16.txt"

        with open(file_path, "r") as f:
            text = f.read()
        if(compressor_id == "zfp"): comp_color = "#96a5d4"
        else: comp_color = "#85ab70"

        pattern = re.compile(
            r"absolute_error:\s*([0-9.eE+-]+).*?"
            r"Overall CR:\s*([0-9.eE+-]+).*?"
            r"Original CR:\s*([0-9.eE+-]+).*?"
            r"Edit_ratio:\s*([0-9.eE+-]+).*?"
            r"Compression_time:\s*([0-9.eE+-]+).*?"
            r"Correction_time:\s*([0-9.eE+-]+).*?",
            re.S
        )
        matches = pattern.findall(text)

        df = pd.DataFrame(matches, columns=[
            "absolute_error", "Overall CR", "Original CR", "Edit_ratio", "Compression_time", "Correction_time"
        ])
        df = df[df["Correction_time"] != 0].astype(float)
        metrics = ["Correction_time"]


        font = 20
        for metric in metrics:
            plt.figure(figsize=(10, 4))
            plt.plot(df["absolute_error"], df[metric], color=comp_color, marker="o", label = "Ours" + " ("+compressor_id.upper() +")")
            plt.xscale("log")  
            plt.xlabel("Absolute Error", fontsize=font)
            
            
            plt.grid(True)

            
            if metric == "Correction_time":
                plt.ylabel("Computation Overhead (seconds)", fontsize=14)
                plt.plot(df["absolute_error"], df["Compression_time"], color="#efb7c0", marker="o", label="Compression Time" + " ("+compressor_id.upper() +")")
                plt.yscale("log")
                
                plt.legend()
            
            plt.tight_layout()
            plt.yticks(fontsize=font)
            plt.xticks(fontsize=font)
            plt.savefig(filename+"_"+compressor_id +"_" + metric+".svg")
            plt.show()
    
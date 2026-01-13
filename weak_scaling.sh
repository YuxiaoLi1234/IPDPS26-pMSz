#!/bin/bash

ranks=(1 2 4 8 16 32 64 128)

base_dim=512


# 循环每个进程数配置
for total_ranks in "${ranks[@]}"; do
  echo "=========================================="
  echo "Weak Scaling Test: Total Ranks = $total_ranks"
  echo "=========================================="
  
 
  if [ $total_ranks -eq 1 ]; then
    rx=1; ry=1; rz=1
  elif [ $total_ranks -eq 2 ]; then
    rx=1; ry=1; rz=2    
  elif [ $total_ranks -eq 4 ]; then
    rx=1; ry=2; rz=2    
  elif [ $total_ranks -eq 8 ]; then
    rx=2; ry=2; rz=2      
  elif [ $total_ranks -eq 16 ]; then
    rx=2; ry=2; rz=4   
  elif [ $total_ranks -eq 32 ]; then
    rx=2; ry=4; rz=4      
  elif [ $total_ranks -eq 64 ]; then
    rx=4; ry=4; rz=4    
  elif [ $total_ranks -eq 128 ]; then
    rx=4; ry=4; rz=8   
  else
    echo "Unsupported rank count: $total_ranks"
    continue
  fi
  
 
  global_x=$((base_dim * rx))
  global_y=$((base_dim * ry))
  global_z=$((base_dim * rz))
  
  echo "Grid decomposition: ${rx}x${ry}x${rz} (优先Z->Y->X)"
  echo "Global data size: ${global_x}x${global_y}x${global_z}"
  echo "Per-rank data size: ${base_dim}x${base_dim}x${base_dim}"
  
  
  for run in {1..10}; do
    echo "-------------------"
    echo "Run $run / 10"
    echo "-------------------"
    
    echo "Running pMSz..."
    srun -n $total_ranks \
        ./pMSz \
        "/pscratch/sd/y/yuxiaoli/datasets/perlin_weak_scaling/perlin_${global_x}_${global_y}_${global_z},${global_x},${global_y},${global_z}" \
        1e-4 sz3 $rx $ry $rz
   
    echo "Running naive_pMSz..."
    srun -n $total_ranks \
        ./naive_pMSz \
        "/pscratch/sd/y/yuxiaoli/datasets/perlin_weak_scaling/perlin_${global_x}_${global_y}_${global_z}.bin.bin,${global_x},${global_y},${global_z}" \
        1e-4 sz3 $rx $ry $rz
  done
  
  echo ""
done

echo "All weak scaling tests completed!"
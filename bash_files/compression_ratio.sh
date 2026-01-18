datasets=(
  "./datasets/at.bin,177,95,48"
  "./datasets/CESM.bin,3600,1800,1"
  "./datasets/vortex.bin,128,128,128"
  "./datasets/jet.bin,480,720,120"
  "./datasets/Nyx.bin,512,512,512"
)

bounds=(
    1.000000e-06
    1.206793e-06
    1.456348e-06
    1.757511e-06
    2.120951e-06
    2.559548e-06
    3.088844e-06
    3.727594e-06
    4.498433e-06
    5.428675e-06
    6.551286e-06
    7.906043e-06
    9.540955e-06
    1.151395e-05
    1.389495e-05
    1.676833e-05
    2.023590e-05
    2.442053e-05
    2.947052e-05
    3.556480e-05
    4.291934e-05
    5.179475e-05
    6.250552e-05
    7.543120e-05
    9.102982e-05
    1.098541e-04
    1.325711e-04
    1.599859e-04
    1.930698e-04
    2.329952e-04
    2.811769e-04
    3.393222e-04
    4.094915e-04
    4.941713e-04
    5.963623e-04
    7.196857e-04
    8.685114e-04
    1.048113e-03
    1.264855e-03
    1.526418e-03
    1.842070e-03
    2.222996e-03
    2.682696e-03
    3.237458e-03
    3.906940e-03
    4.714866e-03
    5.689866e-03
    6.866488e-03
    8.286428e-03
    1.000000e-02
)

compressors=(sz3)

for data in "${datasets[@]}"; do
    for comp in "${compressors[@]}"; do
        for eb in "${bounds[@]}"; do
        echo "Run=$run, EB=$eb, Dataset=$data, Compressor=$comp"
        srun -n 1 ./pMSz $data $eb $comp 1 1 1
        done
    done
done

for data in "${datasets[@]}"; do
    for comp in "${compressors[@]}"; do
        for eb in "${bounds[@]}"; do
        echo "Run=$run, EB=$eb, Dataset=$data, Compressor=$comp"
        srun -n 1 ./find_best_ocr $data $eb $comp
        done
    done
done

datasets=(
  "/pscratch/sd/y/yuxiaoli/datasets/dark_density_normalized.bin,1024,1024,1024"
)

bounds=(
    1e-10
    1e-9
    1e-8
    1e-7
    1e-6
    7e-6
    1e-5
    3e-5
    7e-5
    1e-4
    3e-4
    1e-3
)

compressors=(sz3 zfp)

for data in "${datasets[@]}"; do
    for comp in "${compressors[@]}"; do
        for eb in "${bounds[@]}"; do
        echo "Run=$run, EB=$eb, Dataset=$data, Compressor=$comp"
        srun -n 16 ./distributed_MSz_64 $data $eb $comp 4 4 1
        done
    done
done


 



#!/bin/bash

# 定义数组 emdb
emdb=(
    # 'dataset/emdb/P0/09_outdoor_walk'
    # 'dataset/emdb/P2/19_indoor_walk_off_mvs'
    # 'dataset/emdb/P2/20_outdoor_walk'
    # 'dataset/emdb/P2/24_outdoor_long_walk'
    # 'dataset/emdb/P3/27_indoor_walk_off_mvs'
    # 'dataset/emdb/P3/28_outdoor_walk_lunges'
    # 'dataset/emdb/P3/29_outdoor_stairs_up'
    # 'dataset/emdb/P3/30_outdoor_stairs_down'
    # 'dataset/emdb/P4/35_indoor_walk'
    # 'dataset/emdb/P4/36_outdoor_long_walk'
    # 'dataset/emdb/P4/37_outdoor_run_circle'
    # 'dataset/emdb/P5/40_indoor_walk_big_circle'
    'dataset/emdb/P6/48_outdoor_walk_downhill'
    # 'dataset/emdb/P6/49_outdoor_big_stairs_down'
    # 'dataset/emdb/P7/55_outdoor_walk'
    # 'dataset/emdb/P7/56_outdoor_stairs_up_down'
    # 'dataset/emdb/P7/57_outdoor_rock_chair'
    # 'dataset/emdb/P7/58_outdoor_parcours'
    # 'dataset/emdb/P7/61_outdoor_sit_lie_walk'
    # 'dataset/emdb/P8/64_outdoor_skateboard'
    # 'dataset/emdb/P8/65_outdoor_walk_straight'
    # 'dataset/emdb/P9/77_outdoor_stairs_up'
    # 'dataset/emdb/P9/78_outdoor_stairs_up_down'
    # 'dataset/emdb/P9/79_outdoor_walk_rectangle'
    # 'dataset/emdb/P9/80_outdoor_walk_big_circle'
)

# 遍历数组并执行命令
for input_path in "${emdb[@]}"; do
    echo "Processing: $input_path"
    CUDA_VISIBLE_DEVICES=0 python scripts/inference_all_mast3r_emdb.py --input "$input_path" --output_dir "results" --visualize_mask
done

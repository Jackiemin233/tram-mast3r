'''
Multi-GPUs Inference - NJ

python scripts/inference_all_mast3r_emdb_seq.py --gpu_num 6

Remember modify your running command in Line 93!
command = device+str(idx)+' python scripts/inference_all_mast3r_emdb.py --input path/to/your/data/images

'''
import argparse
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import shlex

parser = argparse.ArgumentParser()

parser.add_argument('-sf', '--source_folder', default='data/', type=str)
parser.add_argument('--gpu_num', default='data/', type=int)

args = parser.parse_args()

source_folder = args.source_folder
gpu_num = args.gpu_num

device = 'CUDA_VISIBLE_DEVICES='

# NOTE: emdb seq hard code
emdb = ['dataset/P0/09_outdoor_walk',
        'dataset/P2/19_indoor_walk_off_mvs',
        'dataset/P2/20_outdoor_walk',
        'dataset/P2/24_outdoor_long_walk',
        'dataset/P3/27_indoor_walk_off_mvs',
        'dataset/P3/28_outdoor_walk_lunges',
        'dataset/P3/29_outdoor_stairs_up',
        'dataset/P3/30_outdoor_stairs_down',
        'dataset/P4/35_indoor_walk',
        'dataset/P4/36_outdoor_long_walk',
        'dataset/P4/37_outdoor_run_circle',
        'dataset/P5/40_indoor_walk_big_circle',
        'dataset/P6/48_outdoor_walk_downhill',
        'dataset/P6/49_outdoor_big_stairs_down',
        'dataset/P7/55_outdoor_walk',
        'dataset/P7/56_outdoor_stairs_up_down',
        'dataset/P7/57_outdoor_rock_chair',
        'dataset/P7/58_outdoor_parcours',
        'dataset/P7/61_outdoor_sit_lie_walk',
        'dataset/P8/64_outdoor_skateboard',
        'dataset/P8/65_outdoor_walk_straight',
        'dataset/P9/77_outdoor_stairs_up',
        'dataset/P9/78_outdoor_stairs_up_down',
        'dataset/P9/79_outdoor_walk_rectangle',
        'dataset/P9/80_outdoor_walk_big_circle',
        ]
# For SWH
# emdb = ['dataset/emdb/P0/09_outdoor_walk',
#         'dataset/emdb/P2/19_indoor_walk_off_mvs',
#         'dataset/emdb/P2/20_outdoor_walk',
#         'dataset/emdb/P2/24_outdoor_long_walk',
#         'dataset/emdb/P3/27_indoor_walk_off_mvs',
#         'dataset/emdb/P3/28_outdoor_walk_lunges',
#         'dataset/emdb/P3/29_outdoor_stairs_up',
#         'dataset/emdb/P3/30_outdoor_stairs_down',
#         'dataset/emdb/P4/35_indoor_walk',
#         'dataset/emdb/P4/36_outdoor_long_walk',
#         'dataset/emdb/P4/37_outdoor_run_circle',
#         'dataset/emdb/P5/40_indoor_walk_big_circle',
#         'dataset/emdb/P6/48_outdoor_walk_downhill',
#         'dataset/emdb/P6/49_outdoor_big_stairs_down',
#         'dataset/emdb/P7/55_outdoor_walk',
#         'dataset/emdb/P7/56_outdoor_stairs_up_down',
#         'dataset/emdb/P7/57_outdoor_rock_chair',
#         'dataset/emdb/P7/58_outdoor_parcours',
#         'dataset/emdb/P7/61_outdoor_sit_lie_walk',
#         'dataset/emdb/P8/64_outdoor_skateboard',
#         'dataset/emdb/P8/65_outdoor_walk_straight',
#         'dataset/emdb/P9/77_outdoor_stairs_up',
#         'dataset/emdb/P9/78_outdoor_stairs_up_down',
#         'dataset/emdb/P9/79_outdoor_walk_rectangle',
#         'dataset/emdb/P9/80_outdoor_walk_big_circle',
#         ]


# 按 gpu_num 个文件分组
for i in range(0, len(emdb), gpu_num):
    # 取出当前的 6 个文件
    selected_files = emdb[i:i+gpu_num]
    print("Selected Dir:", selected_files)
    commands=[]

    for idx, name in enumerate(selected_files):
        # NOTE: Change to your Own running command
        command = device+str(idx)+' python scripts/inference_all_mast3r_emdb.py --input ' + '../' + name + '/images'
        print('command',command)
        commands.append(command)

    # 定义一个函数来运行命令
    def run_command(command_str):
        command = shlex.split(command_str)

        env_vars = {}
        env_vars['CUDA_VISIBLE_DEVICES'] = command[0][-1]

        # 设置环境变量
        env = os.environ.copy()  # 复制当前环境变量
        env.update(env_vars)  # 更新为特定的环境变量

        result = subprocess.run(command[1:], capture_output=True, text=True, env=env)
        return result

    # 使用 ThreadPoolExecutor 来运行命令
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 提交所有命令
        future_to_command = {executor.submit(run_command, cmd): cmd for cmd in commands}

        # 等待所有命令完成
        for future in as_completed(future_to_command):
            cmd = future_to_command[future]
            try:
                result = future.result()
                print(f"Command: {cmd}, Return Code: {result.returncode}")
                if result.returncode == 0:
                    print(f"Output: {result.stdout}")
                else:
                    print(f"Error: {result.stderr}")
            except Exception as e:
                print(f"Command: {cmd} generated an exception: {e}")

    # 所有命令完成后的代码段
    print("All commands completed.")
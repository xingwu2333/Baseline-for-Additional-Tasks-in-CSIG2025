import numpy as np
import os
from collections import defaultdict


def load_tracking_results(file_path):
    """
    加载跟踪结果文件
    返回: {frame_id: [(obj_id, x, y), ...]}
    """
    tracks = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame_id = int(parts[0])
            obj_count = int(parts[1])

            for i in range(obj_count):
                obj_id = int(parts[2 + i * 3])
                x = float(parts[3 + i * 3])
                y = float(parts[4 + i * 3])
                tracks[frame_id].append((obj_id, x, y))
    return tracks


def reassign_close_targets(tracks, dist_threshold=10.0):
    """
    重新关联距离小于阈值的不同ID目标
    参数:
        tracks: {frame_id: [(obj_id, x, y), ...]}
        dist_threshold: 关联阈值(像素)
    返回: 处理后的轨迹数据
    """
    # 收集所有目标轨迹
    all_tracks = defaultdict(list)
    for frame_id in sorted(tracks.keys()):
        for obj_id, x, y in tracks[frame_id]:
            all_tracks[obj_id].append((frame_id, x, y))

    # 计算目标平均移动速度
    obj_velocity = {}
    for obj_id in all_tracks:
        if len(all_tracks[obj_id]) < 2:
            continue
        velocities = []
        for i in range(1, len(all_tracks[obj_id])):
            dx = all_tracks[obj_id][i][1] - all_tracks[obj_id][i - 1][1]
            dy = all_tracks[obj_id][i][2] - all_tracks[obj_id][i - 1][2]
            dt = all_tracks[obj_id][i][0] - all_tracks[obj_id][i - 1][0]
            if dt > 0:
                velocities.append((dx / dt, dy / dt))
        if velocities:
            obj_velocity[obj_id] = np.mean(velocities, axis=0)

    # 寻找可以合并的轨迹对
    merge_pairs = []
    obj_ids = list(all_tracks.keys())

    for i in range(len(obj_ids)):
        id1 = obj_ids[i]
        if id1 not in obj_velocity:
            continue

        for j in range(i + 1, len(obj_ids)):
            id2 = obj_ids[j]
            if id2 not in obj_velocity:
                continue

            # 检查轨迹时间是否重叠
            frames1 = {f[0] for f in all_tracks[id1]}
            frames2 = {f[0] for f in all_tracks[id2]}
            if frames1 & frames2:  # 如果有重叠帧则跳过
                continue

            # 找到两个轨迹最近的时间点
            last_frame1 = max(all_tracks[id1], key=lambda x: x[0])
            first_frame2 = min(all_tracks[id2], key=lambda x: x[0])

            if first_frame2[0] <= last_frame1[0]:  # 时间顺序不对
                continue

            # 预测两个轨迹在中间点的位置
            gap = first_frame2[0] - last_frame1[0]
            pred_x = last_frame1[1] + obj_velocity[id1][0] * gap
            pred_y = last_frame1[2] + obj_velocity[id1][1] * gap

            # 计算实际距离
            dist = np.sqrt((pred_x - first_frame2[1]) ** 2 + (pred_y - first_frame2[2]) ** 2)

            if dist < dist_threshold:
                merge_pairs.append((id1, id2, dist))

    # 合并轨迹
    id_mapping = {}
    for id1, id2, _ in sorted(merge_pairs, key=lambda x: x[2]):
        # 处理已经映射过的ID
        while id1 in id_mapping:
            id1 = id_mapping[id1]
        while id2 in id_mapping:
            id2 = id_mapping[id2]

        if id1 == id2:
            continue

        # 保留较小的ID
        new_id = min(id1, id2)
        old_id = max(id1, id2)
        id_mapping[old_id] = new_id

    # 应用ID映射
    processed_tracks = defaultdict(list)
    for frame_id in tracks:
        for obj_id, x, y in tracks[frame_id]:
            # 处理映射关系
            while obj_id in id_mapping:
                obj_id = id_mapping[obj_id]
            processed_tracks[frame_id].append((obj_id, x, y))

    return processed_tracks


def save_processed_tracks(tracks, output_path):
    """
    保存处理后的轨迹
    """
    with open(output_path, 'w') as f:
        for frame_id in sorted(tracks.keys()):
            objs = tracks[frame_id]
            line = [f"{frame_id:05d}", str(len(objs))]
            for obj_id, x, y in objs:
                line.extend([str(obj_id), f"{x:.2f}", f"{y:.2f}"])
            f.write(' '.join(line) + '\n')


def process_tracking_file(input_path, output_path, dist_threshold=10.0):
    """
    处理单个跟踪结果文件
    """
    tracks = load_tracking_results(input_path)
    processed_tracks = reassign_close_targets(tracks, dist_threshold)
    save_processed_tracks(processed_tracks, output_path)


if __name__ == "__main__":
    import glob

    # 配置参数
    input_dir = "out_centroid_val/res"  # SORT输出目录
    output_dir = "out_centroid_val/res"  # 处理后目录
    dist_threshold = 14.0  # 关联阈值(像素)

    os.makedirs(output_dir, exist_ok=True)

    # 处理所有跟踪结果文件
    for input_file in glob.glob(os.path.join(input_dir, "*.txt")):
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        print(f"处理文件: {filename}")
        process_tracking_file(input_file, output_file, dist_threshold)

    print(f"\n处理完成，结果保存在: {output_dir}")

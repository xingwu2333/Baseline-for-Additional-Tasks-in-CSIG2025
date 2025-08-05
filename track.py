import numpy as np
import os
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """
    使用卡尔曼滤波器跟踪边界框
    """
    count = 0

    def __init__(self, bbox):
        """
        初始化跟踪器，使用检测框作为初始状态

        参数:
            bbox: [x, y] 检测框中心坐标
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # 状态转移矩阵 (x, y, vx, vy)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # 观测矩阵 (只观测x,y)
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])

        # 过程噪声协方差矩阵
        self.kf.Q[2:, 2:] *= 10.

        # 观测噪声协方差矩阵
        self.kf.R[2:, 2:] *= 0.01

        # 初始状态协方差矩阵
        self.kf.P[2:, 2:] *= 1000.

        # 初始状态
        self.kf.x[:2] = bbox.reshape(2, 1)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        使用观测到的边界框更新状态向量
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox[:2])

    def predict(self):
        """
        推进状态向量并返回预测的边界框估计
        """
        if (self.kf.x[2] + self.kf.x[3]) > 0:
            self.hit_streak = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:2].reshape(1, 2))
        return self.history[-1]

    def get_state(self):
        """
        返回当前边界框估计
        """
        return self.kf.x[:2].reshape(1, 2)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    使用匈牙利算法将检测框与现有跟踪器关联
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # 计算检测框和预测框之间的IOU矩阵
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            # 计算中心点距离作为相似度度量
            dist = np.sqrt((det[0] - trk[0]) ** 2 + (det[1] - trk[1]) ** 2)
            # 使用距离的倒数作为相似度(距离越小相似度越高)
            iou_matrix[d, t] = 1.0 / (1.0 + dist)

    # 使用匈牙利算法找到最优匹配
    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.asarray(matched_indices).T

    # 找出未匹配的检测
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    # 找出未匹配的跟踪器
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # 过滤掉低IOU的匹配
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < 1.0 / (1.0 + iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class SORT:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=2.0):
        """
        初始化SORT跟踪器
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        更新跟踪器状态
        """
        self.frame_count += 1

        # 获取跟踪器的预测位置
        trks = np.zeros((len(self.trackers), 2))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1]]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # 更新匹配的跟踪器
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        # 为未匹配的检测创建新的跟踪器
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)

        # 生成输出结果
        for trk in self.trackers:
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(
                    np.concatenate(([trk.id + 1], d)).reshape(1, -1))  # +1 as MOT benchmark requires positive IDs

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 3))


def parse_detection_file(file_path):
    """
    解析检测结果文件
    """
    detections = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            frame_id = int(parts[0])
            target_count = int(parts[1])

            frame_detections = []
            for i in range(target_count):
                x = float(parts[3 + i * 3])
                y = float(parts[4 + i * 3])
                frame_detections.append([x, y])

            detections[frame_id] = frame_detections
    return detections


def interpolate_missing_detections(tracked_results, max_gap=5):
    """
    对缺失的检测进行插值处理
    """
    # 收集所有目标ID
    all_ids = set()
    for frame_id in tracked_results:
        for obj in tracked_results[frame_id]:
            all_ids.add(obj[0])

    # 为每个目标建立完整轨迹
    interpolated_results = {}
    for obj_id in all_ids:
        # 收集该目标的所有观测
        observations = {}
        for frame_id in tracked_results:
            for obj in tracked_results[frame_id]:
                if obj[0] == obj_id:
                    observations[frame_id] = obj[1:]
                    break

        if not observations:
            continue

        # 找到该目标的起始和结束帧
        min_frame = min(observations.keys())
        max_frame = max(observations.keys())

        # 线性插值缺失的帧
        prev_frame = None
        for frame_id in range(min_frame, max_frame + 1):
            if frame_id in observations:
                prev_frame = frame_id
                continue

            if prev_frame is None:
                continue

            # 找到下一个有效帧
            next_frame = None
            for f in range(frame_id + 1, max_frame + 1):
                if f in observations:
                    next_frame = f
                    break

            if next_frame is None or (next_frame - prev_frame) > max_gap:
                continue

            # 计算插值比例
            ratio = (frame_id - prev_frame) / (next_frame - prev_frame)

            # 线性插值坐标
            prev_x, prev_y = observations[prev_frame]
            next_x, next_y = observations[next_frame]
            interp_x = prev_x + ratio * (next_x - prev_x)
            interp_y = prev_y + ratio * (next_y - prev_y)

            observations[frame_id] = [interp_x, interp_y]

        # 将插值结果保存回最终结果
        for frame_id in observations:
            if frame_id not in interpolated_results:
                interpolated_results[frame_id] = []
            interpolated_results[frame_id].append([obj_id] + observations[frame_id])

    return interpolated_results


def process_video_detections(detection_files):
    """
    处理多段视频的检测结果
    """
    results = {}

    for file_path in detection_files:
        video_name = os.path.splitext(os.path.basename(file_path))[0]
        detections = parse_detection_file(file_path)

        mot_tracker = SORT(
            max_age=30,  # 允许更长时间未匹配
            min_hits=1,  # 减少确认所需匹配次数
            iou_threshold=11  # 更大的关联阈值
        )

        tracked_results = {}

        # 第一遍处理获取原始跟踪结果
        for frame_id in sorted(detections.keys()):
            dets = np.array(detections[frame_id])
            tracked_objects = mot_tracker.update(dets)

            frame_results = []
            for obj in tracked_objects:
                obj_id, x, y = int(obj[0]), float(obj[1]), float(obj[2])
                frame_results.append([obj_id, x, y])

            tracked_results[frame_id] = frame_results

        # 应用轨迹插值
        interpolated_results = interpolate_missing_detections(tracked_results, max_gap=1)

        results[video_name] = interpolated_results

    return results


def save_tracking_results(results, output_dir):
    """
    保存跟踪结果到文件
    """
    os.makedirs(output_dir, exist_ok=True)

    for video_name, tracked_data in results.items():
        output_file = os.path.join(output_dir, f"{video_name}.txt")

        with open(output_file, 'w') as f:
            # 按帧ID排序写入
            for frame_id in sorted(tracked_data.keys()):
                objects = tracked_data[frame_id]

                # 格式: frame_id target_count [obj_id x y]...
                line = [f"{frame_id:05d}", str(len(objects))]
                for obj in objects:
                    line.extend([str(obj[0]), f"{obj[1]:.2f}", f"{obj[2]:.2f}"])

                f.write(' '.join(line) + '\n')


if __name__ == "__main__":
    import glob

    # 输入输出目录配置
    detection_dir = "out_centroid_val"  # 检测文件目录
    output_dir = "out_centroid_val/res"  # 输出目录

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有检测文件
    detection_files = glob.glob(os.path.join(detection_dir, "*.txt"))

    if not detection_files:
        print(f"错误: 在目录 {detection_dir} 中未找到任何.txt文件")
        exit(1)

    print(f"找到 {len(detection_files)} 个检测文件:")
    for file in detection_files[:5]:
        print(f"  - {os.path.basename(file)}")
    if len(detection_files) > 5:
        print(f"  ... 和 {len(detection_files) - 5} 个其他文件")

    # 处理并跟踪
    print("\n开始处理跟踪...")
    tracking_results = process_video_detections(detection_files)

    # 保存结果
    save_tracking_results(tracking_results, output_dir)

    print(f"\n跟踪完成，结果已保存到 {output_dir}")
    print("生成的文件:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")

import numpy as np
import os
from pathlib import Path
import cv2
import datetime

class TUMOutput:
    def __init__(self, output_dir="output", width=640, height=480):
        """
        TUMデータセット形式で出力するクラス
        
        Args:
            output_dir (str): 出力ディレクトリ
            width (int): 出力画像の幅
            height (int): 出力画像の高さ
        """
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        
        # 出力ディレクトリの作成
        self.rgb_dir = self.output_dir / "rgb"
        self.depth_dir = self.output_dir / "depth"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        
        # グラウンドトゥルース用のファイル
        self.gt_trajectory_file = self.output_dir / "groundtruth.txt"
        self.rgb_timestamps_file = self.output_dir / "rgb.txt"
        self.depth_timestamps_file = self.output_dir / "depth.txt"
        
        # ヘッダー作成
        self._create_header_files()
        
        # 現在のタイムスタンプを記録
        self.start_timestamp = self._get_timestamp()
    
    def _create_header_files(self):
        """
        TUMデータセット形式のヘッダーファイルを作成する
        """
        # groundtruth.txtのヘッダー
        with open(self.gt_trajectory_file, 'w') as f:
            f.write("# ground truth trajectory\n")
            f.write("# timestamp tx ty tz qx qy qz qw\n")
        
        # rgb.txtのヘッダー
        with open(self.rgb_timestamps_file, 'w') as f:
            f.write("# color images\n")
            f.write("# timestamp filename\n")
        
        # depth.txtのヘッダー
        with open(self.depth_timestamps_file, 'w') as f:
            f.write("# depth images\n")
            f.write("# timestamp filename\n")
    
    def _get_timestamp(self):
        """
        現在のタイムスタンプを取得する
        
        Returns:
            float: タイムスタンプ
        """
        return datetime.datetime.now().timestamp()
    
    def _convert_view_matrix_to_pose(self, view_matrix):
        """
        ビュー行列をTUM形式のポーズに変換する
        
        Args:
            view_matrix (numpy.ndarray): ビュー行列
            
        Returns:
            tuple: (位置, クォータニオン)
        """
        # ビュー行列の逆行列がカメラのポーズ
        pose_matrix = np.linalg.inv(view_matrix)
        
        # 位置抽出
        position = pose_matrix[:3, 3]
        
        # 回転行列をクォータニオンに変換
        rotation_matrix = pose_matrix[:3, :3]
        
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
                qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                qx = 0.25 * s
                qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
                qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qy = 0.25 * s
                qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
                qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                qz = 0.25 * s
        
        return position, (qx, qy, qz, qw)
    
    def save_frame(self, frame_idx, rgb_image, depth_image, camera_position, view_matrix):
        """
        フレームを保存する
        
        Args:
            frame_idx (int): フレームインデックス
            rgb_image (numpy.ndarray): RGB画像
            depth_image (numpy.ndarray): DEPTH画像
            camera_position (numpy.ndarray): カメラ位置
            view_matrix (numpy.ndarray): ビュー行列
        """
        # タイムスタンプの計算（開始時間からの相対時間）
        timestamp = self.start_timestamp + frame_idx * 0.1  # 10Hz想定
        
        # ファイル名の生成
        rgb_filename = f"{frame_idx:06d}.png"
        depth_filename = f"{frame_idx:06d}.png"
        
        # RGB画像の保存（OpenCVはBGRなのでRGBに変換）
        rgb_image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.rgb_dir / rgb_filename), rgb_image_bgr)
        
        # DEPTH画像の保存（16ビット画像として保存）
        cv2.imwrite(str(self.depth_dir / depth_filename), depth_image)
        
        # ポーズの計算
        position, quaternion = self._convert_view_matrix_to_pose(view_matrix)
        
        # groundtruth.txtに記録
        with open(self.gt_trajectory_file, 'a') as f:
            f.write(f"{timestamp:.6f} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} "
                   f"{quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}\n")
        
        # rgb.txtに記録
        with open(self.rgb_timestamps_file, 'a') as f:
            f.write(f"{timestamp:.6f} rgb/{rgb_filename}\n")
        
        # depth.txtに記録
        with open(self.depth_timestamps_file, 'a') as f:
            f.write(f"{timestamp:.6f} depth/{depth_filename}\n")
    
    def create_association_file(self):
        """
        TUMデータセット形式の関連付けファイルを作成する
        """
        # RGB画像とDEPTH画像とポーズの関連付けファイル
        assoc_file = self.output_dir / "associations.txt"
        
        # RGBタイムスタンプの読み込み
        rgb_timestamps = []
        rgb_files = []
        with open(self.rgb_timestamps_file, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    timestamp, filename = parts
                    rgb_timestamps.append(float(timestamp))
                    rgb_files.append(filename)
        
        # DEPTHタイムスタンプの読み込み
        depth_timestamps = []
        depth_files = []
        with open(self.depth_timestamps_file, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    timestamp, filename = parts
                    depth_timestamps.append(float(timestamp))
                    depth_files.append(filename)
        
        # グラウンドトゥルースの読み込み
        gt_timestamps = []
        gt_poses = []
        with open(self.gt_trajectory_file, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) == 8:
                    timestamp = float(parts[0])
                    pose = " ".join(parts[1:])
                    gt_timestamps.append(timestamp)
                    gt_poses.append(pose)
        
        # 関連付けファイルの作成
        with open(assoc_file, 'w') as f:
            f.write("# rgb_timestamp rgb_file depth_timestamp depth_file gt_timestamp tx ty tz qx qy qz qw\n")
            
            for i in range(len(rgb_timestamps)):
                if i < len(depth_timestamps) and i < len(gt_timestamps):
                    f.write(f"{rgb_timestamps[i]:.6f} {rgb_files[i]} "
                           f"{depth_timestamps[i]:.6f} {depth_files[i]} "
                           f"{gt_timestamps[i]:.6f} {gt_poses[i]}\n")
    
    def create_camera_calibration(self, fx=525.0, fy=525.0, cx=319.5, cy=239.5):
        """
        カメラキャリブレーションファイルを作成する
        
        Args:
            fx (float): x方向の焦点距離
            fy (float): y方向の焦点距離
            cx (float): x方向の主点位置
            cy (float): y方向の主点位置
        """
        # カメラキャリブレーションファイルの作成
        calib_file = self.output_dir / "calibration.txt"
        
        with open(calib_file, 'w') as f:
            f.write(f"# Camera calibration parameters\n")
            f.write(f"fx={fx}\n")
            f.write(f"fy={fy}\n")
            f.write(f"cx={cx}\n")
            f.write(f"cy={cy}\n")
            f.write(f"width={self.width}\n")
            f.write(f"height={self.height}\n")
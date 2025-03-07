import os
import cv2
import numpy as np
import datetime

class TUMFormatWriter:
    """TUMデータセット形式でデータを出力するクラス"""
    
    def __init__(self, output_dir="tum_format_output", camera_params=None):
        """
        初期化関数
        
        Parameters:
        output_dir (str): 出力ディレクトリのパス
        camera_params (dict, optional): カメラの内部パラメータ
        """
        self.output_dir = output_dir
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.depth_dir = os.path.join(output_dir, "depth")
        
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        
        # trajectory.txtファイルの初期化
        self.trajectory_file = os.path.join(output_dir, "groundtruth.txt")
        with open(self.trajectory_file, 'w') as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
        
        # rgb.txtファイルの初期化
        self.rgb_file = os.path.join(output_dir, "rgb.txt")
        with open(self.rgb_file, 'w') as f:
            f.write("# timestamp filename\n")

        # depth.txtファイルの初期化
        self.depth_file = os.path.join(output_dir, "depth.txt")
        with open(self.depth_file, 'w') as f:
            f.write("# timestamp filename\n")
            
        # カメラパラメータをファイルに保存
        if camera_params:
            self._write_camera_params(camera_params)
    
    def _write_camera_params(self, params):
        """カメラパラメータをファイルに保存"""
        camera_file = os.path.join(self.output_dir, "camera.txt")
        with open(camera_file, 'w') as f:
            f.write("# TUM Dataset Camera Parameters\n")
            f.write(f"# Resolution: {params['width']}x{params['height']}\n")
            f.write(f"# Field of View: {params['fov']} degrees\n\n")
            f.write("# Camera Matrix (K):\n")
            f.write(f"{params['fx']} 0 {params['cx']}\n")
            f.write(f"0 {params['fy']} {params['cy']}\n")
            f.write("0 0 1\n")
    
    def write_frame(self, frame, depth_map, frame_idx, camera_pos, camera_quaternion):
        """
        フレーム画像、深度画像、カメラ位置情報をTUM形式で出力
        
        Parameters:
        frame (numpy.ndarray): 出力するRGB画像データ
        depth_map (numpy.ndarray): 出力する深度画像データ (メートル単位)
        frame_idx (int): フレームインデックス
        camera_pos (numpy.ndarray): カメラの位置 [x, y, z]
        camera_quaternion (numpy.ndarray): カメラの姿勢を表す四元数 [qx, qy, qz, qw]
        
        Returns:
        float: 生成したタイムスタンプ
        """
        # タイムスタンプの生成（現在時刻を秒で表現）
        timestamp = datetime.datetime.now().timestamp()
        
        # RGBフレーム画像の保存
        frame_filename = f"{frame_idx:06d}.png"
        frame_path = os.path.join(self.rgb_dir, frame_filename)
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # 深度画像の保存
        depth_filename = f"{frame_idx:06d}.png"
        depth_path = os.path.join(self.depth_dir, depth_filename)
        
        # 深度マップをスケーリングして保存（TUMフォーマットに合わせて調整）
        # SLAMテスト用にスケールを5000に設定（メートル単位の5000倍 → ミリメートル単位の5倍）
        scaled_depth = (depth_map * 5000).astype(np.uint16)
        cv2.imwrite(depth_path, scaled_depth)
        
        # rgb.txtファイルへの書き込み
        with open(self.rgb_file, 'a') as f:
            f.write(f"{timestamp:.6f} rgb/{frame_filename}\n")
            
        # depth.txtファイルへの書き込み
        with open(self.depth_file, 'a') as f:
            f.write(f"{timestamp:.6f} depth/{depth_filename}\n")
        
        # trajectory.txtファイルへの書き込み
        with open(self.trajectory_file, 'a') as f:
            # TUM形式: timestamp tx ty tz qx qy qz qw
            f.write(f"{timestamp:.6f} {camera_pos[0]:.6f} {camera_pos[1]:.6f} {camera_pos[2]:.6f} "
                    f"{camera_quaternion[0]:.6f} {camera_quaternion[1]:.6f} "
                    f"{camera_quaternion[2]:.6f} {camera_quaternion[3]:.6f}\n")
        
        return timestamp
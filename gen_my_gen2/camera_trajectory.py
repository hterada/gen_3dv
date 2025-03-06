import numpy as np
import math
import torch

class CameraTrajectory:
    def __init__(self, radius=5.0, height=2.0, speed=0.01, num_frames=360, use_gpu=True):
        """
        カメラ軌道計算クラス
        
        Args:
            radius (float): カメラ軌道の半径
            height (float): カメラの高さ
            speed (float): カメラの移動速度
            num_frames (int): 生成するフレーム数
            use_gpu (bool): GPUを使用するかどうか
        """
        self.radius = radius
        self.height = height
        self.speed = speed
        self.num_frames = num_frames
        self.current_frame = 0
        self.use_gpu = use_gpu
        
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # トラジェクトリデータ計算用
        self.trajectory_data = self._precompute_trajectory()
    
    def _precompute_trajectory(self):
        """
        カメラ軌道のデータを事前計算する
        
        Returns:
            dict: カメラの位置と回転の辞書
        """
        # PyTorchを使用した高速な軌道計算
        if self.use_gpu:
            angles = torch.linspace(0, 2 * math.pi, self.num_frames, device=self.device)
            positions = torch.zeros((self.num_frames, 3), device=self.device, dtype=torch.float32)
            
            # 位置の計算
            positions[:, 0] = self.radius * torch.cos(angles)  # X座標
            positions[:, 1] = torch.ones_like(angles) * self.height  # Y座標（高さ）
            positions[:, 2] = self.radius * torch.sin(angles)  # Z座標
            
            # カメラの向きのデータ（見る点は原点）
            look_at = torch.zeros((self.num_frames, 3), device=self.device, dtype=torch.float32)
            
            # カメラの上方向ベクトル
            up = torch.tensor([[0, 1, 0]], device=self.device, dtype=torch.float32).repeat(self.num_frames, 1)
            
            # 視点行列の計算
            view_matrices = torch.zeros((self.num_frames, 4, 4), device=self.device, dtype=torch.float32)
            
            for i in range(self.num_frames):
                view_matrices[i] = self._look_at_torch(positions[i], look_at[i], up[i])
            
            return {
                'positions': positions.cpu().numpy(),
                'view_matrices': view_matrices.cpu().numpy()
            }
        else:
            # NumPyを使用した計算
            angles = np.linspace(0, 2 * math.pi, self.num_frames)
            positions = np.zeros((self.num_frames, 3))
            
            # 位置の計算
            positions[:, 0] = self.radius * np.cos(angles)  # X座標
            positions[:, 1] = np.ones_like(angles) * self.height  # Y座標（高さ）
            positions[:, 2] = self.radius * np.sin(angles)  # Z座標
            
            # 視点行列の計算
            view_matrices = np.zeros((self.num_frames, 4, 4))
            
            for i in range(self.num_frames):
                view_matrices[i] = self._look_at_numpy(
                    positions[i],
                    np.array([0, 0, 0]),  # 原点を見る
                    np.array([0, 1, 0])   # 上方向
                )
            
            return {
                'positions': positions,
                'view_matrices': view_matrices
            }
    
    def _look_at_torch(self, eye, target, up):
        """
        視点行列を計算する（PyTorch実装）
        
        Args:
            eye (torch.Tensor): カメラの位置
            target (torch.Tensor): 見る点の位置
            up (torch.Tensor): カメラの上方向
            
        Returns:
            torch.Tensor: 視点行列
        """
        # 全てのテンソルがfloat型であることを保証
        eye = eye.float()
        target = target.float()
        up = up.float()
        
        forward = target - eye
        forward = forward / torch.norm(forward)
        
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        
        new_up = torch.cross(right, forward)
        
        # 視点行列を構築
        view_matrix = torch.eye(4, device=self.device)
        
        view_matrix[0, 0] = right[0]
        view_matrix[0, 1] = right[1]
        view_matrix[0, 2] = right[2]
        
        view_matrix[1, 0] = new_up[0]
        view_matrix[1, 1] = new_up[1]
        view_matrix[1, 2] = new_up[2]
        
        view_matrix[2, 0] = -forward[0]
        view_matrix[2, 1] = -forward[1]
        view_matrix[2, 2] = -forward[2]
        
        view_matrix[0, 3] = -torch.dot(right, eye)
        view_matrix[1, 3] = -torch.dot(new_up, eye)
        view_matrix[2, 3] = torch.dot(forward, eye)
        
        return view_matrix
    
    def _look_at_numpy(self, eye, target, up):
        """
        視点行列を計算する（NumPy実装）
        
        Args:
            eye (numpy.ndarray): カメラの位置
            target (numpy.ndarray): 見る点の位置
            up (numpy.ndarray): カメラの上方向
            
        Returns:
            numpy.ndarray: 視点行列
        """
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        new_up = np.cross(right, forward)
        
        # 視点行列を構築
        view_matrix = np.eye(4)
        
        view_matrix[0, 0] = right[0]
        view_matrix[0, 1] = right[1]
        view_matrix[0, 2] = right[2]
        
        view_matrix[1, 0] = new_up[0]
        view_matrix[1, 1] = new_up[1]
        view_matrix[1, 2] = new_up[2]
        
        view_matrix[2, 0] = -forward[0]
        view_matrix[2, 1] = -forward[1]
        view_matrix[2, 2] = -forward[2]
        
        view_matrix[0, 3] = -np.dot(right, eye)
        view_matrix[1, 3] = -np.dot(new_up, eye)
        view_matrix[2, 3] = np.dot(forward, eye)
        
        return view_matrix
    
    def get_next_frame_data(self):
        """
        次のフレームのカメラデータを取得する
        
        Returns:
            tuple: (カメラ位置, ビュー行列, 現在のフレーム番号)
        """
        if self.current_frame >= self.num_frames:
            # 一周完了したらリセット
            self.current_frame = 0
        
        position = self.trajectory_data['positions'][self.current_frame]
        view_matrix = self.trajectory_data['view_matrices'][self.current_frame]
        
        frame_idx = self.current_frame
        self.current_frame += 1
        
        return position, view_matrix, frame_idx
    
    def get_projection_matrix(self, fov=60.0, aspect=4/3, near=0.1, far=100.0):
        """
        射影行列を計算する
        
        Args:
            fov (float): 視野角（度）
            aspect (float): アスペクト比
            near (float): 近平面の距離
            far (float): 遠平面の距離
            
        Returns:
            numpy.ndarray: 射影行列
        """
        # 透視投影行列を計算
        fov_rad = np.radians(fov)
        f = 1.0 / np.tan(fov_rad / 2)
        
        projection = np.zeros((4, 4))
        projection[0, 0] = f / aspect
        projection[1, 1] = f
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = (2 * far * near) / (near - far)
        projection[3, 2] = -1
        
        return projection
    
    def get_total_frames(self):
        """
        総フレーム数を取得する
        
        Returns:
            int: 総フレーム数
        """
        return self.num_frames
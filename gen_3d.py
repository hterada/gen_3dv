import numpy as np
import cv2
import os
import math
from PIL import Image, ImageDraw
import datetime
import torch
import time


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


class GPURenderer:
    def __init__(self, width=640, height=480, fov=60, device=None):
        """
        初期化関数
        
        Parameters:
        width (int): 画像の幅
        height (int): 画像の高さ
        fov (float): 視野角（度）
        device (torch.device, optional): 使用するデバイス。指定がなければGPUが利用可能なら使用
        """
        self.width = width
        self.height = height
        self.fov = fov
        self.aspect_ratio = width / height
        self.near = 0.1
        self.far = 100
        
        # デバイスの設定
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"PyTorchで使用するデバイス: {self.device}")
        
        # カメラの内部パラメータを計算
        self.focal_length = self.width / (2 * math.tan(math.radians(self.fov / 2)))
        self.cx = self.width / 2
        self.cy = self.height / 2
        self.fx = self.focal_length
        self.fy = self.focal_length
        
        # カメラの内部パラメータを表示
        self._print_camera_intrinsics()
        
        # 射影行列
        self.projection_matrix = self._get_perspective_projection()
        
        # キューブのテクスチャを作成
        self.create_cube_textures()
        
        # ピクセル座標のメッシュグリッドを事前計算（GPU上）
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing='ij'
        )
        
        self.pixel_coords = torch.stack([
            (x_coords.float() / self.width) * 2 - 1,        # NDC x座標
            1 - (y_coords.float() / self.height) * 2,       # NDC y座標
            -torch.ones_like(x_coords, device=self.device)  # NDC z座標（常に-1）
        ], dim=-1)
        
        # NDC座標からレイ方向への変換に使用する逆射影行列
        self.inv_projection = torch.inverse(torch.tensor(self.projection_matrix, device=self.device))
    
    def _print_camera_intrinsics(self):
        """カメラの内部パラメータを表示"""
        print("カメラ内部パラメータ:")
        print(f"解像度: {self.width}x{self.height}")
        print(f"画角(FOV): {self.fov}度")
        print(f"焦点距離(fx, fy): ({self.fx:.2f}, {self.fy:.2f})")
        print(f"画像中心(cx, cy): ({self.cx:.2f}, {self.cy:.2f})")
        print(f"カメラ行列 K:")
        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        print(K)
        print("")
    
    def _get_perspective_projection(self):
        """透視投影行列を作成"""
        f = 1.0 / math.tan(math.radians(self.fov / 2))
        return np.array([
            [f / self.aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (self.far + self.near) / (self.near - self.far), (2 * self.far * self.near) / (self.near - self.far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
    
    def create_cube_textures(self):
        """キューブの各面に異なるテクスチャを作成"""
        # キューブの頂点（原点中心の1x1x1）
        self.vertices = np.array([
            [-0.5, -0.5, -0.5, 1.0],  # 0
            [0.5, -0.5, -0.5, 1.0],   # 1
            [0.5, 0.5, -0.5, 1.0],    # 2
            [-0.5, 0.5, -0.5, 1.0],   # 3
            [-0.5, -0.5, 0.5, 1.0],   # 4
            [0.5, -0.5, 0.5, 1.0],    # 5
            [0.5, 0.5, 0.5, 1.0],     # 6
            [-0.5, 0.5, 0.5, 1.0]     # 7
        ], dtype=np.float32)
        
        # GPU用に変換
        self.vertices_gpu = torch.tensor(self.vertices, device=self.device)
        
        # 面（各面は4つの頂点で構成、反時計回りに指定）
        self.faces = [
            [0, 1, 2, 3],  # 前面
            [5, 4, 7, 6],  # 背面
            [1, 5, 6, 2],  # 右面
            [4, 0, 3, 7],  # 左面
            [3, 2, 6, 7],  # 上面
            [1, 0, 4, 5]   # 底面
        ]
        
        # 三角形に分割した面のインデックス（GPU上での計算用）
        self.triangles = []
        self.triangle_face_mapping = []  # 各三角形がどの面に属するかのマッピング
        
        for face_idx, face in enumerate(self.faces):
            # 面を2つの三角形に分割
            self.triangles.append([face[0], face[1], face[2]])  # 三角形1
            self.triangles.append([face[0], face[2], face[3]])  # 三角形2
            self.triangle_face_mapping.extend([face_idx, face_idx])
        
        # GPU用に変換
        self.triangles_gpu = torch.tensor(self.triangles, device=self.device)
        self.triangle_face_mapping_gpu = torch.tensor(self.triangle_face_mapping, device=self.device)
        
        # テクスチャマッピング用のUV座標
        self.uvs = np.array([
            [0, 0],  # 左下
            [1, 0],  # 右下
            [1, 1],  # 右上
            [0, 1]   # 左上
        ], dtype=np.float32)
        
        # 各面に異なるテクスチャを作成（異なる色の市松模様）
        self.textures = []
        self.textures_gpu = []
        colors = [
            ((255, 0, 0), (200, 0, 0)),       # 赤
            ((0, 255, 0), (0, 200, 0)),       # 緑
            ((0, 0, 255), (0, 0, 200)),       # 青
            ((255, 255, 0), (200, 200, 0)),   # 黄
            ((0, 255, 255), (0, 200, 200)),   # シアン
            ((255, 0, 255), (200, 0, 200))    # マゼンタ
        ]
        
        for color1, color2 in colors:
            texture = self._create_checkboard_texture(128, 128, color1, color2)
            self.textures.append(texture)
            # NumPy配列をPyTorchテンソルに変換
            texture_gpu = torch.tensor(texture, device=self.device).float() / 255.0
            self.textures_gpu.append(texture_gpu)
    
    def _create_checkboard_texture(self, width, height, color1, color2, squares=8):
        """指定サイズの市松模様テクスチャを作成"""
        image = Image.new("RGB", (width, height), color1)
        draw = ImageDraw.Draw(image)
        
        square_width = width // squares
        square_height = height // squares
        
        for i in range(squares):
            for j in range(squares):
                if (i + j) % 2 == 0:
                    continue
                draw.rectangle(
                    [i * square_width, j * square_height, (i + 1) * square_width, (j + 1) * square_height],
                    fill=color2
                )
        
        return np.array(image)
    
    def quaternion_from_euler(self, roll, pitch, yaw):
        """オイラー角を四元数に変換"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw])
    
    def get_view_matrix(self, camera_pos, target_pos, up_vector):
        """カメラ位置、ターゲット、上ベクトルからビュー行列を作成"""
        # カメラ座標系の作成
        z_axis = camera_pos - target_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(up_vector, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        # ビュー行列の作成
        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[0, :3] = x_axis
        view_matrix[1, :3] = y_axis
        view_matrix[2, :3] = z_axis
        
        # 移動成分
        view_matrix[0, 3] = -np.dot(x_axis, camera_pos)
        view_matrix[1, 3] = -np.dot(y_axis, camera_pos)
        view_matrix[2, 3] = -np.dot(z_axis, camera_pos)
        
        return view_matrix
    
    def render_scene_gpu(self, angle):
        """GPUを使用してキューブの周りを周回するカメラからシーンをレンダリング"""
        # 処理時間の計測を開始
        start_time = time.time()
        
        # 円上のカメラ位置を計算
        radius = 3.0
        camera_x = radius * math.sin(angle)
        camera_z = radius * math.cos(angle)
        camera_pos = np.array([camera_x, 0, camera_z], dtype=np.float32)
        
        # 原点（キューブの位置）を注視
        target_pos = np.array([0, 0, 0], dtype=np.float32)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        
        # ビュー行列の作成
        view_matrix = self.get_view_matrix(camera_pos, target_pos, up_vector)
        
        # GPU用に変換
        view_matrix_gpu = torch.tensor(view_matrix, device=self.device)
        inv_view_matrix_gpu = torch.inverse(view_matrix_gpu)
        
        # カメラ回転の四元数表現を取得
        camera_quaternion = self.quaternion_from_euler(0, -angle + math.pi, 0)
        
        # 頂点をカメラ空間に変換
        homogeneous_vertices = self.vertices_gpu.clone()
        camera_space_vertices = torch.matmul(view_matrix_gpu, homogeneous_vertices.T).T
        
        # 白色背景のフレームと深度バッファを初期化
        frame = torch.ones((self.height, self.width, 3), device=self.device)
        depth_buffer = torch.ones((self.height, self.width), device=self.device) * self.far
        
        # 全ピクセルのレイ方向を計算
        # ray_clip: [height, width, 3]
        ray_clip = self.pixel_coords.clone()
        
        # ray_clip を 4次元に拡張 [height, width, 4]
        ray_clip = torch.cat([ray_clip, torch.ones((self.height, self.width, 1), device=self.device)], dim=-1)
        
        # ray_eye: [height, width, 4]
        # 逆射影行列を適用
        ray_eye = torch.matmul(self.inv_projection, ray_clip.reshape(-1, 4).T).T
        ray_eye = ray_eye.reshape(self.height, self.width, 4)
        ray_eye = torch.cat([
            ray_eye[..., :2], 
            -torch.ones((self.height, self.width, 1), device=self.device), 
            torch.zeros((self.height, self.width, 1), device=self.device)
        ], dim=-1)
        
        # ray_world: [height, width, 4]
        # カメラ空間からワールド空間に変換
        ray_world_full = torch.matmul(inv_view_matrix_gpu, ray_eye.reshape(-1, 4).T).T
        ray_world_full = ray_world_full.reshape(self.height, self.width, 4)
        
        # 方向ベクトルのみを取り出す [height, width, 3]
        ray_world = ray_world_full[..., :3]
        
        # 正規化
        ray_world_norm = torch.nn.functional.normalize(ray_world, dim=-1)
        
        # カメラの位置を拡張 [height, width, 3]
        origin = torch.tensor(camera_pos, device=self.device).expand(self.height, self.width, 3)
        
        # 面ごとに処理（近いものから遠いものへ）
        for tri_idx in range(len(self.triangles)):
            # 三角形の頂点インデックス
            v_idx = self.triangles_gpu[tri_idx]
            
            # 三角形の3つの頂点
            v0 = self.vertices_gpu[v_idx[0], :3]
            v1 = self.vertices_gpu[v_idx[1], :3]
            v2 = self.vertices_gpu[v_idx[2], :3]
            
            # 面の法線ベクトルを計算
            normal = torch.cross(v1 - v0, v2 - v0)
            normal = torch.nn.functional.normalize(normal, dim=0)
            
            # レイキャスティングによる交点計算
            # (p0 - origin) · normal / (ray_dir · normal)
            denom = torch.sum(ray_world_norm * normal, dim=-1)
            
            # ゼロ除算を回避（法線とレイが平行の場合）
            mask_valid = torch.abs(denom) > 1e-6
            
            # 交点計算（t値）
            t = torch.zeros((self.height, self.width), device=self.device)
            
            # 効率化：有効なピクセルだけで計算
            if mask_valid.any():
                p0_minus_origin = v0 - origin[0, 0]
                numer = torch.sum(p0_minus_origin * normal, dim=-1)
                t[mask_valid] = numer / denom[mask_valid]
            
            # 交点がカメラの前方にあるかどうか
            mask_positive = t > 0
            
            # 深度テストのマスク
            mask_depth = t < depth_buffer
            
            # 両方の条件を満たすピクセルだけを処理
            mask_process = mask_valid & mask_positive & mask_depth
            
            if not mask_process.any():
                continue
            
            # 交点座標を計算
            intersections = origin.clone()
            intersections[mask_process] += ray_world_norm[mask_process] * t[mask_process].unsqueeze(-1)
            
            # バリセントリック座標を計算
            # 効率化：マスクを適用したテンソルで計算
            masked_intersections = intersections[mask_process]
            
            # 三角形の頂点をGPUテンソルに変換
            edge1 = v1 - v0
            edge2 = v2 - v0
            
            # マスク適用後の形状に合わせて拡張
            v0_expanded = v0.expand(masked_intersections.shape[0], 3)
            
            # バリセントリック座標の計算
            vp = masked_intersections - v0_expanded
            
            # 各ベクトルの内積を計算
            d00 = torch.sum(edge1 * edge1)
            d01 = torch.sum(edge1 * edge2)
            d11 = torch.sum(edge2 * edge2)
            d20 = torch.sum(vp * edge1, dim=1)
            d21 = torch.sum(vp * edge2, dim=1)
            
            # バリセントリック座標の係数を計算
            denom = d00 * d11 - d01 * d01
            inv_denom = 1.0 / denom
            
            v = (d11 * d20 - d01 * d21) * inv_denom
            w = (d00 * d21 - d01 * d20) * inv_denom
            u = 1.0 - v - w
            
            # 三角形の内部かどうかを判断
            mask_inside = (u >= 0) & (v >= 0) & (w >= 0)
            
            # 最終的なマスク
            final_mask = torch.zeros_like(mask_process, dtype=torch.bool, device=self.device)
            final_mask[mask_process] = mask_inside
            
            if not final_mask.any():
                continue
                
            # 処理するピクセルだけにマスクを適用
            intersections_flat = intersections[final_mask]
            t_flat = t[final_mask]
            
            # 各ピクセルのUV座標を計算
            uvs_flat = torch.zeros((intersections_flat.shape[0], 2), device=self.device)
            
            # バリセントリック座標から三角形の各頂点のUV座標を補間
            face_idx = self.triangle_face_mapping_gpu[tri_idx]
            face = self.faces[face_idx]
            
            # 三角形がどの面に属するか
            if tri_idx % 2 == 0:  # 最初の三角形
                uv0 = torch.tensor(self.uvs[0], device=self.device)
                uv1 = torch.tensor(self.uvs[1], device=self.device)
                uv2 = torch.tensor(self.uvs[2], device=self.device)
            else:  # 2番目の三角形
                uv0 = torch.tensor(self.uvs[0], device=self.device)
                uv1 = torch.tensor(self.uvs[2], device=self.device)
                uv2 = torch.tensor(self.uvs[3], device=self.device)
            
            # バリセントリック座標からUV座標を計算
            u_flat = u[mask_inside]
            v_flat = v[mask_inside]
            w_flat = w[mask_inside]
            
            # UV座標の補間
            uvs_flat = u_flat.unsqueeze(1) * uv0 + v_flat.unsqueeze(1) * uv1 + w_flat.unsqueeze(1) * uv2
            
            # テクスチャサンプリング
            texture = self.textures_gpu[face_idx]
            
            # UV座標からテクスチャの座標を計算
            uv_x = torch.clamp((uvs_flat[:, 0] * texture.shape[1]).long(), 0, texture.shape[1] - 1)
            uv_y = torch.clamp((uvs_flat[:, 1] * texture.shape[0]).long(), 0, texture.shape[0] - 1)
            
            # テクスチャの色を取得
            colors_flat = texture[uv_y, uv_x]
            
            # フレームと深度バッファを更新
            flat_indices = torch.nonzero(final_mask)
            
            # 最終的な色と深度の更新
            frame[flat_indices[:, 0], flat_indices[:, 1]] = colors_flat
            depth_buffer[flat_indices[:, 0], flat_indices[:, 1]] = t_flat
        
        # 深度マップをメートル単位の実距離に変換
        # カメラの位置からの実際の距離に
        actual_depth = depth_buffer.clone()
        
        # トーチテンソルをNumpyへ変換
        frame_np = (frame * 255).byte().cpu().numpy()
        depth_np = actual_depth.cpu().numpy()
        
        # 処理時間を計測
        end_time = time.time()
        print(f"レンダリング時間: {(end_time - start_time) * 1000:.2f}ms")
        
        return frame_np, depth_np, camera_pos, camera_quaternion
    
    def generate_sequence(self, num_frames=180, output_dir="tum_format_output"):
        """キューブの周りを周回するカメラのフレームシーケンスを生成"""
        # カメラパラメータの辞書を作成
        camera_params = {
            'width': self.width,
            'height': self.height,
            'fov': self.fov,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy
        }
        
        # TUM形式のデータ出力用クラスのインスタンス化
        tum_writer = TUMFormatWriter(output_dir, camera_params)
        
        # 総処理時間を計測
        total_start_time = time.time()
        
        for frame_idx in range(num_frames):
            # このフレームの角度を計算
            angle = 2 * math.pi * frame_idx / num_frames
            
            # フレームと深度マップをレンダリング（GPU使用）
            frame, depth_map, camera_pos, camera_quaternion = self.render_scene_gpu(angle)
            
            # TUM形式でフレーム、深度マップ、カメラ情報を出力
            tum_writer.write_frame(frame, depth_map, frame_idx, camera_pos, camera_quaternion)
            
            print(f"フレーム {frame_idx+1}/{num_frames} 生成完了")
        
        # 総処理時間を表示
        total_time = time.time() - total_start_time
        print(f"総処理時間: {total_time:.2f}秒 (平均 {total_time/num_frames:.2f}秒/フレーム)")
        print("TUMデータセット形式でのシーケンス生成が完了しました。")


if __name__ == "__main__":
    # GPU版レンダラーを使用
    renderer = GPURenderer(width=640, height=480, fov=60)
    renderer.generate_sequence(num_frames=180)  # 180フレーム生成（1フレームあたり2度）
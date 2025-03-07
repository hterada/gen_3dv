import numpy as np
import torch
import math
import time

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
        
        # シーンの初期化（後でセットする）
        self.scene = None
    
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
    
    def set_scene(self, scene):
        """
        レンダリングするシーンをセット
        
        Parameters:
        scene (Scene): レンダリングするシーンオブジェクト
        """
        self.scene = scene
    
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
    
    def render_scene_gpu(self, angle, vertices=None, triangles=None, triangle_face_mapping=None, textures_gpu=None, floor_texture_idx=None):
        """
        GPUを使用してシーンをレンダリング
        
        Parameters:
        angle (float): カメラの回転角度
        vertices (torch.Tensor, optional): 頂点データ。指定がなければシーン全体の頂点を使用
        triangles (torch.Tensor, optional): 三角形の頂点インデックス。指定がなければシーン全体のデータを使用
        triangle_face_mapping (torch.Tensor, optional): 三角形と面のマッピング。指定がなければシーン全体のデータを使用
        textures_gpu (list, optional): テクスチャデータのリスト。指定がなければシーン全体のデータを使用
        floor_texture_idx (int, optional): 床のテクスチャインデックス。指定がなければシーン全体のデータを使用
        
        Returns:
        tuple: (フレーム画像, 深度マップ, カメラ位置, カメラ四元数)
        """
        # 処理時間の計測を開始
        start_time = time.time()
        
        # シーンデータのチェック
        if self.scene is None and vertices is None:
            raise ValueError("シーンがセットされていないか、頂点データが指定されていません")
        
        # シーンデータの取得
        if vertices is None:
            vertices, triangles, triangle_face_mapping, textures_gpu = self.scene.get_gpu_data()
            floor_texture_idx = self.scene.floor_texture_idx
        
        if textures_gpu is None and self.scene is not None:
            textures_gpu = self.scene.textures_gpu
        
        # 円上のカメラ位置を計算
        radius = 8.0  # カメラの距離を増やして全体を見渡せるように
        camera_x = radius * math.sin(angle)
        camera_z = radius * math.cos(angle)
        camera_y = 2.5  # カメラの高さを調整
        camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)
        
        # シーンの中心を注視
        target_pos = np.array([0, 0, 0], dtype=np.float32)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        
        # ビュー行列の作成
        view_matrix = self.get_view_matrix(camera_pos, target_pos, up_vector)
        
        # GPU用に変換
        view_matrix_gpu = torch.tensor(view_matrix, device=self.device)
        inv_view_matrix_gpu = torch.inverse(view_matrix_gpu)
        
        # カメラ回転の四元数表現を取得
        camera_quaternion = self.quaternion_from_euler(0, -angle + math.pi, 0)
        
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
        
        # シーン全体の三角形を処理
        for tri_idx in range(len(triangles)):
            # 三角形の頂点インデックス
            v_idx = triangles[tri_idx]
            
            # 三角形の3つの頂点
            v0 = vertices[v_idx[0], :3]
            v1 = vertices[v_idx[1], :3]
            v2 = vertices[v_idx[2], :3]
            
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
            
            # テクスチャのインデックスを取得
            face_idx = triangle_face_mapping[tri_idx]
            
            # 三角形がどの面に属するか判断し、適切なUV座標を計算
            face_type = tri_idx % 2  # 0: 最初の三角形, 1: 2番目の三角形
            
            # UVマッピングで使う座標
            uvs = torch.tensor(self.scene.uvs if self.scene else [[0,0], [1,0], [1,1], [0,1]], device=self.device)
            
            if face_type == 0:  # 最初の三角形
                uv0 = uvs[0]
                uv1 = uvs[1]
                uv2 = uvs[2]
            else:  # 2番目の三角形
                uv0 = uvs[0]
                uv1 = uvs[2]
                uv2 = uvs[3]
            
            # バリセントリック座標からUV座標を計算
            u_flat = u[mask_inside]
            v_flat = v[mask_inside]
            w_flat = w[mask_inside]
            
            # UV座標の補間
            uvs_flat = u_flat.unsqueeze(1) * uv0 + v_flat.unsqueeze(1) * uv1 + w_flat.unsqueeze(1) * uv2
            
            # テクスチャサンプリング
            texture = textures_gpu[face_idx]
            
            # 床のテクスチャの場合、UV座標をタイリング（繰り返し）
            if floor_texture_idx is not None and face_idx == floor_texture_idx:
                uvs_flat = uvs_flat * 10  # 床のテクスチャを10x10回繰り返す
                uvs_flat = uvs_flat - uvs_flat.floor()  # 0-1の範囲にラップ
            
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
        """
        キューブの周りを周回するカメラのフレームシーケンスを生成
        
        Parameters:
        num_frames (int): 生成するフレーム数
        output_dir (str): 出力ディレクトリのパス
        """
        # シーンのチェック
        if self.scene is None:
            raise ValueError("シーンがセットされていません。先に set_scene() を呼び出してください。")
        
        # TUM形式の出力クラスをインポート
        from . import TUMFormatWriter
        
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
            # シーン全体のデータを使用
            frame, depth_map, camera_pos, camera_quaternion = self.render_scene_gpu(angle)
            
            # TUM形式でフレーム、深度マップ、カメラ情報を出力
            tum_writer.write_frame(frame, depth_map, frame_idx, camera_pos, camera_quaternion)
            
            print(f"フレーム {frame_idx+1}/{num_frames} 生成完了")
        
        # 総処理時間を表示
        total_time = time.time() - total_start_time
        print(f"総処理時間: {total_time:.2f}秒 (平均 {total_time/num_frames:.2f}秒/フレーム)")
        print("TUMデータセット形式でのシーケンス生成が完了しました。")
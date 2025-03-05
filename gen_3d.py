import numpy as np
import cv2
import os
import math
from PIL import Image, ImageDraw
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
        depth_map (numpy.ndarray): 出力する深度画像データ
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
        # TUMフォーマット: 深度値は16ビット画像で、単位はm(メートル)の1000倍
        scaled_depth = (depth_map * 1000).astype(np.uint16)
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


class Renderer:
    def __init__(self, width=640, height=480, fov=60):
        self.width = width
        self.height = height
        self.fov = fov
        self.aspect_ratio = width / height
        self.near = 0.1
        self.far = 100
        
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
        ])
    
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
        ])
        
        # 面（各面は4つの頂点で構成、反時計回りに指定）
        self.faces = [
            [0, 1, 2, 3],  # 前面
            [5, 4, 7, 6],  # 背面
            [1, 5, 6, 2],  # 右面
            [4, 0, 3, 7],  # 左面
            [3, 2, 6, 7],  # 上面
            [1, 0, 4, 5]   # 底面
        ]
        
        # テクスチャマッピング用のUV座標
        self.uvs = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        
        # 各面に異なるテクスチャを作成（異なる色の市松模様）
        self.textures = []
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
    
    def get_rotation_matrix(self, roll, pitch, yaw):
        """オイラー角から回転行列を作成"""
        # ロール（X）、ピッチ（Y）、ヨー（Z）
        # まず各軸の回転行列を作成
        Rx = np.array([
            [1, 0, 0, 0],
            [0, math.cos(roll), -math.sin(roll), 0],
            [0, math.sin(roll), math.cos(roll), 0],
            [0, 0, 0, 1]
        ])
        
        Ry = np.array([
            [math.cos(pitch), 0, math.sin(pitch), 0],
            [0, 1, 0, 0],
            [-math.sin(pitch), 0, math.cos(pitch), 0],
            [0, 0, 0, 1]
        ])
        
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0, 0],
            [math.sin(yaw), math.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 回転を組み合わせる: R = Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    def get_view_matrix(self, camera_pos, target_pos, up_vector):
        """カメラ位置、ターゲット、上ベクトルからビュー行列を作成"""
        # カメラ座標系の作成
        z_axis = camera_pos - target_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(up_vector, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        # ビュー行列の作成
        view_matrix = np.eye(4)
        view_matrix[0, :3] = x_axis
        view_matrix[1, :3] = y_axis
        view_matrix[2, :3] = z_axis
        
        # 移動成分
        view_matrix[0, 3] = -np.dot(x_axis, camera_pos)
        view_matrix[1, 3] = -np.dot(y_axis, camera_pos)
        view_matrix[2, 3] = -np.dot(z_axis, camera_pos)
        
        return view_matrix
    
    def is_face_visible(self, transformed_vertices, face_indices):
        """裏面カリングを用いて面がカメラから見えるかチェック"""
        # 面から3つの頂点を取得
        v0 = transformed_vertices[face_indices[0]]
        v1 = transformed_vertices[face_indices[1]]
        v2 = transformed_vertices[face_indices[2]]
        
        # 外積を使って面の法線を計算
        edge1 = v1[:3] - v0[:3]
        edge2 = v2[:3] - v0[:3]
        normal = np.cross(edge1, edge2)
        
        # 法線がカメラ方向を向いているか確認（カメラ空間ではzが負）
        return normal[2] < 0
    
    def render_scene(self, angle):
        """キューブの周りを周回するカメラからシーンをレンダリング"""
        # 白色背景の作成
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # 深度バッファの初期化（遠い距離で初期化）
        depth_buffer = np.ones((self.height, self.width)) * self.far
        
        # 円上のカメラ位置を計算
        radius = 3.0
        camera_x = radius * math.sin(angle)
        camera_z = radius * math.cos(angle)
        camera_pos = np.array([camera_x, 0, camera_z])
        
        # 原点（キューブの位置）を注視
        target_pos = np.array([0, 0, 0])
        up_vector = np.array([0, 1, 0])
        
        # ビュー行列の作成
        view_matrix = self.get_view_matrix(camera_pos, target_pos, up_vector)
        
        # カメラ回転の四元数表現を取得
        # TUM形式のために、カメラからワールドへの変換をワールドからカメラへの変換に
        # 原点を見ている(x,0,z)位置のカメラの場合、回転はY軸周りの回転
        camera_quaternion = self.quaternion_from_euler(0, -angle + math.pi, 0)
        
        # 頂点をオブジェクト空間からカメラ空間に変換
        transformed_vertices = np.zeros_like(self.vertices)
        camera_space_vertices = np.zeros_like(self.vertices)
        
        for i, vertex in enumerate(self.vertices):
            # カメラ空間への変換
            camera_vertex = view_matrix @ vertex
            camera_space_vertices[i] = camera_vertex
            
            # 射影変換
            transformed = self.projection_matrix @ camera_vertex
            
            # 透視除算
            if transformed[3] != 0:
                transformed = transformed / transformed[3]
            
            transformed_vertices[i] = transformed
        
        # 深度による面のソート（簡易的なペインターアルゴリズム）
        face_depths = []
        for i, face in enumerate(self.faces):
            # 面の平均Z深度を計算
            z_depth = sum(transformed_vertices[idx][2] for idx in face) / 4
            face_depths.append((i, z_depth))
        
        # 面を奥から手前にソート
        face_depths.sort(key=lambda x: x[1], reverse=True)
        
        # 各面をレンダリング
        for face_idx, _ in face_depths:
            face = self.faces[face_idx]
            
            # カメラから見えない面はスキップ
            if not self.is_face_visible(transformed_vertices, face):
                continue
            
            # 面の頂点のスクリーン座標を取得
            screen_coords = []
            for idx in face:
                x = (transformed_vertices[idx][0] + 1) * self.width / 2
                y = (1 - transformed_vertices[idx][1]) * self.height / 2
                screen_coords.append((int(x), int(y)))
            
            # 面のマスクを作成
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(screen_coords)], 255)
            
            # この面のテクスチャを取得
            texture = self.textures[face_idx]
            
            # 簡易的な透視補正テクスチャマッピング
            for y in range(self.height):
                for x in range(self.width):
                    if mask[y, x] == 0:
                        continue
                    
                    # バリセントリック座標の計算
                    # これは単純化されたアプローチで長方形を使用
                    # 完全なレンダラーでは、適切なバリセントリック座標を使用する
                    
                    # UV座標の簡易的なバイリニア補間
                    min_x = min(c[0] for c in screen_coords)
                    max_x = max(c[0] for c in screen_coords)
                    min_y = min(c[1] for c in screen_coords)
                    max_y = max(c[1] for c in screen_coords)
                    
                    # ゼロ除算を回避
                    width = max(1, max_x - min_x)
                    height = max(1, max_y - min_y)
                    
                    # UV座標の計算
                    u = (x - min_x) / width
                    v = (y - min_y) / height
                    
                    # 深度値の補間（簡易的）
                    # 正確なレンダラーでは、バリセントリック座標を使用してZ値を補間する
                    depth_sum = 0
                    weight_sum = 0
                    
                    for idx in face:
                        # スクリーン座標を取得
                        sx = int((transformed_vertices[idx][0] + 1) * self.width / 2)
                        sy = int((1 - transformed_vertices[idx][1]) * self.height / 2)
                        
                        # 現在のピクセルとの距離
                        dist = math.sqrt((x - sx) ** 2 + (y - sy) ** 2) + 0.0001  # ゼロ除算回避
                        weight = 1 / dist
                        
                        # カメラ空間でのZ値（負の値）
                        z_value = -camera_space_vertices[idx][2]
                        
                        depth_sum += z_value * weight
                        weight_sum += weight
                    
                    # 補間された深度値
                    interpolated_depth = depth_sum / weight_sum
                    
                    # 深度テストと更新
                    if interpolated_depth < depth_buffer[y, x]:
                        # テクスチャの色を取得
                        tx = min(int(u * texture.shape[1]), texture.shape[1] - 1)
                        ty = min(int(v * texture.shape[0]), texture.shape[0] - 1)
                        
                        frame[y, x] = texture[ty, tx]
                        depth_buffer[y, x] = interpolated_depth
        
        # 深度マップを0-1の範囲に正規化
        normalized_depth = np.zeros_like(depth_buffer)
        depth_min = np.min(depth_buffer[depth_buffer < self.far])
        depth_max = np.max(depth_buffer[depth_buffer < self.far])
        
        # 背景（最大深度の値）を処理
        mask = depth_buffer < self.far
        if np.any(mask):  # マスクが空でないことを確認
            normalized_depth[mask] = (depth_buffer[mask] - depth_min) / (depth_max - depth_min)
        
        # 背景を最大深度に設定（背景は無限遠）
        normalized_depth[depth_buffer >= self.far] = 1.0
        
        return frame, normalized_depth, camera_pos, camera_quaternion
    
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
        
        for frame_idx in range(num_frames):
            # このフレームの角度を計算
            angle = 2 * math.pi * frame_idx / num_frames
            
            # フレームと深度マップをレンダリング
            frame, depth_map, camera_pos, camera_quaternion = self.render_scene(angle)
            
            # TUM形式でフレーム、深度マップ、カメラ情報を出力
            tum_writer.write_frame(frame, depth_map, frame_idx, camera_pos, camera_quaternion)
            
            print(f"フレーム {frame_idx+1}/{num_frames} 生成完了")


if __name__ == "__main__":
    renderer = Renderer(width=640, height=480, fov=60)
    renderer.generate_sequence(num_frames=180)  # 180フレーム生成（1フレームあたり2度）
    print("TUMデータセット形式でのシーケンス生成が完了しました。")
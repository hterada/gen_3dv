import numpy as np
import torch
import os
from PIL import Image, ImageDraw

class Scene:
    """シーンデータを管理するクラス"""
    
    def __init__(self, device=None):
        """
        初期化関数
        
        Parameters:
        device (torch.device, optional): 使用するデバイス。指定がなければGPUが利用可能なら使用
        """
        # デバイスの設定
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # シーンデータの初期化
        self.vertices = []         # 頂点データ (CPU)
        self.triangles = []        # 三角形の頂点インデックス (CPU)
        self.triangle_face_mapping = []  # 三角形と面のマッピング (CPU)
        self.textures = []         # テクスチャデータ (CPU)
        
        # GPUデータ
        self.vertices_gpu = None
        self.triangles_gpu = None
        self.triangle_face_mapping_gpu = None
        self.textures_gpu = []     # テクスチャデータ (GPU)
        
        # UV座標の設定
        self.uvs = np.array([
            [0, 0],  # 左下
            [1, 0],  # 右下
            [1, 1],  # 右上
            [0, 1]   # 左上
        ], dtype=np.float32)
        
        # フロアテクスチャのインデックス
        self.floor_texture_idx = None
    
    def add_cube(self, position=(0, 0, 0), size=1.0, texture_indices=None):
        """
        シーンに立方体を追加
        
        Parameters:
        position (tuple): 立方体の中心位置 (x, y, z)
        size (float): 立方体の一辺の長さ
        texture_indices (list): 各面に適用するテクスチャのインデックス（6面分）
        
        Returns:
        int: 追加された頂点の開始インデックス
        """
        # 立方体の頂点を生成
        half_size = size / 2
        x, y, z = position
        
        cube_vertices = np.array([
            [x-half_size, y-half_size, z-half_size, 1.0],  # 0
            [x+half_size, y-half_size, z-half_size, 1.0],  # 1
            [x+half_size, y+half_size, z-half_size, 1.0],  # 2
            [x-half_size, y+half_size, z-half_size, 1.0],  # 3
            [x-half_size, y-half_size, z+half_size, 1.0],  # 4
            [x+half_size, y-half_size, z+half_size, 1.0],  # 5
            [x+half_size, y+half_size, z+half_size, 1.0],  # 6
            [x-half_size, y+half_size, z+half_size, 1.0]   # 7
        ], dtype=np.float32)
        
        # 面の定義（頂点インデックス）
        faces = [
            [0, 1, 2, 3],  # 前面
            [5, 4, 7, 6],  # 背面
            [1, 5, 6, 2],  # 右面
            [4, 0, 3, 7],  # 左面
            [3, 2, 6, 7],  # 上面
            [1, 0, 4, 5]   # 底面
        ]
        
        # 現在の頂点数（オフセット）
        vertex_offset = len(self.vertices)
        
        # 頂点を追加
        self.vertices.extend(cube_vertices)
        
        # テクスチャインデックスのデフォルト値
        if texture_indices is None:
            texture_indices = list(range(6))  # 0〜5のインデックス
        
        # 各面を2つの三角形に分割して追加
        for face_idx, face in enumerate(faces):
            # テクスチャインデックス
            tex_idx = texture_indices[face_idx]
            
            # 面を三角形に分割
            tri1 = [vertex_offset + face[0], vertex_offset + face[1], vertex_offset + face[2]]
            tri2 = [vertex_offset + face[0], vertex_offset + face[2], vertex_offset + face[3]]
            
            self.triangles.append(tri1)
            self.triangle_face_mapping.append(tex_idx)
            
            self.triangles.append(tri2)
            self.triangle_face_mapping.append(tex_idx)
        
        return vertex_offset
    
    def add_floor(self, position=(0, 0, 0), size=20.0, texture_idx=None):
        """
        シーンに床を追加
        
        Parameters:
        position (tuple): 床の中心位置 (x, y, z)
        size (float): 床の一辺の長さ
        texture_idx (int): 床に適用するテクスチャのインデックス
        
        Returns:
        int: テクスチャインデックス
        """
        # 床の中心位置を取得
        x, y, z = position
        half_size = size / 2
        
        # 床の頂点を生成
        floor_vertices = np.array([
            [x-half_size, y, z-half_size, 1.0],  # 左下奥
            [x+half_size, y, z-half_size, 1.0],  # 右下奥
            [x+half_size, y, z+half_size, 1.0],  # 右下手前
            [x-half_size, y, z+half_size, 1.0],  # 左下手前
        ], dtype=np.float32)
        
        # 現在の頂点数（オフセット）
        vertex_offset = len(self.vertices)
        
        # 頂点を追加
        self.vertices.extend(floor_vertices)
        
        # テクスチャインデックスの設定
        if texture_idx is None:
            texture_idx = len(self.textures) - 1  # 最後のテクスチャを使用
        
        self.floor_texture_idx = texture_idx
        
        # 床を三角形に分割して追加
        # 三角形1
        tri1 = [vertex_offset + 0, vertex_offset + 1, vertex_offset + 2]
        self.triangles.append(tri1)
        self.triangle_face_mapping.append(texture_idx)
        
        # 三角形2
        tri2 = [vertex_offset + 0, vertex_offset + 2, vertex_offset + 3]
        self.triangles.append(tri2)
        self.triangle_face_mapping.append(texture_idx)
        
        return texture_idx
    
    def create_checkboard_texture(self, width, height, color1, color2, squares=8):
        """
        市松模様のテクスチャを作成
        
        Parameters:
        width (int): テクスチャの幅
        height (int): テクスチャの高さ
        color1 (tuple): 一つ目の色 (R, G, B)
        color2 (tuple): 二つ目の色 (R, G, B)
        squares (int): 市松模様の分割数
        
        Returns:
        int: 追加されたテクスチャのインデックス
        """
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
        
        texture = np.array(image)
        self.textures.append(texture)
        
        # GPUテクスチャも追加
        texture_gpu = torch.tensor(texture, device=self.device).float() / 255.0
        self.textures_gpu.append(texture_gpu)
        
        return len(self.textures) - 1
    
    def load_texture_from_file(self, file_path):
        """
        画像ファイルからテクスチャを読み込む
        
        Parameters:
        file_path (str): 画像ファイルのパス
        
        Returns:
        int: 追加されたテクスチャのインデックス、失敗した場合は-1
        """
        try:
            if os.path.exists(file_path):
                # 画像ファイルの読み込み
                image = Image.open(file_path)
                # リサイズ（オプション）
                image = image.resize((512, 512))
                texture = np.array(image)
                
                self.textures.append(texture)
                
                # GPUテクスチャも追加
                texture_gpu = torch.tensor(texture, device=self.device).float() / 255.0
                self.textures_gpu.append(texture_gpu)
                
                print(f"テクスチャを読み込みました: {file_path}")
                return len(self.textures) - 1
            else:
                print(f"テクスチャファイルが見つかりません: {file_path}")
                return -1
        except Exception as e:
            print(f"テクスチャ読み込みエラー: {e}")
            return -1
    
    def create_default_textures(self):
        """
        デフォルトの立方体テクスチャを生成
        
        Returns:
        list: テクスチャインデックスのリスト
        """
        texture_indices = []
        
        # 立方体の各面に異なるテクスチャを作成
        colors = [
            ((255, 0, 0), (200, 0, 0)),       # 赤
            ((0, 255, 0), (0, 200, 0)),       # 緑
            ((0, 0, 255), (0, 0, 200)),       # 青
            ((255, 255, 0), (200, 200, 0)),   # 黄
            ((0, 255, 255), (0, 200, 200)),   # シアン
            ((255, 0, 255), (200, 0, 200))    # マゼンタ
        ]
        
        for color1, color2 in colors:
            idx = self.create_checkboard_texture(128, 128, color1, color2)
            texture_indices.append(idx)
        
        return texture_indices
    
    def create_default_floor_texture(self, squares=4):
        """
        デフォルトの床テクスチャを生成
        
        Parameters:
        squares (int): 市松模様の分割数（数が少ないほど大きな市松模様になる）
        
        Returns:
        int: テクスチャインデックス
        """
        return self.create_checkboard_texture(512, 512, (120, 120, 120), (80, 80, 80), squares=squares)
    
    def create_multiple_cubes_scene(self, rows=3, cols=3, spacing=2.0, floor_texture_path=None, floor_checker_size=4):
        """
        複数の立方体と床からなるシーンを作成
        
        Parameters:
        rows (int): 立方体の行数
        cols (int): 立方体の列数
        spacing (float): 立方体間の間隔
        floor_texture_path (str): 床のテクスチャ画像のパス
        floor_checker_size (int): 床の市松模様の分割数（数が少ないほど大きな市松模様になる）
        """
        # デフォルトのテクスチャを作成
        cube_textures = self.create_default_textures()
        
        # 床のテクスチャを設定
        if floor_texture_path and os.path.exists(floor_texture_path):
            floor_tex_idx = self.load_texture_from_file(floor_texture_path)
            if floor_tex_idx == -1:
                # 読み込み失敗時はデフォルトテクスチャを使用
                floor_tex_idx = self.create_default_floor_texture(squares=floor_checker_size)
        else:
            floor_tex_idx = self.create_default_floor_texture(squares=floor_checker_size)
        
        # 床のY座標
        floor_y = -1.0
        
        # 床を追加
        self.add_floor((0, floor_y, 0), size=20.0, texture_idx=floor_tex_idx)
        
        # 立方体のサイズ
        cube_size = 1.0
        
        # 複数の立方体を配置
        for row in range(rows):
            for col in range(cols):
                # 立方体の位置を計算
                x = (col - (cols-1)/2) * spacing
                z = (row - (rows-1)/2) * spacing
                
                # 床の上に立方体を配置 - 立方体の底面が床に接するように y 座標を調整
                # 立方体のY座標は中心なので、床のY + 立方体の半分の高さ
                y = floor_y + (cube_size / 2)
                
                self.add_cube((x, y, z), size=cube_size, texture_indices=cube_textures)
        
        print(f"{rows}x{cols}={rows*cols}個の立方体と床を配置したシーンを作成しました")
        print(f"床のY座標: {floor_y}, 立方体のY座標: {floor_y + (cube_size / 2)} (サイズ: {cube_size})")
        
        # データをGPUに転送
        self.to_gpu()
    
    def to_gpu(self):
        """シーンデータをGPUに転送"""
        self.vertices_gpu = torch.tensor(np.array(self.vertices), device=self.device)
        self.triangles_gpu = torch.tensor(np.array(self.triangles), device=self.device)
        self.triangle_face_mapping_gpu = torch.tensor(np.array(self.triangle_face_mapping), device=self.device)
        
        print(f"シーンデータをGPUに転送: {len(self.vertices)}頂点, {len(self.triangles)}三角形")
    
    def get_gpu_data(self):
        """
        GPUデータを取得
        
        Returns:
        tuple: (vertices_gpu, triangles_gpu, triangle_face_mapping_gpu, textures_gpu)
        """
        return (
            self.vertices_gpu, 
            self.triangles_gpu, 
            self.triangle_face_mapping_gpu, 
            self.textures_gpu
        )
    
    def get_stats(self):
        """
        シーンの統計情報を取得
        
        Returns:
        dict: 統計情報
        """
        return {
            'vertices': len(self.vertices),
            'triangles': len(self.triangles),
            'textures': len(self.textures)
        }
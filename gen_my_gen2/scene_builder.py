import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import cv2

class SceneBuilder:
    def __init__(self, renderer, use_gpu=True):
        """
        3Dシーン構築クラス
        
        Args:
            renderer: Rendererオブジェクト
            use_gpu (bool): GPUを使用するかどうか
        """
        self.renderer = renderer
        self.use_gpu = use_gpu
        self.scene_objects = []
        
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # テクスチャとカラーの作成用ニューラルネットワーク
        self.texture_generator = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        ).to(self.device)
        
        # 初期化後にウェイトを設定
        self._initialize_texture_generator()
        
        # テクスチャディレクトリ作成
        self.texture_dir = Path("textures")
        self.texture_dir.mkdir(exist_ok=True)
    
    def _initialize_texture_generator(self):
        """
        テクスチャジェネレータの初期化
        """
        # ランダムなウェイトを設定
        for layer in self.texture_generator.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)
    
    def _generate_procedural_texture(self, texture_id, size=512, pattern_type="noise"):
        """
        手続き的テクスチャを生成する
        
        Args:
            texture_id (str): テクスチャのID
            size (int): テクスチャのサイズ
            pattern_type (str): テクスチャのパターンタイプ
        """
        texture_path = self.texture_dir / f"{texture_id}.png"
        
        if texture_path.exists():
            # 既に存在する場合はロードする
            self.renderer.load_texture(str(texture_path), texture_id)
            return
        
        # GPUを使った高速なテクスチャ生成
        if self.use_gpu:
            if pattern_type == "noise":
                # ノイズテクスチャ
                texture = torch.randn(size, size, 4, device=self.device)
                # 値を0-1の範囲に正規化
                texture = (texture - texture.min()) / (texture.max() - texture.min())
                texture[..., 3] = 1.0  # アルファチャンネルを1に設定
                
            elif pattern_type == "grid":
                # グリッドテクスチャ
                x = torch.linspace(0, 1, size, device=self.device)
                y = torch.linspace(0, 1, size, device=self.device)
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                
                # グリッドラインを作成
                grid_width = 0.05
                grid_x = torch.abs((xx % 0.2) - 0.1) < grid_width
                grid_y = torch.abs((yy % 0.2) - 0.1) < grid_width
                grid = grid_x | grid_y
                
                # RGBAテクスチャ作成
                texture = torch.zeros(size, size, 4, device=self.device)
                # ベースカラー
                base_color = torch.tensor([0.8, 0.8, 0.8], device=self.device)
                # グリッドカラー
                grid_color = torch.tensor([0.2, 0.2, 0.2], device=self.device)
                
                for i in range(3):
                    texture[..., i] = torch.where(grid, grid_color[i], base_color[i])
                
                texture[..., 3] = 1.0  # アルファチャンネルを1に設定
                
            elif pattern_type == "checkerboard":
                # チェッカーボードテクスチャ
                x = torch.linspace(0, 1, size, device=self.device)
                y = torch.linspace(0, 1, size, device=self.device)
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                
                # チェッカーパターンを作成
                checker_size = 0.1
                checker = ((xx.floor() + yy.floor()) % 2 == 0)
                
                # RGBAテクスチャ作成
                texture = torch.zeros(size, size, 4, device=self.device)
                # カラー1
                color1 = torch.tensor([0.9, 0.9, 0.9], device=self.device)
                # カラー2
                color2 = torch.tensor([0.2, 0.2, 0.2], device=self.device)
                
                for i in range(3):
                    texture[..., i] = torch.where(checker, color1[i], color2[i])
                
                texture[..., 3] = 1.0  # アルファチャンネルを1に設定
                
            else:  # パターンが指定されていない場合はニューラルネットワークで生成
                x = torch.linspace(0, 1, size, device=self.device)
                y = torch.linspace(0, 1, size, device=self.device)
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                
                # 入力座標を作成
                coords = torch.stack([xx, yy], dim=-1).view(-1, 2)
                
                # バッチサイズを小さくして処理
                batch_size = 1024
                texture_rgb = []
                
                for i in range(0, coords.size(0), batch_size):
                    batch = coords[i:i+batch_size]
                    # ニューラルネットワークでテクスチャを生成
                    with torch.no_grad():  # 勾配計算を無効化
                        rgb = self.texture_generator(batch)
                    texture_rgb.append(rgb)
                
                texture_rgb = torch.cat(texture_rgb, dim=0).view(size, size, 3)
                
                # アルファチャンネルを追加
                alpha = torch.ones(size, size, 1, device=self.device)
                texture = torch.cat([texture_rgb, alpha], dim=-1)
            
            # NumPy配列に変換
            texture_np = (texture.detach().cpu().numpy() * 255).astype(np.uint8)
            
        else:
            # CPUでの処理
            if pattern_type == "noise":
                # ノイズテクスチャ
                texture_np = np.random.rand(size, size, 4) * 255
                texture_np[:, :, 3] = 255  # アルファチャンネルを255に設定
                
            elif pattern_type == "grid":
                # グリッドテクスチャ
                texture_np = np.ones((size, size, 4), dtype=np.uint8) * 255
                grid_width = int(size * 0.01)
                grid_spacing = int(size * 0.1)
                
                for i in range(0, size, grid_spacing):
                    texture_np[i:i+grid_width, :, :3] = 50
                    texture_np[:, i:i+grid_width, :3] = 50
                
            elif pattern_type == "checkerboard":
                # チェッカーボードテクスチャ
                texture_np = np.ones((size, size, 4), dtype=np.uint8) * 255
                check_size = int(size * 0.1)
                
                for i in range(0, size, check_size * 2):
                    for j in range(0, size, check_size * 2):
                        texture_np[i:i+check_size, j:j+check_size, :3] = 50
                        texture_np[i+check_size:i+check_size*2, j+check_size:j+check_size*2, :3] = 50
                
            else:
                # 単純なグラデーションテクスチャ
                x = np.linspace(0, 1, size)
                y = np.linspace(0, 1, size)
                xx, yy = np.meshgrid(x, y)
                
                texture_np = np.zeros((size, size, 4), dtype=np.uint8)
                texture_np[:, :, 0] = (xx * 255).astype(np.uint8)
                texture_np[:, :, 1] = (yy * 255).astype(np.uint8)
                texture_np[:, :, 2] = ((1 - xx - yy) * 255).astype(np.uint8)
                texture_np[:, :, 3] = 255
        
        # テクスチャを保存
        img = Image.fromarray(texture_np)
        img.save(texture_path)
        
        # レンダラーにテクスチャをロード
        self.renderer.load_texture(str(texture_path), texture_id)
    
    def build_scene(self):
        """
        シーンを構築する
        
        Returns:
            list: シーンオブジェクトのリスト
            tuple: 光源の位置
        """
        self.scene_objects = []
        
        # 部屋のサイズ
        room_size = 10.0
        
        # 1. 部屋の作成（6面それぞれに異なる色を設定）
        room_vao = self.renderer.create_room_vao(room_size)
        
        # 部屋の壁の色（それぞれ異なる色）
        wall_colors = [
            (0.9, 0.2, 0.2),  # 赤（前面）
            (0.2, 0.2, 0.9),  # 青（後面）
            (0.8, 0.8, 0.8),  # 白（上面）
            (0.2, 0.9, 0.2),  # 緑（下面）
            (0.9, 0.9, 0.2),  # 黄（右面）
            (0.9, 0.2, 0.9),  # 紫（左面）
        ]
        
        # 部屋の各面のテクスチャIDを生成
        for i, color in enumerate(wall_colors):
            wall_texture_id = f"wall_{i}"
            self.renderer.create_color_texture(color, wall_texture_id)
        
        # 部屋の6面をそれぞれ追加
        faces = ["front", "back", "top", "bottom", "right", "left"]
        for i, face in enumerate(faces):
            model_matrix = np.eye(4)
            
            # 面によって位置を調整
            if face == "front":
                model_matrix[2, 3] = room_size / 2
            elif face == "back":
                model_matrix[2, 3] = -room_size / 2
            elif face == "top":
                model_matrix[1, 3] = room_size / 2
            elif face == "bottom":
                model_matrix[1, 3] = -room_size / 2
            elif face == "right":
                model_matrix[0, 3] = room_size / 2
            elif face == "left":
                model_matrix[0, 3] = -room_size / 2
            
            wall_obj = {
                'name': f"wall_{face}",
                'vao': room_vao,
                'model_matrix': model_matrix,
                'color': wall_colors[i],
                'texture_id': f"wall_{i}"
            }
            
            self.scene_objects.append(wall_obj)
        
        # 2. 中心の直方体の作成
        cube_size = 1.0
        cube_vao = self.renderer.create_cube_vao(cube_size)
        
        # 直方体の6面に異なるテクスチャを生成
        texture_patterns = [
            "noise", "grid", "checkerboard", 
            "neural", "neural", "neural"
        ]
        
        for i, pattern in enumerate(texture_patterns):
            texture_id = f"cube_texture_{i}"
            self._generate_procedural_texture(texture_id, size=512, pattern_type=pattern)
        
        # 直方体オブジェクトの追加
        cube_obj = {
            'name': "center_cube",
            'vao': cube_vao,
            'model_matrix': np.eye(4),  # 原点に配置
            'texture_id': "cube_texture_0"  # 最初のテクスチャを使用
        }
        
        self.scene_objects.append(cube_obj)
        
        # 3. 光源位置
        light_pos = (0, room_size / 2 - 0.1, 0)  # 天井のすぐ下
        
        return self.scene_objects, light_pos
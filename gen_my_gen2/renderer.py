import numpy as np
import torch
import torch.nn as nn
import moderngl
import cv2
import os
from pathlib import Path
from PIL import Image

class Renderer:
    def __init__(self, width=640, height=480, use_gpu=True):
        """
        OpenGLを使用した3Dレンダラー
        
        Args:
            width (int): 出力画像の幅
            height (int): 出力画像の高さ
            use_gpu (bool): GPUを使用するかどうか
        """
        self.width = width
        self.height = height
        self.use_gpu = use_gpu
        self.ctx = moderngl.create_standalone_context()
        
        # シェーダープログラム
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                
                uniform mat4 model;
                uniform mat4 view;
                uniform mat4 projection;
                
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;
                
                out vec3 normal;
                out vec3 fragPos;
                out vec2 uv;
                
                void main() {
                    uv = in_texcoord_0;
                    normal = mat3(transpose(inverse(model))) * in_normal;
                    fragPos = vec3(model * vec4(in_position, 1.0));
                    gl_Position = projection * view * model * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                
                uniform vec3 lightPos;
                uniform vec3 viewPos;
                uniform sampler2D texture0;
                uniform bool has_texture;
                uniform vec3 objectColor;
                
                in vec3 normal;
                in vec3 fragPos;
                in vec2 uv;
                
                out vec4 fragColor;
                
                void main() {
                    // ライティング計算
                    vec3 norm = normalize(normal);
                    vec3 lightDir = normalize(lightPos - fragPos);
                    
                    // 環境光
                    float ambientStrength = 0.3;
                    vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
                    
                    // 拡散反射
                    float diff = max(dot(norm, lightDir), 0.0);
                    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
                    
                    // 鏡面反射
                    float specularStrength = 0.5;
                    vec3 viewDir = normalize(viewPos - fragPos);
                    vec3 reflectDir = reflect(-lightDir, norm);
                    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
                    vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
                    
                    // 最終的な色の計算
                    vec3 baseColor;
                    if (has_texture) {
                        baseColor = texture(texture0, uv).rgb;
                    } else {
                        baseColor = objectColor;
                    }
                    
                    vec3 result = (ambient + diffuse + specular) * baseColor;
                    fragColor = vec4(result, 1.0);
                }
            '''
        )
        
        # フレームバッファを設定
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)],
            depth_attachment=self.ctx.depth_texture((width, height))
        )
        
        self.textures = {}
        
        # GPU設定
        if self.use_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
            
        # 画像処理モデル
        self.post_processor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to(self.device)
    
    def load_texture(self, texture_path, texture_id):
        """
        テクスチャをロードする
        
        Args:
            texture_path (str): テクスチャファイルのパス
            texture_id (str): テクスチャのID
        """
        img = Image.open(texture_path).convert('RGBA')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        texture = self.ctx.texture(img.size, 4, img.tobytes())
        texture.build_mipmaps()
        self.textures[texture_id] = texture
    
    def create_color_texture(self, color, texture_id, size=(64, 64)):
        """
        単色テクスチャを作成する
        
        Args:
            color (tuple): RGB色(0-1)
            texture_id (str): テクスチャのID
            size (tuple): テクスチャサイズ
        """
        r, g, b = color
        data = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        data[:, :, 0] = int(r * 255)
        data[:, :, 1] = int(g * 255)
        data[:, :, 2] = int(b * 255)
        data[:, :, 3] = 255
        
        texture = self.ctx.texture(size, 4, data.tobytes())
        texture.build_mipmaps()
        self.textures[texture_id] = texture
    
    def create_cube_vao(self, size=1.0):
        """
        立方体のVAOを作成する
        
        Args:
            size (float): 立方体のサイズ
            
        Returns:
            moderngl.VertexArray: 立方体のVAO
        """
        half = size / 2
        
        # 立方体の頂点座標
        vertices = np.array([
            # 前面
            -half, -half, half,  0, 0, 1,  0, 0,  # 頂点0: 座標, 法線, テクスチャ座標
            half, -half, half,   0, 0, 1,  1, 0,  # 頂点1
            half, half, half,    0, 0, 1,  1, 1,  # 頂点2
            -half, half, half,   0, 0, 1,  0, 1,  # 頂点3
            
            # 後面
            -half, -half, -half, 0, 0, -1, 1, 0,
            -half, half, -half,  0, 0, -1, 1, 1,
            half, half, -half,   0, 0, -1, 0, 1,
            half, -half, -half,  0, 0, -1, 0, 0,
            
            # 上面
            -half, half, -half,  0, 1, 0,  0, 0,
            -half, half, half,   0, 1, 0,  0, 1,
            half, half, half,    0, 1, 0,  1, 1,
            half, half, -half,   0, 1, 0,  1, 0,
            
            # 下面
            -half, -half, -half, 0, -1, 0, 0, 1,
            half, -half, -half,  0, -1, 0, 1, 1,
            half, -half, half,   0, -1, 0, 1, 0,
            -half, -half, half,  0, -1, 0, 0, 0,
            
            # 右面
            half, -half, -half,  1, 0, 0,  0, 0,
            half, half, -half,   1, 0, 0,  0, 1,
            half, half, half,    1, 0, 0,  1, 1,
            half, -half, half,   1, 0, 0,  1, 0,
            
            # 左面
            -half, -half, -half, -1, 0, 0, 1, 0,
            -half, -half, half,  -1, 0, 0, 0, 0,
            -half, half, half,   -1, 0, 0, 0, 1,
            -half, half, -half,  -1, 0, 0, 1, 1,
        ], dtype='f4')
        
        # インデックス
        indices = np.array([
            0, 1, 2, 2, 3, 0,       # 前面
            4, 5, 6, 6, 7, 4,       # 後面
            8, 9, 10, 10, 11, 8,    # 上面
            12, 13, 14, 14, 15, 12, # 下面
            16, 17, 18, 18, 19, 16, # 右面
            20, 21, 22, 22, 23, 20, # 左面
        ], dtype='i4')
        
        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        
        vao = self.ctx.vertex_array(self.prog, [
            (vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')
        ], ibo)
        
        return vao
    
    def create_room_vao(self, size=10.0):
        """
        部屋（立方体の内側）のVAOを作成する
        
        Args:
            size (float): 部屋のサイズ
            
        Returns:
            moderngl.VertexArray: 部屋のVAO
        """
        half = size / 2
        
        # 部屋の頂点座標（立方体の内側）
        vertices = np.array([
            # 前面 (内側向き)
            half, -half, half,    0, 0, -1,  0, 0,
            -half, -half, half,   0, 0, -1,  1, 0,
            -half, half, half,    0, 0, -1,  1, 1,
            half, half, half,     0, 0, -1,  0, 1,
            
            # 後面 (内側向き)
            half, -half, -half,   0, 0, 1,   1, 0,
            half, half, -half,    0, 0, 1,   1, 1,
            -half, half, -half,   0, 0, 1,   0, 1,
            -half, -half, -half,  0, 0, 1,   0, 0,
            
            # 上面 (内側向き)
            -half, half, half,    0, -1, 0,  0, 0,
            -half, half, -half,   0, -1, 0,  0, 1,
            half, half, -half,    0, -1, 0,  1, 1,
            half, half, half,     0, -1, 0,  1, 0,
            
            # 下面 (内側向き)
            -half, -half, half,   0, 1, 0,   0, 1,
            half, -half, half,    0, 1, 0,   1, 1,
            half, -half, -half,   0, 1, 0,   1, 0,
            -half, -half, -half,  0, 1, 0,   0, 0,
            
            # 右面 (内側向き)
            half, -half, half,    -1, 0, 0,  0, 0,
            half, half, half,     -1, 0, 0,  0, 1,
            half, half, -half,    -1, 0, 0,  1, 1,
            half, -half, -half,   -1, 0, 0,  1, 0,
            
            # 左面 (内側向き)
            -half, -half, -half,  1, 0, 0,   1, 0,
            -half, half, -half,   1, 0, 0,   1, 1,
            -half, half, half,    1, 0, 0,   0, 1,
            -half, -half, half,   1, 0, 0,   0, 0,
        ], dtype='f4')
        
        # インデックス
        indices = np.array([
            0, 1, 2, 2, 3, 0,       # 前面
            4, 5, 6, 6, 7, 4,       # 後面
            8, 9, 10, 10, 11, 8,    # 上面
            12, 13, 14, 14, 15, 12, # 下面
            16, 17, 18, 18, 19, 16, # 右面
            20, 21, 22, 22, 23, 20, # 左面
        ], dtype='i4')
        
        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        
        vao = self.ctx.vertex_array(self.prog, [
            (vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')
        ], ibo)
        
        return vao
    
    def _get_camera_position(self, view_matrix):
        """
        ビュー行列からカメラ位置を取得する
        
        Args:
            view_matrix (numpy.ndarray): ビュー行列
            
        Returns:
            tuple: カメラの位置 (x, y, z)
        """
        # ビュー行列の逆行列からカメラ位置を抽出
        inv_view = np.linalg.inv(view_matrix)
        return (inv_view[0, 3], inv_view[1, 3], inv_view[2, 3])
    
    def calculate_world_space_depth(self, depth_buffer, view_matrix, projection_matrix):
        """
        デプスバッファから実際の距離（ワールド空間）を計算する
        
        Args:
            depth_buffer (numpy.ndarray): デプスバッファ[0,1]
            view_matrix (numpy.ndarray): ビュー行列
            projection_matrix (numpy.ndarray): 射影行列
            
        Returns:
            numpy.ndarray: ワールド空間の深度値
        """
        # カメラ位置
        camera_pos = np.array(self._get_camera_position(view_matrix))
        
        # スクリーン空間の座標を作成
        height, width = depth_buffer.shape
        y, x = np.mgrid[0:height, 0:width]
        x = (2.0 * x / width - 1.0).astype(np.float32)
        y = (1.0 - 2.0 * y / height).astype(np.float32)
        
        # 同次座標に変換
        z = depth_buffer * 2.0 - 1.0  # [0,1]から[-1,1]へ
        ones = np.ones_like(z)
        clip_space = np.stack((x, y, z, ones), axis=-1)
        
        # 投影行列と視点行列の逆行列
        inv_proj = np.linalg.inv(projection_matrix)
        inv_view = np.linalg.inv(view_matrix)
        
        # クリップ空間からワールド空間へ
        eye_space = np.dot(clip_space, inv_proj.T)
        eye_space = eye_space / eye_space[..., 3:4]  # 同次座標の正規化
        world_space = np.dot(eye_space, inv_view.T)
        world_space = world_space / world_space[..., 3:4]  # 同次座標の正規化
        
        # カメラ位置からの距離を計算
        world_points = world_space[..., :3]
        depths = np.linalg.norm(world_points - camera_pos, axis=-1)
        
        return depths
    
    def render_scene(self, camera_matrix, projection_matrix, scene_objects, light_pos):
        """
        シーンをレンダリングする
        
        Args:
            camera_matrix (numpy.ndarray): カメラのビュー行列
            projection_matrix (numpy.ndarray): 射影行列
            scene_objects (list): レンダリングするオブジェクトのリスト
            light_pos (tuple): 光源の位置
            
        Returns:
            tuple: (RGB画像, DEPTH画像)
        """
        # RGB画像のレンダリング
        self.fbo.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        camera_pos = self._get_camera_position(camera_matrix)
        
        for obj in scene_objects:
            model_matrix = obj['model_matrix']
            vao = obj['vao']
            color = obj.get('color', (1.0, 1.0, 1.0))
            texture_id = obj.get('texture_id', None)
            
            # シェーダー変数の設定
            self.prog['model'].write(model_matrix.astype('f4').tobytes())
            self.prog['view'].write(camera_matrix.astype('f4').tobytes())
            self.prog['projection'].write(projection_matrix.astype('f4').tobytes())
            self.prog['lightPos'].value = light_pos
            self.prog['viewPos'].value = camera_pos
            self.prog['objectColor'].value = color
            
            if texture_id and texture_id in self.textures:
                self.prog['has_texture'].value = True
                self.textures[texture_id].use(0)
            else:
                self.prog['has_texture'].value = False
            
            vao.render(moderngl.TRIANGLES)
        
        # RGB画像を読み取り
        rgb_data = self.fbo.read(components=3, alignment=1)
        rgb_img = np.frombuffer(rgb_data, dtype=np.uint8).reshape(self.height, self.width, 3)
        
        # RGB画像のポストプロセッシング（GPUを使用）
        if self.use_gpu:
            try:
                rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
                rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
                with torch.no_grad():  # 勾配計算を無効化
                    processed_tensor = self.post_processor(rgb_tensor)
                rgb_img = (processed_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            except Exception as e:
                print(f"ポストプロセッシングエラー: {e}")
        
        # デプスバッファを読み取る
        depth_buffer = self.fbo.depth_attachment.read()
        depth_img = np.frombuffer(depth_buffer, dtype=np.float32).reshape(self.height, self.width)
        
        # シンプルな方法: 0-1のデプス値をTUM形式（ミリメートル単位のuint16）に変換
        # これは近似値であり、実際の距離を100mm〜10000mmの範囲にマッピングします
        near_plane = 0.1  # メートル
        far_plane = 10.0  # メートル
        
        # 非線形デプスバッファを線形化
        linearized_depth = (2.0 * near_plane) / (far_plane + near_plane - depth_img * (far_plane - near_plane))
        
        # 最大距離を制限（10メートル）
        linearized_depth = np.clip(linearized_depth, 0, 10.0)
        
        # メートルからミリメートルに変換 (1m = 1000mm)
        depth_mm = (linearized_depth * 1000.0).astype(np.uint16)
        
        return rgb_img, depth_mm
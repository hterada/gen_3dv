import os
import numpy as np
import cv2
import torch
import moderngl
import pyrr
from PIL import Image
import time
from datetime import datetime
import math

class Renderer:
    """OpenGLを使ったレンダリングを行うクラス"""
    
    def __init__(self, width=640, height=480):
        """
        レンダラーの初期化
        
        Args:
            width: 出力画像の幅
            height: 出力画像の高さ
        """
        self.width = width
        self.height = height
        
        # ModernGLのコンテキストを作成
        self.ctx = moderngl.create_standalone_context(require=330)
        print("ModernGLコンテキスト作成成功")
        
        # デプスとカラーテクスチャのためのフレームバッファの設定
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)],
            depth_attachment=self.ctx.depth_texture((width, height))
        )
        print(f"フレームバッファ作成成功: {width}x{height}")
        
        # シェーダーの設定
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
                out vec2 uv;
                out vec3 frag_pos;
                
                void main() {
                    gl_Position = projection * view * model * vec4(in_position, 1.0);
                    normal = mat3(transpose(inverse(model))) * in_normal;
                    uv = in_texcoord_0;
                    frag_pos = vec3(model * vec4(in_position, 1.0));
                }
            ''',
            fragment_shader='''
                #version 330
                
                uniform sampler2D texture0;
                uniform vec3 light_pos;
                uniform vec3 view_pos;
                
                in vec3 normal;
                in vec2 uv;
                in vec3 frag_pos;
                
                out vec4 f_color;
                
                void main() {
                    // アンビエント
                    float ambient_strength = 0.5;
                    vec3 ambient = ambient_strength * vec3(1.0, 1.0, 1.0);
                    
                    // ディフューズ
                    vec3 norm = normalize(normal);
                    vec3 light_dir = normalize(light_pos - frag_pos);
                    float diff = max(dot(norm, light_dir), 0.0);
                    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
                    
                    // スペキュラー
                    float specular_strength = 0.5;
                    vec3 view_dir = normalize(view_pos - frag_pos);
                    vec3 reflect_dir = reflect(-light_dir, norm);
                    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
                    vec3 specular = specular_strength * spec * vec3(1.0, 1.0, 1.0);
                    
                    // テクスチャと合成
                    vec4 texture_color = texture(texture0, uv);
                    
                    // デバッグ: テクスチャがロードされていない場合は赤色を表示
                    if (texture_color.a < 0.01) {
                        f_color = vec4(1.0, 0.0, 0.0, 1.0);
                        return;
                    }
                    
                    vec3 result = (ambient + diffuse + specular) * texture_color.rgb;
                    f_color = vec4(result, texture_color.a);
                    
                    // デバッグ: 計算結果が全て0になっている場合は緑色を表示
                    if (length(result) < 0.01) {
                        f_color = vec4(0.0, 1.0, 0.0, 1.0);
                    }
                }
            '''
        )
        print("シェーダープログラム作成成功")
        
        # GPUメモリの確保
        self.vbo = self.ctx.buffer(reserve=1024 * 1024)
        self.ibo = self.ctx.buffer(reserve=1024 * 1024)
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')])
        
        # デプスバッファの有効化
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        
    def load_texture(self, image_path):
        """
        テクスチャを読み込む
        
        Args:
            image_path: テクスチャ画像のパス
            
        Returns:
            読み込まれたテクスチャオブジェクト
        """
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGBA')
                print(f"テクスチャ読み込み成功: {image_path}, サイズ: {img.size}")
                texture = self.ctx.texture(img.size, 4, img.tobytes())
                texture.build_mipmaps()
                texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                
                # バインドしてテストする
                texture.use(0)
                return texture
        except Exception as e:
            print(f"テクスチャ読み込みエラー: {e}")
            # エラーが発生した場合は、デフォルトのテクスチャを生成
            fallback_img = Image.new('RGBA', (64, 64), color=(255, 0, 255, 255))
            # チェッカーボードパターンを追加（PILのDrawモジュールを使わずに直接ピクセルを設定）
            for y in range(8):
                for x in range(8):
                    if (x + y) % 2 == 0:
                        for py in range(y*8, (y+1)*8):
                            for px in range(x*8, (x+1)*8):
                                fallback_img.putpixel((px, py), (0, 0, 0, 255))
            
            texture = self.ctx.texture((64, 64), 4, fallback_img.tobytes())
            texture.use(0)
            return texture
    
    def create_cube(self, size=1.0):
        """
        立方体の頂点とインデックスを生成
        
        Args:
            size: 立方体の大きさ
            
        Returns:
            (vertices, indices)のタプル
        """
        s = size / 2
        
        vertices = np.array([
            # 前面 (z+)
            -s, -s,  s,  0,  0,  1,  0, 0,  # 左下
             s, -s,  s,  0,  0,  1,  1, 0,  # 右下
             s,  s,  s,  0,  0,  1,  1, 1,  # 右上
            -s,  s,  s,  0,  0,  1,  0, 1,  # 左上
            
            # 背面 (z-)
             s, -s, -s,  0,  0, -1,  0, 0,  # 左下
            -s, -s, -s,  0,  0, -1,  1, 0,  # 右下
            -s,  s, -s,  0,  0, -1,  1, 1,  # 右上
             s,  s, -s,  0,  0, -1,  0, 1,  # 左上
            
            # 右面 (x+)
             s, -s,  s,  1,  0,  0,  0, 0,  # 左下
             s, -s, -s,  1,  0,  0,  1, 0,  # 右下
             s,  s, -s,  1,  0,  0,  1, 1,  # 右上
             s,  s,  s,  1,  0,  0,  0, 1,  # 左上
            
            # 左面 (x-)
            -s, -s, -s, -1,  0,  0,  0, 0,  # 左下
            -s, -s,  s, -1,  0,  0,  1, 0,  # 右下
            -s,  s,  s, -1,  0,  0,  1, 1,  # 右上
            -s,  s, -s, -1,  0,  0,  0, 1,  # 左上
            
            # 上面 (y+)
            -s,  s,  s,  0,  1,  0,  0, 0,  # 左下
             s,  s,  s,  0,  1,  0,  1, 0,  # 右下
             s,  s, -s,  0,  1,  0,  1, 1,  # 右上
            -s,  s, -s,  0,  1,  0,  0, 1,  # 左上
            
            # 下面 (y-)
            -s, -s, -s,  0, -1,  0,  0, 0,  # 左下
             s, -s, -s,  0, -1,  0,  1, 0,  # 右下
             s, -s,  s,  0, -1,  0,  1, 1,  # 右上
            -s, -s,  s,  0, -1,  0,  0, 1,  # 左上
        ], dtype='f4')
        
        indices = np.array([
            0, 1, 2, 2, 3, 0,       # 前面
            4, 5, 6, 6, 7, 4,       # 背面
            8, 9, 10, 10, 11, 8,    # 右面
            12, 13, 14, 14, 15, 12, # 左面
            16, 17, 18, 18, 19, 16, # 上面
            20, 21, 22, 22, 23, 20, # 下面
        ], dtype='i4')
        
        return vertices, indices
    
    def create_room(self, size=10.0):
        """
        部屋（直方体）の頂点とインデックスを生成
        
        Args:
            size: 部屋の大きさ
            
        Returns:
            (vertices, indices)のタプル
        """
        s = size / 2
        
        vertices = np.array([
            # 前面 (z+) (内向き)
            -s, -s,  s,  0,  0, -1,  0, 0,
             s, -s,  s,  0,  0, -1,  1, 0,
             s,  s,  s,  0,  0, -1,  1, 1,
            -s,  s,  s,  0,  0, -1,  0, 1,
            
            # 背面 (z-) (内向き)
             s, -s, -s,  0,  0,  1,  0, 0,
            -s, -s, -s,  0,  0,  1,  1, 0,
            -s,  s, -s,  0,  0,  1,  1, 1,
             s,  s, -s,  0,  0,  1,  0, 1,
            
            # 右面 (x+) (内向き)
             s, -s,  s, -1,  0,  0,  0, 0,
             s, -s, -s, -1,  0,  0,  1, 0,
             s,  s, -s, -1,  0,  0,  1, 1,
             s,  s,  s, -1,  0,  0,  0, 1,
            
            # 左面 (x-) (内向き)
            -s, -s, -s,  1,  0,  0,  0, 0,
            -s, -s,  s,  1,  0,  0,  1, 0,
            -s,  s,  s,  1,  0,  0,  1, 1,
            -s,  s, -s,  1,  0,  0,  0, 1,
            
            # 上面 (y+) (内向き)
            -s,  s,  s,  0, -1,  0,  0, 0,
             s,  s,  s,  0, -1,  0,  1, 0,
             s,  s, -s,  0, -1,  0,  1, 1,
            -s,  s, -s,  0, -1,  0,  0, 1,
            
            # 下面 (y-) (内向き)
            -s, -s, -s,  0,  1,  0,  0, 0,
             s, -s, -s,  0,  1,  0,  1, 0,
             s, -s,  s,  0,  1,  0,  1, 1,
            -s, -s,  s,  0,  1,  0,  0, 1,
        ], dtype='f4')
        
        indices = np.array([
            0, 2, 1, 0, 3, 2,       # 前面
            4, 6, 5, 4, 7, 6,       # 背面
            8, 10, 9, 8, 11, 10,    # 右面
            12, 14, 13, 12, 15, 14, # 左面
            16, 18, 17, 16, 19, 18, # 上面
            20, 22, 21, 20, 23, 22, # 下面
        ], dtype='i4')
        
        return vertices, indices
    
    def render_scene(self, camera, scene):
        """
        シーンをレンダリングする
        
        Args:
            camera: カメラオブジェクト
            scene: シーンオブジェクト
            
        Returns:
            (rgb_image, depth_image) のタプル
        """
        # フレームバッファをクリア
        self.ctx.clear(0.2, 0.2, 0.2, 1.0)  # 暗い灰色で背景をクリア
        self.fbo.use()
        
        # カメラのビュー行列と射影行列を設定
        view = pyrr.matrix44.create_look_at(
            eye=camera.position,
            target=camera.target,
            up=camera.up
        )
        projection = pyrr.matrix44.create_perspective_projection(
            fovy=45.0, aspect=self.width/self.height, near=0.1, far=100.0
        )
        
        # シェーダーに各種パラメータをセット
        self.prog['view'].write(view.astype('f4').tobytes())
        self.prog['projection'].write(projection.astype('f4').tobytes())
        self.prog['light_pos'].value = (0.0, 10.0, 0.0)  # 光源位置を調整
        self.prog['view_pos'].value = tuple(camera.position)
        
        # デバッグ情報を出力
        print(f"カメラ位置: {camera.position}")
        print(f"視線方向: {camera.target - camera.position}")
        print(f"オブジェクト数: {len(scene.objects)}")
        
        # シーン内のオブジェクトをレンダリング
        for i, obj in enumerate(scene.objects):
            # モデル行列をセット
            model = pyrr.matrix44.create_from_translation(obj['position'])
            self.prog['model'].write(model.astype('f4').tobytes())
            
            # テクスチャをバインド
            obj['texture'].use(0)
            
            # 頂点データとインデックスデータをVBOとIBOにアップロード
            self.vbo.write(obj['vertices'].tobytes())
            self.ibo.write(obj['indices'].tobytes())
            
            # 描画
            self.vao.render(moderngl.TRIANGLES, vertices=len(obj['indices']))
            
            if i < 3:  # 最初の数オブジェクトのみ情報を出力
                print(f"オブジェクト{i} 頂点数: {len(obj['vertices']) // 8}, 三角形数: {len(obj['indices']) // 3}")
        
        # レンダリング結果を読み取り
        rgb_data = self.fbo.read(components=3, attachment=0)
        depth_data = self.fbo.read(attachment=-1)
        
        # NumPy配列に変換
        rgb_image = np.frombuffer(rgb_data, dtype=np.uint8).reshape(self.height, self.width, 3)
        
        # 深度データのサイズを確認
        depth_float_count = len(depth_data) // 4  # float32は4バイト
        print(f"深度データサイズ: {len(depth_data)} バイト, {depth_float_count} 要素")
        
        # 深度データが正方形の解像度と仮定して処理
        depth_size = int(np.sqrt(depth_float_count))
        print(f"深度推定解像度: {depth_size}x{depth_size}")
        
        try:
            # 正方形として読み込み
            depth_image = np.frombuffer(depth_data, dtype=np.float32).reshape(depth_size, depth_size)
            # 要求サイズにリサイズ
            depth_image = cv2.resize(depth_image, (self.width, self.height))
        except ValueError:
            print(f"深度バッファの形状を自動推定できませんでした。代替方法を使用します。")
            # 代替: ダミーの深度マップを生成
            depth_image = np.ones((self.height, self.width), dtype=np.float32)
            # カメラからの距離に基づく簡易深度マップ
            for y in range(self.height):
                for x in range(self.width):
                    # 画面中心からの距離
                    cx, cy = self.width // 2, self.height // 2
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2) / max(self.width, self.height) * 2
                    # カメラと中心オブジェクトの距離を基準に変化
                    base_depth = np.linalg.norm(camera.position - camera.target)
                    depth_image[y, x] = base_depth * (1.0 + 0.2 * dist)
        
        # RGBの順番を入れ替え（OpenGLはBGR）
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # 画像の上下を反転（OpenGLは上下が逆）
        rgb_image = cv2.flip(rgb_image, 0)
        depth_image = cv2.flip(depth_image, 0)
        
        # デプス値を正規化（0～1から実際の距離に変換）
        z_near, z_far = 0.1, 100.0
        depth_image = 2.0 * z_near * z_far / (z_far + z_near - (2.0 * depth_image - 1.0) * (z_far - z_near))
        
        # デバッグ: RGBイメージが全て真っ黒かチェック
        if np.mean(rgb_image) < 5:
            print("警告: レンダリング結果が真っ黒です。照明やテクスチャを確認してください。")
            # デバッグ用にカラーバーを追加
            h, w = rgb_image.shape[:2]
            for x in range(w):
                color = [int(255 * x / w), int(255 * (1 - x / w)), 128]
                rgb_image[:10, x] = color
        
        return rgb_image, depth_image


class CameraTrajectory:
    """カメラの軌道を計算するクラス"""
    
    def __init__(self, center=(0, 0, 0), radius=5.0, height=2.0, total_frames=360):
        """
        カメラ軌道の初期化
        
        Args:
            center: 軌道の中心座標
            radius: 軌道の半径
            height: カメラの高さ
            total_frames: 一周するのに必要なフレーム数
        """
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
        self.height = height
        self.total_frames = total_frames
        self.current_frame = 0
        
    def get_next_position(self):
        """
        次のカメラ位置を計算
        
        Returns:
            (position, target, up, timestamp) のタプル
        """
        # 角度を計算（0～360度）
        angle = 2 * np.pi * self.current_frame / self.total_frames
        
        # 位置を計算（円軌道）
        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.height
        z = self.center[2] + self.radius * np.sin(angle)
        position = np.array([x, y, z], dtype=np.float32)
        
        # 常に中心を見る
        target = self.center.copy()
        
        # 上方向
        up = np.array([0, 1, 0], dtype=np.float32)
        
        # タイムスタンプ（現在時刻をナノ秒単位で）
        timestamp = time.time()
        
        # フレームカウンタを更新
        self.current_frame = (self.current_frame + 1) % self.total_frames
        
        return position, target, up, timestamp
    
    def calculate_motion(self, prev_position, prev_target, position, target):
        """
        カメラの動きを計算
        
        Args:
            prev_position: 前フレームのカメラ位置
            prev_target: 前フレームのカメラターゲット
            position: 現在のカメラ位置
            target: 現在のカメラターゲット
            
        Returns:
            (translation, rotation_quaternion) のタプル
        """
        # 並進ベクトル
        translation = position - prev_position
        
        # 回転行列を計算
        # 前フレームのカメラ座標系を計算
        forward_prev = prev_target - prev_position
        forward_prev = forward_prev / np.linalg.norm(forward_prev)
        
        right_prev = np.cross(forward_prev, np.array([0, 1, 0]))
        right_prev = right_prev / np.linalg.norm(right_prev)
        
        up_prev = np.cross(right_prev, forward_prev)
        
        # 現在フレームのカメラ座標系を計算
        forward_curr = target - position
        forward_curr = forward_curr / np.linalg.norm(forward_curr)
        
        right_curr = np.cross(forward_curr, np.array([0, 1, 0]))
        right_curr = right_curr / np.linalg.norm(right_curr)
        
        up_curr = np.cross(right_curr, forward_curr)
        
        # 前フレームから現在フレームへの回転行列
        rot_mat = np.array([
            [np.dot(right_curr, right_prev), np.dot(right_curr, up_prev), np.dot(right_curr, -forward_prev)],
            [np.dot(up_curr, right_prev), np.dot(up_curr, up_prev), np.dot(up_curr, -forward_prev)],
            [np.dot(-forward_curr, right_prev), np.dot(-forward_curr, up_prev), np.dot(-forward_curr, -forward_prev)]
        ])
        
        # 回転行列からクォータニオンに変換
        trace = rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (rot_mat[2, 1] - rot_mat[1, 2]) * s
            qy = (rot_mat[0, 2] - rot_mat[2, 0]) * s
            qz = (rot_mat[1, 0] - rot_mat[0, 1]) * s
        else:
            if rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
                qw = (rot_mat[2, 1] - rot_mat[1, 2]) / s
                qx = 0.25 * s
                qy = (rot_mat[0, 1] + rot_mat[1, 0]) / s
                qz = (rot_mat[0, 2] + rot_mat[2, 0]) / s
            elif rot_mat[1, 1] > rot_mat[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
                qw = (rot_mat[0, 2] - rot_mat[2, 0]) / s
                qx = (rot_mat[0, 1] + rot_mat[1, 0]) / s
                qy = 0.25 * s
                qz = (rot_mat[1, 2] + rot_mat[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
                qw = (rot_mat[1, 0] - rot_mat[0, 1]) / s
                qx = (rot_mat[0, 2] + rot_mat[2, 0]) / s
                qy = (rot_mat[1, 2] + rot_mat[2, 1]) / s
                qz = 0.25 * s
        
        rotation_quaternion = np.array([qw, qx, qy, qz])
        
        return translation, rotation_quaternion


class Scene:
    """シーンを構築するクラス"""
    
    def __init__(self, renderer):
        """
        シーンの初期化
        
        Args:
            renderer: レンダラーオブジェクト
        """
        self.renderer = renderer
        self.objects = []
        
    def create_scene(self, textures_dir):
        """
        シーンを構築する
        
        Args:
            textures_dir: テクスチャファイルのディレクトリ
        """
        print(f"シーン構築開始: テクスチャディレクトリ = {textures_dir}")
        
        # テクスチャの読み込み
        try:
            wall_texture = self.renderer.load_texture(os.path.join(textures_dir, 'wall.jpg'))
            floor_texture = self.renderer.load_texture(os.path.join(textures_dir, 'floor.jpg'))
            ceiling_texture = self.renderer.load_texture(os.path.join(textures_dir, 'ceiling.jpg'))
            
            cube_textures = [
                self.renderer.load_texture(os.path.join(textures_dir, 'cube_front.jpg')),
                self.renderer.load_texture(os.path.join(textures_dir, 'cube_back.jpg')),
                self.renderer.load_texture(os.path.join(textures_dir, 'cube_right.jpg')),
                self.renderer.load_texture(os.path.join(textures_dir, 'cube_left.jpg')),
                self.renderer.load_texture(os.path.join(textures_dir, 'cube_top.jpg')),
                self.renderer.load_texture(os.path.join(textures_dir, 'cube_bottom.jpg')),
            ]
            
            print("テクスチャのロード完了")
        except Exception as e:
            print(f"テクスチャロードエラー: {e}")
            return
        
        # 部屋を作成
        room_vertices, room_indices = self.renderer.create_room(10.0)
        print(f"部屋の頂点数: {len(room_vertices) // 8}, インデックス数: {len(room_indices)}")
        
        # 立方体を作成
        cube_vertices, cube_indices = self.renderer.create_cube(2.0)
        print(f"立方体の頂点数: {len(cube_vertices) // 8}, インデックス数: {len(cube_indices)}")
        
        # 部屋の各面の頂点インデックスを計算
        vertices_per_face = 4
        indices_per_face = 6
        
        # 部屋の各面をオブジェクトとして追加
        for i in range(6):
            face_vertices = room_vertices[i * vertices_per_face * 8:(i + 1) * vertices_per_face * 8]
            face_indices = room_indices[i * indices_per_face:(i + 1) * indices_per_face]
            
            # インデックスの調整（オフセットを計算）
            adjusted_indices = []
            for idx in face_indices:
                adjusted_indices.append(idx - i * vertices_per_face)
            
            # テクスチャの選択
            if i < 4:  # 壁面
                texture = wall_texture
            elif i == 4:  # 天井
                texture = ceiling_texture
            else:  # 床
                texture = floor_texture
            
            self.objects.append({
                'vertices': np.array(face_vertices, dtype='f4'),
                'indices': np.array(adjusted_indices, dtype='i4'),
                'texture': texture,
                'position': np.array([0, 0, 0], dtype=np.float32)
            })
        
        # 立方体の各面をオブジェクトとして追加
        for i in range(6):
            face_vertices = cube_vertices[i * vertices_per_face * 8:(i + 1) * vertices_per_face * 8]
            face_indices = cube_indices[i * indices_per_face:(i + 1) * indices_per_face]
            
            # インデックスの調整
            adjusted_indices = []
            for idx in face_indices:
                adjusted_indices.append(idx - i * vertices_per_face)
            
            self.objects.append({
                'vertices': np.array(face_vertices, dtype='f4'),
                'indices': np.array(adjusted_indices, dtype='i4'),
                'texture': cube_textures[i],
                'position': np.array([0, 0, 0], dtype=np.float32)
            })
        
        print(f"シーン構築完了: オブジェクト数 = {len(self.objects)}")
        
        # デバッグ：最初の数オブジェクトのデータを表示
        for i in range(min(2, len(self.objects))):
            obj = self.objects[i]
            print(f"オブジェクト{i}: 頂点数={len(obj['vertices']) // 8}, インデックス数={len(obj['indices'])}")
            # 最初の数頂点を表示
            print(f"  最初の頂点: {obj['vertices'][:24]}")
            print(f"  最初のインデックス: {obj['indices'][:6]}")
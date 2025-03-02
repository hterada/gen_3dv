import math
import random
import os
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import time  # タイムスタンプ用


# テクスチャの準備 (前処理)
def generate_checkerboard_texture(width, height):
    texture_data = []
    for y in range(height):
        for x in range(width):
            if (x // 32 + y // 32) % 2 == 0:
                texture_data.append((255, 255, 255, 255))
            else:
                texture_data.append((0, 0, 0, 255))
    return texture_data


def generate_tum_format_data(output_dir, num_frames, width, height,
                             cube_size, cube_position, cube_orientation,
                             camera_trajectory, textures):
    """
    TUM RGB-Dデータセット形式のデータを出力するCG描画プログラム。

    Args:
        output_dir: 出力ディレクトリ。
        num_frames: フレーム数。
        width: 画像の幅。
        height: 画像の高さ。
        cube_size: 直方体のサイズ。
        cube_position: 直方体の初期位置。
        cube_orientation: 直方体の初期姿勢。
        camera_trajectory: カメラの軌跡。
        textures: テクスチャのリスト (ファイルパスまたは生成関数)。
    """

    # ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)


    def load_or_generate_texture(texture_spec, width, height):
        if isinstance(texture_spec, str):
            try:
                img = Image.open(texture_spec).resize((width, height)).convert("RGBA")
                return list(img.getdata())
            except FileNotFoundError:
                print(f"警告: {texture_spec} が見つかりません。チェッカーボードを使用。")
                return generate_checkerboard_texture(width, height)
        elif callable(texture_spec):
            return texture_spec(width, height)
        else:
            raise ValueError("texture_spec はファイルパスまたは生成関数")

    texture_data = [load_or_generate_texture(tex, 128, 128) for tex in textures]

    # 頂点、面などの計算 (前処理、以前のコードから)
    def get_cube_vertices(size, position, orientation):
        half_x, half_y, half_z = size[0] / 2, size[1] / 2, size[2] / 2
        vertices = [
            [-half_x, -half_y, -half_z], [ half_x, -half_y, -half_z],
            [ half_x,  half_y, -half_z], [-half_x,  half_y, -half_z],
            [-half_x, -half_y,  half_z], [ half_x, -half_y,  half_z],
            [ half_x,  half_y,  half_z], [-half_x,  half_y,  half_z],
        ]
        rotation = Quaternion(axis=[1, 0, 0], angle=orientation[0]) * \
                   Quaternion(axis=[0, 1, 0], angle=orientation[1]) * \
                   Quaternion(axis=[0, 0, 1], angle=orientation[2])
        rotated_vertices = [rotation.rotate(v) for v in vertices]
        translated_vertices = [[v[0] + position[0], v[1] + position[1], v[2] + position[2]] for v in rotated_vertices]
        return np.array(translated_vertices)

    def get_cube_faces():
        return [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
        ]

    def project_point(point, camera_position, camera_orientation, fov_x, aspect_ratio):
        camera_rotation = Quaternion(axis=[1, 0, 0], angle=camera_orientation[0]) * \
                          Quaternion(axis=[0, 1, 0], angle=camera_orientation[1]) * \
                          Quaternion(axis=[0, 0, 1], angle=camera_orientation[2])
        camera_rotation = camera_rotation.inverse
        point_camera_space = camera_rotation.rotate(point - np.array(camera_position))

        if point_camera_space[2] <= 0:
            return None, None  # 2D座標と深度の両方を返す

        focal_length_x = 1 / math.tan(fov_x / 2)
        focal_length_y = focal_length_x * aspect_ratio

        screen_x = focal_length_x * (point_camera_space[0] / point_camera_space[2])
        screen_y = focal_length_y * (point_camera_space[1] / point_camera_space[2])
        pixel_x = int(width / 2 + screen_x * width / 2)
        pixel_y = int(height / 2 - screen_y * height / 2)

        depth = point_camera_space[2]  # カメラ座標系でのZ座標が深度
        return (pixel_x, pixel_y), depth



    def draw_face(image_data, depth_data, vertices, face, texture, camera_position, camera_orientation, fov_x, aspect_ratio):
      projected_vertices = [project_point(v, camera_position, camera_orientation, fov_x, aspect_ratio) for v in vertices[face]]

      # 投影できない頂点がある場合、この面は描画しない
      if all(v[0] is None for v in projected_vertices):
          return
      #少なくとも３つの有効な点が必要
      visible_vertices = [(v,d) for (v,d) in projected_vertices if v is not None]
      if len(visible_vertices) < 3:
        return

      # バウンディングボックスでクリッピング
      min_x = max(0, min(v[0][0] for v in visible_vertices))
      min_y = max(0, min(v[0][1] for v in visible_vertices))
      max_x = min(width - 1, max(v[0][0] for v in visible_vertices))
      max_y = min(height - 1, max(v[0][1] for v in visible_vertices))

      for y in range(min_y, max_y + 1):
          for x in range(min_x, max_x + 1):

              if len(visible_vertices) == 4:
                    p0, p1, p2, p3 = [v[0] for v in visible_vertices] #座標だけ取り出す
                    v0, v1, v2 = np.array(p1) - np.array(p0), np.array(p3) - np.array(p0), np.array([x,y]) - np.array(p0)
                    d00, d01, d11, d20, d21 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1), np.dot(v2, v0), np.dot(v2, v1)
                    denom = d00 * d11 - d01 * d01
                    if denom == 0: continue
                    v = (d11 * d20 - d01 * d21) / denom
                    w = (d00 * d21 - d01 * d20) / denom
                    u = 1.0 - v - w

                    if not (0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1):
                        continue

                    tex_x, tex_y = int(u * 127 + v * 127), int(w * 127)
                    depth = u * visible_vertices[0][1] + v * visible_vertices[1][1] + w*visible_vertices[3][1] # 深度も補間

              elif len(visible_vertices) == 3:
                    p0, p1, p2 = [v[0] for v in visible_vertices] #座標だけ
                    v0, v1, v2 = np.array(p1) - np.array(p0), np.array(p2) - np.array(p0), np.array([x, y]) - np.array(p0)
                    d00, d01, d11, d20, d21 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1), np.dot(v2, v0), np.dot(v2, v1)
                    denom = d00 * d11 - d01 * d01
                    if denom == 0: continue

                    v = (d11 * d20 - d01 * d21) / denom
                    w = (d00 * d21 - d01 * d20) / denom
                    u = 1.0 - v - w

                    if not(0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1):
                        continue

                    tex_x, tex_y = int(u * 127 + v * 127), int(w * 127)
                    depth = u * visible_vertices[0][1] + v * visible_vertices[1][1] + w * visible_vertices[2][1]  # 深度も補間

              else:
                continue
              tex_x, tex_y = max(0, min(tex_x, 127)), max(0, min(tex_y, 127))
              pixel_index = y * width + x

              # Z-buffering (より近いピクセルだけ描画)
              if depth < depth_data[pixel_index]:
                image_data[pixel_index] = texture[tex_y * 128 + tex_x]
                depth_data[pixel_index] = depth




    # --- データ生成ループ ---
    fov_x = math.radians(60)
    aspect_ratio = height / width
    depth_scale = 5000.0  # TUMデータセットの深度スケール

    rgb_file = open(os.path.join(output_dir, "rgb.txt"), "w")
    depth_file = open(os.path.join(output_dir, "depth.txt"), "w")
    groundtruth_file = open(os.path.join(output_dir, "groundtruth.txt"), "w")


    start_time = time.time()  # 最初のフレームの時刻 (相対時間)

    for frame_num in range(num_frames):
        print(f"フレーム {frame_num + 1}/{num_frames} をレンダリング中...")

        timestamp = start_time + (frame_num / 30.0)  # 30FPSを仮定
        timestamp_str = f"{timestamp:.6f}"

        camera_position, camera_orientation = camera_trajectory[frame_num]

        image_data = [(255, 255, 255, 255)] * (width * height) #RGBA
        depth_data = [float('inf')] * (width * height)  # 初期値を無限遠に

        cube_vertices = get_cube_vertices(cube_size, cube_position, cube_orientation)
        cube_faces = get_cube_faces()

        for i, face in enumerate(cube_faces):
            draw_face(image_data, depth_data, cube_vertices, face, texture_data[i],
                      camera_position, camera_orientation, fov_x, aspect_ratio)


        # RGB画像の保存
        image_array = np.array(image_data, dtype=np.uint8).reshape((height, width, 4))
        img = Image.fromarray(image_array)
        rgb_filename = f"{timestamp_str}.png"
        img.save(os.path.join(rgb_dir, rgb_filename))
        rgb_file.write(f"{timestamp_str} rgb/{rgb_filename}\n")

        # 深度画像の保存 (16-bit PNG, スケール調整)
        depth_array = np.array(depth_data, dtype=np.float32).reshape((height, width))
        depth_array = (depth_array * depth_scale).astype(np.uint16)  # スケール変換と型変換
        depth_img = Image.fromarray(depth_array, mode="I;16") # I;16モード (16-bit unsigned integer)
        depth_filename = f"{timestamp_str}.png"
        depth_img.save(os.path.join(depth_dir, depth_filename))
        depth_file.write(f"{timestamp_str} depth/{depth_filename}\n")

        # Ground truth (カメラポーズ) の保存
        tx, ty, tz = camera_position
        qx, qy, qz, qw = Quaternion(axis=[1, 0, 0], angle=camera_orientation[0]) * \
                         Quaternion(axis=[0, 1, 0], angle=camera_orientation[1]) * \
                         Quaternion(axis=[0, 0, 1], angle=camera_orientation[2])
        groundtruth_file.write(f"{timestamp_str} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    rgb_file.close()
    depth_file.close()
    groundtruth_file.close()

    print(f"TUM形式のデータを {output_dir} に保存しました。")


def generate_random_texture(width, height):
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255) for _ in range(width * height)]

# --- 使用例 ---
if __name__ == "__main__":
    num_frames = 100
    width, height = 640, 480
    cube_size = (1, 1, 1)
    cube_position = (0, 0, 5)
    cube_orientation = (math.radians(30), math.radians(45), 0)

    camera_trajectory = []
    for i in range(num_frames):
        angle = 2 * math.pi * i / num_frames
        camera_x = 10 * math.cos(angle)
        camera_z = 10 * math.sin(angle) + 5
        camera_y = 1
        camera_roll = 0
        camera_pitch = 0
        camera_yaw = -angle + math.pi / 2
        camera_trajectory.append(((camera_x, camera_y, camera_z), (camera_roll, camera_pitch, camera_yaw)))


    textures = [
        "texture1.png", "texture2.jpg",
        generate_checkerboard_texture,
        generate_random_texture,
        "texture3.bmp",
        lambda w, h: [(int(x/w * 255), int(y/h * 255) ,0 , 255) for y in range(h) for x in range(w)]
    ]
    output_directory = "tum_output"  # 出力ディレクトリ
    generate_tum_format_data(output_directory, num_frames, width, height,
                             cube_size, cube_position, cube_orientation,
                             camera_trajectory, textures)
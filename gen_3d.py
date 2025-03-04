import math
import random
import os
import numpy as np
from PIL import Image
from pyquaternion import Quaternion  # リー代数を使う前に、まずは現状のコードを修正
import time


def generate_checkerboard_texture(width, height):
    """チェッカーボードテクスチャを生成"""
    texture_data = []
    for y in range(height):
        for x in range(width):
            if (x // 32 + y // 32) % 2 == 0:
                texture_data.append((255, 255, 255, 255))
            else:
                texture_data.append((0, 0, 0, 255))
    return texture_data


def generate_random_texture(width, height):
    return [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
        for _ in range(width * height)
    ]


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
    
def get_cube_vertices(size, position, orientation):
    half_x, half_y, half_z = size[0] / 2, size[1] / 2, size[2] / 2
    vertices = [
        [-half_x, -half_y, -half_z],
        [half_x, -half_y, -half_z],
        [half_x, half_y, -half_z],
        [-half_x, half_y, -half_z],
        [-half_x, -half_y, half_z],
        [half_x, -half_y, half_z],
        [half_x, half_y, half_z],
        [-half_x, half_y, half_z],
    ]
    rotation = (
        Quaternion(axis=[1, 0, 0], angle=orientation[0])
        * Quaternion(axis=[0, 1, 0], angle=orientation[1])
        * Quaternion(axis=[0, 0, 1], angle=orientation[2])
    )
    rotated_vertices = [rotation.rotate(v) for v in vertices]
    return np.array(
        [
            [v[0] + position[0], v[1] + position[1], v[2] + position[2]]
            for v in rotated_vertices
        ]
    )

def get_cube_faces():
    return [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]

def project_point(
    point, camera_position, camera_rotation_matrix, fov_x, aspect_ratio
):
    """3D点を2Dスクリーン座標に投影 (修正版)"""

    # カメラ座標系への変換
    point_camera_space = np.dot(
        camera_rotation_matrix, point - np.array(camera_position)
    )
    print(f"  point_camera_space (z): {point_camera_space[2]}") # デバッグ出力

    if point_camera_space[2] <= 0:
        return None, None

    focal_length_x = 1 / math.tan(fov_x / 2)
    focal_length_y = focal_length_x * aspect_ratio

    screen_x = focal_length_x * (point_camera_space[0] / point_camera_space[2])
    screen_y = focal_length_y * (point_camera_space[1] / point_camera_space[2])

    pixel_x = int(width / 2 + screen_x * width / 2)
    pixel_y = int(height / 2 - screen_y * height / 2)

    depth = point_camera_space[2]
    return (pixel_x, pixel_y), depth

def draw_face(
    image_data,
    depth_data,
    vertices,
    face,
    texture,
    camera_position,
    camera_rotation_matrix,
    fov_x,
    aspect_ratio,
    tex_width,
    tex_height,
):
    projected_vertices = [
        project_point(
            v, camera_position, camera_rotation_matrix, fov_x, aspect_ratio
        )
        for v in vertices[face]
    ]

    if all(v[0] is None for v in projected_vertices):
        return
    visible_vertices = [(v, d) for (v, d) in projected_vertices if v is not None]
    if len(visible_vertices) < 3:
        return

    min_x = max(0, min(v[0][0] for v in visible_vertices))
    min_y = max(0, min(v[0][1] for v in visible_vertices))
    max_x = min(width - 1, max(v[0][0] for v in visible_vertices))
    max_y = min(height - 1, max(v[0][1] for v in visible_vertices))

    depths_in_face = []  # この面における深度値を格納

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if len(visible_vertices) == 4:
                p0, p1, p2, p3 = [v[0] for v in visible_vertices]
                v0, v1, v2 = (
                    np.array(p1) - np.array(p0),
                    np.array(p3) - np.array(p0),
                    np.array([x, y]) - np.array(p0),
                )
                d00, d01, d11, d20, d21 = (
                    np.dot(v0, v0),
                    np.dot(v0, v1),
                    np.dot(v1, v1),
                    np.dot(v2, v0),
                    np.dot(v2, v1),
                )
                denom = d00 * d11 - d01 * d01
                if denom == 0:
                    continue
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                u = 1.0 - v - w
                if not (0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1):
                    continue
                tex_x, tex_y = int(u * (tex_width - 1)), int(
                    (v + w) * (tex_height - 1)
                )
                depth = (
                    u * visible_vertices[0][1]
                    + v * visible_vertices[1][1]
                    + w * visible_vertices[3][1]
                )

            elif len(visible_vertices) == 3:
                p0, p1, p2 = [v[0] for v in visible_vertices]
                v0, v1, v2 = (
                    np.array(p1) - np.array(p0),
                    np.array(p2) - np.array(p0),
                    np.array([x, y]) - np.array(p0),
                )
                d00, d01, d11, d20, d21 = (
                    np.dot(v0, v0),
                    np.dot(v0, v1),
                    np.dot(v1, v1),
                    np.dot(v2, v0),
                    np.dot(v2, v1),
                )
                denom = d00 * d11 - d01 * d01
                if denom == 0:
                    continue
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                u = 1.0 - v - w
                if not (0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1):
                    continue
                tex_x, tex_y = int(u * (tex_width - 1)), int(
                    (v + w) * (tex_height - 1)
                )
                depth = (
                    u * visible_vertices[0][1]
                    + v * visible_vertices[1][1]
                    + w * visible_vertices[2][1]
                )
            else:
                continue

            tex_x, tex_y = max(0, min(tex_x, tex_width - 1)), max(
                0, min(tex_y, tex_height - 1)
            )
            pixel_index = y * width + x
            if depth < depth_data[pixel_index]:
                image_data[pixel_index] = texture[tex_y * tex_width + tex_x]
                depth_data[pixel_index] = depth
                depths_in_face.append(depth)  # 深度値をリストに追加

    # デバッグ出力 (面の描画後)
    if depths_in_face:  # depthsが空でない場合のみ
        print(
            f"    depths (min/max/avg): {min(depths_in_face):.3f} / {max(depths_in_face):.3f} / {sum(depths_in_face) / len(depths_in_face):.3f}"
        )


def generate_tum_format_data(
    output_dir,
    num_frames,
    width,
    height,
    cube_size,
    cube_position,
    cube_orientation,
    textures,
):
    """TUM形式のデータを出力"""

    os.makedirs(output_dir, exist_ok=True)
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    texture_data = [load_or_generate_texture(tex, 128, 128) for tex in textures]


    fov_x = math.radians(60)
    aspect_ratio = height / width
    depth_scale = 5000.0
    # depth_scale = 1000.0  # デバッグ用にスケールを小さくしてみる
    rgb_file = open(os.path.join(output_dir, "rgb.txt"), "w")
    depth_file = open(os.path.join(output_dir, "depth.txt"), "w")
    groundtruth_file = open(os.path.join(output_dir, "groundtruth.txt"), "w")
    start_time = time.time()

    camera_distance = 10.0  # 5.0  # カメラと直方体の距離
    camera_elevation = math.radians(20)

    for frame_num in range(num_frames):
        print(f"フレーム {frame_num + 1}/{num_frames} をレンダリング中...")
        timestamp = start_time + (frame_num / 30.0)
        timestamp_str = f"{timestamp:.6f}"

        angle = 2 * math.pi * frame_num / num_frames
        camera_x = cube_position[0] + camera_distance * math.cos(
            camera_elevation
        ) * math.sin(angle)
        camera_y = cube_position[1] + camera_distance * math.sin(camera_elevation)
        camera_z = cube_position[2] + camera_distance * math.cos(
            camera_elevation
        ) * math.cos(angle)
        camera_position = (camera_x, camera_y, camera_z)

        look_at_vector = np.array(cube_position) - np.array(camera_position)
        look_at_vector = look_at_vector / np.linalg.norm(look_at_vector)

        up_vector = np.array([0, 1, 0])
        right_vector = np.cross(look_at_vector, up_vector)
        right_vector = right_vector / np.linalg.norm(right_vector)
        up_vector = np.cross(right_vector, look_at_vector)
        camera_rotation_matrix = np.linalg.inv(
            np.array([right_vector, up_vector, -look_at_vector]).T
        )

        image_data = [(255, 255, 255, 255)] * (width * height)
        depth_data = [float("inf")] * (width * height)  # 初期化
        cube_vertices = get_cube_vertices(cube_size, cube_position, cube_orientation)
        cube_faces = get_cube_faces()
        for i, face in enumerate(cube_faces):
            draw_face(
                image_data,
                depth_data,
                cube_vertices,
                face,
                texture_data[i],
                camera_position,
                camera_rotation_matrix,
                fov_x,
                aspect_ratio,
                128,
                128,
            )

        image_array = np.array(image_data, dtype=np.uint8).reshape((height, width, 4))
        img = Image.fromarray(image_array)
        rgb_filename = f"{timestamp_str}.png"
        img.save(os.path.join(rgb_dir, rgb_filename))
        rgb_file.write(f"{timestamp_str} rgb/{rgb_filename}\n")

        # 深度画像の保存 (クリッピングと型変換)
        depth_array = np.array(depth_data, dtype=np.float32).reshape((height, width))
        depth_array = np.clip(depth_array * depth_scale, 0, 65535).astype(
            np.uint16
        )  # 重要
        depth_img = Image.fromarray(depth_array, mode="I;16")
        depth_filename = f"{timestamp_str}.png"
        depth_img.save(os.path.join(depth_dir, depth_filename))
        depth_file.write(f"{timestamp_str} depth/{depth_filename}\n")

        tx, ty, tz = camera_position
        qw, qx, qy, qz = Quaternion(matrix=camera_rotation_matrix).elements
        groundtruth_file.write(
            f"{timestamp_str} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
        )

    rgb_file.close()
    depth_file.close()
    groundtruth_file.close()
    print(f"TUM形式のデータを {output_dir} に保存しました。")


# --- 使用例 ---
if __name__ == "__main__":
    num_frames = 120
    width, height = 640, 480
    cube_size = (1, 1, 1)
    cube_position = (0, 0, 0)
    cube_orientation = (math.radians(30), math.radians(45), 0)

    textures = [
        "texture1.png",
        "texture2.jpg",
        generate_checkerboard_texture,
        generate_random_texture,
        "texture3.bmp",
        lambda w, h: [
            (int(x / w * 255), int(y / h * 255), 0, 255)
            for y in range(h)
            for x in range(w)
        ],
    ]
    output_directory = "tum_output"
    generate_tum_format_data(
        output_directory,
        num_frames,
        width,
        height,
        cube_size,
        cube_position,
        cube_orientation,
        textures,
    )

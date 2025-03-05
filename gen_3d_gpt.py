import os
import math
import numpy as np
import trimesh
import pyrender
from PIL import Image

#import pyglet
#pyglet.options['shadow_window'] = False
#window = pyglet.window.Window(visible=False)

# --- ユーティリティ関数 ---


def look_at(camera_position, target, up):
    """
    カメラ座標系からワールド座標系への変換行列を作成する。
    カメラ位置 camera_position から target を注視し、up 方向を上とする。
    戻り値は 4x4 の同次変換行列。
    """
    camera_position = np.array(camera_position, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    # カメラは -Z 軸方向を撮影方向とするため、forward の符号を反転する
    R = np.eye(4, dtype=np.float32)
    R[:3, 0] = right
    R[:3, 1] = true_up
    R[:3, 2] = -forward  # ここを修正
    R[:3, 3] = camera_position
    return R

def rotation_matrix_to_quaternion(R):
    """
    3x3 の回転行列 R からクォータニオン [qx, qy, qz, qw] に変換する。
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def create_texture_image(color, size=(256, 256)):
    """
    指定した RGB 値のテクスチャ画像（PIL.Image）を作成する。
    color は (R,G,B)（例: (255, 0, 0)）のタプル。
    """
    img = Image.new("RGB", size, color)
    return np.array(img)


def create_face(face, w, h, d):
    """
    直方体の各面（'front', 'back', 'left', 'right', 'top', 'bottom'）
    に対して、4頂点・2三角形からなる面ジオメトリ（頂点，面，UV座標）を作成する。
    w, h, d は直方体の幅，高さ，奥行き。
    戻り値は vertices, faces, uvs (すべて numpy.array)。
    """
    if face == "front":
        # 面は z = d/2
        # 外向きになるよう、頂点順序を調整（ここでは top-left, bottom-left, bottom-right, top-right）
        vertices = np.array(
            [
                [-w / 2, h / 2, d / 2],
                [-w / 2, -h / 2, d / 2],
                [w / 2, -h / 2, d / 2],
                [w / 2, h / 2, d / 2],
            ],
            dtype=np.float32,
        )
        uvs = np.array([[0, 1], [0, 0], [1, 0], [1, 1]], dtype=np.float32)
    elif face == "back":
        # 面は z = -d/2。順序は外向き (normal = -z) になるよう： top-right, bottom-right, bottom-left, top-left
        vertices = np.array(
            [
                [w / 2, h / 2, -d / 2],
                [w / 2, -h / 2, -d / 2],
                [-w / 2, -h / 2, -d / 2],
                [-w / 2, h / 2, -d / 2],
            ],
            dtype=np.float32,
        )
        uvs = np.array([[0, 1], [0, 0], [1, 0], [1, 1]], dtype=np.float32)
    elif face == "left":
        # 面は x = -w/2。順序： top-back, bottom-back, bottom-front, top-front
        vertices = np.array(
            [
                [-w / 2, h / 2, -d / 2],
                [-w / 2, -h / 2, -d / 2],
                [-w / 2, -h / 2, d / 2],
                [-w / 2, h / 2, d / 2],
            ],
            dtype=np.float32,
        )
        uvs = np.array([[0, 1], [0, 0], [1, 0], [1, 1]], dtype=np.float32)
    elif face == "right":
        # 面は x = w/2。順序： top-front, bottom-front, bottom-back, top-back
        vertices = np.array(
            [
                [w / 2, h / 2, d / 2],
                [w / 2, -h / 2, d / 2],
                [w / 2, -h / 2, -d / 2],
                [w / 2, h / 2, -d / 2],
            ],
            dtype=np.float32,
        )
        uvs = np.array([[0, 1], [0, 0], [1, 0], [1, 1]], dtype=np.float32)
    elif face == "top":
        # 面は y = h/2。順序： front-left, front-right, back-right, back-left
        vertices = np.array(
            [
                [-w / 2, h / 2, d / 2],
                [w / 2, h / 2, d / 2],
                [w / 2, h / 2, -d / 2],
                [-w / 2, h / 2, -d / 2],
            ],
            dtype=np.float32,
        )
        uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    elif face == "bottom":
        # 面は y = -h/2。順序： front-right, front-left, back-left, back-right (外向き: normal = (0,-1,0))
        vertices = np.array(
            [
                [w / 2, -h / 2, d / 2],
                [-w / 2, -h / 2, d / 2],
                [-w / 2, -h / 2, -d / 2],
                [w / 2, -h / 2, -d / 2],
            ],
            dtype=np.float32,
        )
        uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    else:
        raise ValueError("Unknown face name: " + face)

    faces_idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    return vertices, faces_idx, uvs


# --- メイン処理 ---


def main():
    # 出力ディレクトリの作成
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # 直方体の寸法（幅 w, 高さ h, 奥行 d）
    w, h, d = 2.0, 1.0, 3.0

    # 各面用のテクスチャ（例として、各面に異なる定色画像を生成）
    texture_colors = {
        "front": (255, 0, 0),  # 赤
        "back": (0, 255, 0),  # 緑
        "left": (0, 0, 255),  # 青
        "right": (255, 255, 0),  # 黄
        "top": (255, 0, 255),  # マゼンタ
        "bottom": (0, 255, 255),  # シアン
    }

    # シーンに追加する各面の Mesh を作成
    mesh_list = []
    for face in ["front", "back", "left", "right", "top", "bottom"]:
        vertices, faces_idx, uvs = create_face(face, w, h, d)
        # テクスチャ画像を作成
        texture_img = create_texture_image(texture_colors[face], size=(256, 256))

        # trimesh でメッシュ作成し、テクスチャ情報を与える
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces_idx, process=False)
        # TextureVisuals により UV 座標と画像を設定
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs, image=texture_img)
        # pyrender 用の Mesh を生成
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        mesh_list.append(pyrender_mesh)

    # pyrender シーンの作成
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])
    for m in mesh_list:
        scene.add(m)

    # シーンにライトを追加（シンプルな方向性ライト）
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    # ライトノードは、例えばシーンの上部から照らす
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([4, 4, 4])
    scene.add(light, pose=light_pose)

    # 追加のライト
    light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    light_pose2 = np.eye(4)
    light_pose2[:3,3] = np.array([-4, 4, 4])
    scene.add(light2, pose=light_pose2)

    # 斜め上からのライトを追加（look_at関数を利用して、ライトの -Z 軸が原点を向くように設定）
    diagonal_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    # ここではライトの位置として、斜め上（例: x=-4, y=8, z=-4）から原点を見るように設定
    light_pose_diag = look_at(camera_position=[-4, 8, -4], target=[0, 0, 0], up=[0, 1, 0])
    scene.add(diagonal_light, pose=light_pose_diag)

    # 点光源を作成（色は白、強度は適宜調整）
    point_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=100.0)
    # 点光源の位置を設定（例: シーンの上方かつやや横に配置）
    point_light_pose = np.eye(4)
    point_light_pose[:3, 3] = np.array([3, 5, 3])  # ここで座標を変更可能
    scene.add(point_light, pose=point_light_pose)

    # オフスクリーンレンダラーの作成（例: 640x480）
    viewport_width = 640
    viewport_height = 480
    renderer = pyrender.OffscreenRenderer(viewport_width, viewport_height)

    # カメラの設定（画角など）
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(50.0))
    
    # カメラの内部パラメータを取り出す
    proj_matrix = camera.get_projection_matrix(width=viewport_width, height=viewport_height)
    print("カメラの射影行列:")
    print(proj_matrix)
    
    # 射影行列から内部パラメータを計算
    fx = proj_matrix[0, 0] * viewport_width / 2.0
    fy = proj_matrix[1, 1] * viewport_height / 2.0
    cx = viewport_width / 2.0
    cy = viewport_height / 2.0
    
    print(f"\nカメラの内部パラメータ:")
    print(f"焦点距離 fx: {fx:.2f}")
    print(f"焦点距離 fy: {fy:.2f}")
    print(f"主点 cx: {cx:.2f}")
    print(f"主点 cy: {cy:.2f}")
    
    
    # 初期カメラ位置（仮の値）
    init_pose = np.eye(4)
    camera_node = scene.add(camera, pose=init_pose)

    # カメラ軌道のパラメータ
    num_frames = 360
    radius = 6.0  # カメラが円軌道上に配置される半径
    cam_height = 2.0  # カメラの高さ（y座標）
    dt = 0.1  # 各フレームのタイムステップ（例: 0.1秒）

    # 出力用リスト（TUM 用の groundtruth と rgb.txt）
    gt_lines = []
    rgb_lines = []

    for i in range(num_frames):
        theta = 2 * np.pi * i / num_frames
        # カメラ位置（x,z は円軌道上、y は一定）
        cam_pos = np.array(
            [radius * np.cos(theta), cam_height, radius * np.sin(theta)],
            dtype=np.float32,
        )
        # 常に原点 (0,0,0) を注視
        cam_pose = look_at(cam_pos, target=[0, 0, 0], up=[0, 1, 0])

        # カメラノードの姿勢を更新
        scene.set_pose(camera_node, cam_pose)

        # レンダリング（背景は白）
        color, _ = renderer.render(
            scene, flags=pyrender.RenderFlags.RGBA
        )
        # RGBA 画像を RGB に変換
        color = color[:, :, :3]

        # 画像を保存
        relative_path="rgb"
        os.makedirs(f"{out_dir}/{relative_path}", exist_ok=True)
        
        img_filename = f"frame_{i:04d}.png"
        img_path = os.path.join(f"{out_dir}/{relative_path}", img_filename)
        Image.fromarray(color).save(img_path)

        # カメラ姿勢を TUM 形式（timestamp tx ty tz qx qy qz qw）で記録
        timestamp = i * dt
        # 抽出する回転部分は cam_pose の上3×3
        R_cam = cam_pose[:3, :3]
        quat = rotation_matrix_to_quaternion(R_cam)
        t = cam_pose[:3, 3]
        
        # groundtruth.txt に記録（timestamp と姿勢）
        gt_line = f"{timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
        gt_lines.append(gt_line)
        
        # rgb.txt に記録（timestamp とファイル名）
        rgb_line = f"{timestamp:.6f} {relative_path}/{img_filename}"
        rgb_lines.append(rgb_line)

        print(f"Frame {i+1}/{num_frames} rendered.")

    # レンダラーの解放
    renderer.delete()

    # TUM 用ファイルの保存
    with open(os.path.join(out_dir, "groundtruth.txt"), "w") as f:
        for line in gt_lines:
            f.write(line + "\n")
    with open(os.path.join(out_dir, "rgb.txt"), "w") as f:
        for line in rgb_lines:
            f.write(line + "\n")

    print("レンダリング完了。出力はフォルダ 'output' に保存されました。")


if __name__ == "__main__":
    main()

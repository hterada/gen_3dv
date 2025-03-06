import argparse
import os
import time
from pathlib import Path
import numpy as np

from renderer import Renderer
from camera_trajectory import CameraTrajectory
from scene_builder import SceneBuilder
from tum_output import TUMOutput

def main():
    parser = argparse.ArgumentParser(description='3D SLAM Test Scene Generator')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--width', type=int, default=640, help='Output image width')
    parser.add_argument('--height', type=int, default=480, help='Output image height')
    parser.add_argument('--frames', type=int, default=360, help='Number of frames to generate')
    parser.add_argument('--radius', type=float, default=5.0, help='Camera trajectory radius')
    parser.add_argument('--height_offset', type=float, default=2.0, help='Camera height')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # GPU使用フラグ
    use_gpu = not args.no_gpu
    
    print(f"Generating {args.frames} frames of size {args.width}x{args.height}")
    print(f"Output directory: {args.output}")
    print(f"Using GPU: {use_gpu}")
    
    # レンダラーの初期化
    renderer = Renderer(width=args.width, height=args.height, use_gpu=use_gpu)
    
    # カメラ軌道の設定
    camera = CameraTrajectory(
        radius=args.radius,
        height=args.height_offset,
        num_frames=args.frames,
        use_gpu=use_gpu
    )
    
    # シーンの構築
    scene_builder = SceneBuilder(renderer, use_gpu=use_gpu)
    scene_objects, light_pos = scene_builder.build_scene()
    
    # 出力設定
    output = TUMOutput(output_dir=args.output, width=args.width, height=args.height)
    
    # カメラの投影行列の計算
    projection_matrix = camera.get_projection_matrix(fov=60.0, aspect=args.width/args.height)
    
    # カメラキャリブレーション情報の保存
    fov = 60.0
    f = 0.5 * args.height / np.tan(np.radians(fov) / 2)
    output.create_camera_calibration(
        fx=f,
        fy=f,
        cx=args.width / 2,
        cy=args.height / 2
    )
    
    # 全フレームの処理
    total_frames = camera.get_total_frames()
    start_time = time.time()
    
    for i in range(total_frames):
        # 進捗表示
        print(f"Rendering frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)", end='\r')
        
        # カメラ位置の更新
        camera_position, view_matrix, frame_idx = camera.get_next_frame_data()
        
        # レンダリング
        rgb_image, depth_image = renderer.render_scene(
            view_matrix,
            projection_matrix,
            scene_objects,
            light_pos
        )
        
        # フレームの保存
        output.save_frame(frame_idx, rgb_image, depth_image, camera_position, view_matrix)
    
    # 関連付けファイルの作成
    output.create_association_file()
    
    # 処理時間の表示
    elapsed_time = time.time() - start_time
    print(f"\nRendering completed in {elapsed_time:.2f} seconds")
    print(f"Average FPS: {total_frames / elapsed_time:.2f}")
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
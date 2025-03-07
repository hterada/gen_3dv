#!/usr/bin/env python3
"""
ビジュアルSLAMテストデータ生成プログラム
"""

import argparse
from libgen3d.scene import Scene
from libgen3d.renderer import GPURenderer

def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="GPU加速版ビジュアルSLAMテストジェネレーター")
    parser.add_argument("--width", type=int, default=640, help="出力画像の幅")
    parser.add_argument("--height", type=int, default=480, help="出力画像の高さ")
    parser.add_argument("--fov", type=float, default=60, help="視野角（度）")
    parser.add_argument("--frames", type=int, default=180, help="生成するフレーム数")
    parser.add_argument("--output", type=str, default="tum_format_output", help="出力ディレクトリのパス")
    parser.add_argument("--floor-texture", type=str, default=None, help="床のテクスチャ画像のパス")
    parser.add_argument("--floor-checker-size", type=int, default=4, help="床の市松模様の分割数（数が少ないほど大きなパターン）")
    parser.add_argument("--rows", type=int, default=3, help="立方体の行数")
    parser.add_argument("--cols", type=int, default=3, help="立方体の列数")
    parser.add_argument("--spacing", type=float, default=2.0, help="立方体間の間隔")
    parser.add_argument("--shading", action="store_true", default=True, help="シェーディングを有効にする")
    parser.add_argument("--ambient", type=float, default=0.2, help="環境光の強さ (0.0-1.0)")
    parser.add_argument("--diffuse", type=float, default=0.7, help="拡散反射の強さ (0.0-1.0)")
    parser.add_argument("--specular", type=float, default=0.3, help="鏡面反射の強さ (0.0-1.0)")
    parser.add_argument("--shininess", type=float, default=32.0, help="光沢度（鏡面反射の鋭さ）")
    parser.add_argument("--no-shading", dest="shading", action="store_false", help="シェーディングを無効にする")
    
    args = parser.parse_args()
    
    # GPU版レンダラーを初期化
    renderer = GPURenderer(
        width=args.width, 
        height=args.height, 
        fov=args.fov
    )
    
    # シェーディングの設定
    renderer.enable_shading = args.shading
    if args.shading:
        renderer.set_shading_parameters(
            ambient=args.ambient,
            diffuse=args.diffuse,
            specular=args.specular,
            shininess=args.shininess
        )
        print(f"シェーディングを有効化: ambient={args.ambient}, diffuse={args.diffuse}, "
              f"specular={args.specular}, shininess={args.shininess}")
    else:
        print("シェーディングを無効化")
    
    # シーンを作成
    scene = Scene(device=renderer.device)
    scene.create_multiple_cubes_scene(
        rows=args.rows,
        cols=args.cols,
        spacing=args.spacing,
        floor_texture_path=args.floor_texture,
        floor_checker_size=args.floor_checker_size
    )
    
    # レンダラーにシーンをセット
    renderer.set_scene(scene)
    
    # シーケンスを生成
    renderer.generate_sequence(num_frames=args.frames, output_dir=args.output)

if __name__ == "__main__":
    main()
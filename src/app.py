"""
Gradio 前端 — 3D 点云分类与分割 Demo

功能：
1. 选择预设形状 或 上传点云文件（ply/pcd/txt/npy）
2. 模式切换：分类 / 分割 / 分类+分割
3. 3D 可视化 + 结果展示
4. 旋转视角控制

运行:
  python src/app.py
  python src/app.py --device cuda
"""
import sys
import gradio as gr
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import PointCloudInference
from src.utils import (
    load_pointcloud,
    visualize_pointcloud,
    visualize_topk_classification,
    generate_shape,
    SHAPES,
)
from src.pointnet import PART_NAMES


class PointCloudApp:
    def __init__(self, device: str = "cpu", num_points: int = 1024):
        print("初始化点云 Demo...")
        self.inferencer = PointCloudInference(num_points=num_points, device=device)
        self.num_points = num_points
        print("初始化完成\n")

    def process_shape(self, shape_name: str, mode: str, elev: int, azim: int):
        """处理预设形状"""
        points = generate_shape(shape_name, self.num_points)
        return self._run_inference(points, mode, elev, azim, f"Preset: {shape_name}")

    def process_upload(self, file_path: str, mode: str, elev: int, azim: int):
        """处理上传的点云文件"""
        if not file_path:
            return None, "请上传点云文件或选择预设形状", ""

        try:
            points = load_pointcloud(file_path, self.num_points)
            return self._run_inference(points, mode, elev, azim, f"Uploaded: {Path(file_path).name}")
        except Exception as e:
            return None, f"加载失败: {e}", ""

    def _run_inference(self, points: np.ndarray, mode: str, elev: int, azim: int, title: str):
        """执行推理并生成可视化"""
        results = {}

        if mode in ["classification", "both"]:
            cls_results = self.inferencer.classify(points, top_k=5)
            cls_image = visualize_topk_classification(points, cls_results, title)
            results["cls_image"] = cls_image
            cls_text = "\n".join(f"{cls}: {prob:.2%}" for cls, prob in cls_results.items())
            results["cls_text"] = cls_text

        if mode in ["segmentation", "both"]:
            seg_labels = self.inferencer.segment(points)
            seg_image = visualize_pointcloud(
                points, seg_labels, f"{title} — Segmentation", elev=elev, azim=azim
            )
            unique_labels = np.unique(seg_labels)
            seg_parts = [PART_NAMES[l] if l < len(PART_NAMES) else f"Part_{l}" for l in unique_labels]
            results["seg_image"] = seg_image
            results["seg_text"] = f"检测到 {len(unique_labels)} 个部件: {', '.join(seg_parts)}"

        if mode == "classification":
            return results["cls_image"], results["cls_text"], ""
        elif mode == "segmentation":
            return results["seg_image"], "", results["seg_text"]
        else:
            return [results["cls_image"], results["seg_image"]], results["cls_text"], results["seg_text"]

    def build(self):
        with gr.Blocks(title="3D Point Cloud Demo") as demo:
            gr.Markdown("# 3D 点云分类与分割 Demo")
            gr.Markdown("选择预设形状或上传点云文件，体验 PointNet 的分类与分割能力。")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 输入")
                    mode = gr.Radio(
                        ["classification", "segmentation", "both"],
                        label="模式",
                        value="both",
                    )
                    shape = gr.Dropdown(
                        SHAPES, label="预设形状", value="chair"
                    )
                    file_input = gr.File(
                        label="上传点云文件（可选，覆盖预设）",
                        file_types=[".ply", ".pcd", ".txt", ".npy", ".xyz"],
                    )
                    elev = gr.Slider(0, 90, value=30, step=5, label="视角 仰角")
                    azim = gr.Slider(0, 360, value=45, step=5, label="视角 方位角")
                    run_btn = gr.Button("运行", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### 结果")
                    with gr.Row():
                        cls_output = gr.Image(label="分类结果")
                        seg_output = gr.Image(label="分割结果")
                    cls_text = gr.Textbox(label="Top-5 分类", lines=6, interactive=False)
                    seg_text = gr.Textbox(label="分割部件", lines=3, interactive=False)

            run_btn.click(
                fn=lambda f, s, m, e, a: self.process_upload(f, m, e, a) if f else self.process_shape(s, m, e, a),
                inputs=[file_input, shape, mode, elev, azim],
                outputs=[seg_output, cls_text, seg_text],
            )

            run_btn.click(
                fn=lambda f, s, m, e, a: self.process_upload(f, m, e, a) if f else self.process_shape(s, m, e, a),
                inputs=[file_input, shape, mode, elev, azim],
                outputs=[cls_output, cls_text, seg_text],
            )

            with gr.Tab("帮助"):
                gr.Markdown("""
                ### 使用说明

                **预设形状**：选择一个形状（球/立方体/圆柱/锥/椅子/桌子/飞机），点击运行。

                **上传文件**：支持 .ply / .pcd / .txt / .npy / .xyz 格式。上传后会覆盖预设形状。

                **模式**：
                - `classification`：输出 Top-5 类别预测
                - `segmentation`：输出逐点部件分割
                - `both`：同时输出分类和分割

                **视角控制**：调整仰角和方位角改变 3D 可视化角度。

                ### 技术说明

                - **PointNet**：用 max-pooling 聚合无序点云的全局特征（对称函数），解决置换不变性
                - **分类**：ModelNet40（40类物体）
                - **分割**：ShapeNet Part（16类物体 × 部件级分割）

                ### 注意

                如果没有预训练权重（`models/` 目录为空），模型为随机初始化，分类/分割结果不可靠。
                请下载预训练权重或自行训练后使用。

                ```bash
                # 预训练权重下载（可选）
                # PointNet 官方权重: https://github.com/charlesq34/pointnet
                ```
                """)

        return demo


if __name__ == "__main__":
    args = sys.argv[1:]
    device = "cpu"
    for i, arg in enumerate(args):
        if arg == "--device" and i + 1 < len(args):
            device = args[i + 1]
        elif arg == "--cuda":
            device = "cuda"

    app = PointCloudApp(device=device)
    demo = app.build()
    demo.launch(server_name="0.0.0.0", server_port=7860)

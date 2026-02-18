import cv2
import numpy as np
from pathlib import Path
import os

def compute_cover_fields(mask_path, pixel_size_mm, fov_width_mm=2.0, fov_height_mm=2.0,
                         reference_point=(0,0), image_path=None, output_vis_path="coverage_validation.png",
                         method='bbox'):
    """
    根据掩码图像计算每个样本所需的显微镜视场（支持矩形FOV），并生成验证图像。

    参数：
        mask_path (str): 二值掩码图像路径（单通道，样本区域为255）
        pixel_size_mm (float): 每个像素对应的毫米数
        fov_width_mm (float): 视场宽度，默认2.0 mm
        fov_height_mm (float): 视场高度，默认2.0 mm
        reference_point (tuple): 图像左上角对应的物理坐标（x0, y0），默认为(0,0)
        image_path (str, optional): 原始图像路径，用于绘制背景。若为None，则用黑色背景
        output_vis_path (str): 验证图像保存路径（若为相对路径，则保存在mask所在目录）
        method (str): 覆盖方法，'bbox' 使用bounding box均匀网格（居中），'exact' 使用像素精确覆盖

    返回：
        tuple: (samples_info, all_centers)
            samples_info: dict，每个样本的详细信息，格式为
                {
                    'sample_1': {
                        'num_fields': int,
                        'centers': [(x1,y1,w1,h1), (x2,y2,w2,h2), ...]  # 每个元素为 (中心x_mm, 中心y_mm, 视场宽_mm, 视场高_mm)
                    },
                    ...
                }
            all_centers: list，所有视场中心坐标及尺寸的平面列表，按样本顺序排列
                    每个元素为 (x_mm, y_mm, w_mm, h_mm)
    """
    # 展开用户路径
    mask_path = Path(mask_path).expanduser()
    if not mask_path.exists():
        raise FileNotFoundError(f"掩码文件不存在：{mask_path}")

    # 处理输出路径：若为相对路径，则保存在mask所在目录
    output_vis = Path(output_vis_path)
    if not output_vis.is_absolute():
        output_vis = mask_path.parent / output_vis
    output_vis = output_vis.expanduser()

    if image_path:
        image_path = Path(image_path).expanduser()

    # 读取掩码
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取掩码图像：{mask_path}")

    # 确保二值化
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 读取原图（如果提供）用于验证
    if image_path and image_path.exists():
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"警告：无法读取原图 {image_path}，将使用黑色背景")
            img_bgr = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    else:
        img_bgr = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # 创建验证图像副本
    vis_img = img_bgr.copy()

    # 查找所有连通区域（每个样本）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    samples_info = {}
    all_centers = []

    half_w = fov_width_mm / 2.0
    half_h = fov_height_mm / 2.0

    # 物理坐标到像素坐标的辅助函数
    def phys_to_pixel(phys_x, phys_y):
        pix_x = (phys_x - reference_point[0]) / pixel_size_mm
        pix_y = (phys_y - reference_point[1]) / pixel_size_mm
        return int(round(pix_x)), int(round(pix_y))

    for idx, contour in enumerate(contours, start=1):
        sample_name = f"sample_{idx}"

        # 获取样本的 bounding box（像素坐标）
        x, y, w, h = cv2.boundingRect(contour)

        # 计算样本轮廓质心（作为中心）
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = x + w // 2
            cY = y + h // 2
        center_x_mm = cX * pixel_size_mm + reference_point[0]
        center_y_mm = cY * pixel_size_mm + reference_point[1]

        # 转换为物理坐标范围
        x_min_mm = x * pixel_size_mm + reference_point[0]
        y_min_mm = y * pixel_size_mm + reference_point[1]
        x_max_mm = (x + w) * pixel_size_mm + reference_point[0]
        y_max_mm = (y + h) * pixel_size_mm + reference_point[1]
        width_mm = x_max_mm - x_min_mm
        height_mm = y_max_mm - y_min_mm

        if method == 'bbox':
            # ---------- 方法1：基于bounding box的居中均匀网格 ----------
            # 计算所需网格行列数
            cols = int(np.ceil(width_mm / fov_width_mm))
            rows = int(np.ceil(height_mm / fov_height_mm))

            # 计算网格整体尺寸
            total_width_mm = cols * fov_width_mm
            total_height_mm = rows * fov_height_mm

            # 计算起始左上角，使网格整体中心与样本中心对齐
            left = center_x_mm - total_width_mm / 2
            top = center_y_mm - total_height_mm / 2

            selected_centers = []

            # 判断是否为单视场且需要调整尺寸
            if rows == 1 and cols == 1:
                # 检查是否需要调整视场尺寸
                new_w = fov_width_mm
                new_h = fov_height_mm
                # if width_mm < (2/3) * fov_width_mm:
                #     new_w = min(fov_width_mm, width_mm * 1.5)  # 缩小至样本宽度的1.2倍，但不超过原始FOV
                if height_mm < (2/3) * fov_height_mm:
                    new_h = min(fov_height_mm, height_mm * 1.5)
                # 如果任一维度调整，则使用新尺寸
                if new_w != fov_width_mm or new_h != fov_height_mm:
                    # 重新计算单网格的左上角（中心仍为样本中心）
                    left = center_x_mm - new_w / 2
                    top = center_y_mm - new_h / 2
                    selected_centers.append((center_x_mm, center_y_mm, new_w, new_h))
                else:
                    selected_centers.append((center_x_mm, center_y_mm, fov_width_mm, fov_height_mm))
            else:
                # 多视场：使用原始尺寸生成所有网格
                for row in range(rows):
                    for col in range(cols):
                        cx = left + (col + 0.5) * fov_width_mm
                        cy = top + (row + 0.5) * fov_height_mm
                        selected_centers.append((cx, cy, fov_width_mm, fov_height_mm))

        elif method == 'exact':
            # ---------- 方法2：精确覆盖（只保留包含样本像素的网格）----------
            # （此方法未实现居中，保持原逻辑）
            # 生成候选网格中心（覆盖bounding box周边区域）
            first_center_x = np.floor(x_min_mm / fov_width_mm) * fov_width_mm + half_w
            first_center_y = np.floor(y_min_mm / fov_height_mm) * fov_height_mm + half_h
            last_center_x = np.ceil(x_max_mm / fov_width_mm) * fov_width_mm - half_w
            last_center_y = np.ceil(y_max_mm / fov_height_mm) * fov_height_mm - half_h

            centers_x = np.arange(first_center_x, last_center_x + 1e-9, fov_width_mm)
            centers_y = np.arange(first_center_y, last_center_y + 1e-9, fov_height_mm)

            grid_centers = [(cx, cy) for cy in centers_y for cx in centers_x]

            # 创建只包含当前样本的掩码
            sample_mask = np.zeros_like(mask)
            cv2.drawContours(sample_mask, [contour], -1, 255, -1)

            selected_centers = []
            for cx, cy in grid_centers:
                left_pix, top_pix = phys_to_pixel(cx - half_w, cy - half_h)
                right_pix, bottom_pix = phys_to_pixel(cx + half_w, cy + half_h)
                left_pix = max(0, min(left_pix, mask.shape[1]-1))
                right_pix = max(0, min(right_pix, mask.shape[1]-1))
                top_pix = max(0, min(top_pix, mask.shape[0]-1))
                bottom_pix = max(0, min(bottom_pix, mask.shape[0]-1))
                if left_pix >= right_pix or top_pix >= bottom_pix:
                    continue
                roi = sample_mask[top_pix:bottom_pix, left_pix:right_pix]
                if np.any(roi):
                    selected_centers.append((cx, cy, fov_width_mm, fov_height_mm))  # 精确覆盖不调整尺寸
        else:
            raise ValueError("method 必须是 'bbox' 或 'exact'")

        # 保存样本信息
        samples_info[sample_name] = {
            'num_fields': len(selected_centers),
            'centers': selected_centers
        }
        all_centers.extend(selected_centers)

        # 在验证图像上绘制当前样本的轮廓（绿色）
        cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)

        # 绘制选中的网格矩形（红色边框）和中心点（蓝色）
        for (cx, cy, w_fov, h_fov) in selected_centers:
            half_w_fov = w_fov / 2
            half_h_fov = h_fov / 2
            left_pix, top_pix = phys_to_pixel(cx - half_w_fov, cy - half_h_fov)
            right_pix, bottom_pix = phys_to_pixel(cx + half_w_fov, cy + half_h_fov)
            # 确保坐标顺序正确
            left_pix = max(0, min(left_pix, mask.shape[1]-1))
            right_pix = max(0, min(right_pix, mask.shape[1]-1))
            top_pix = max(0, min(top_pix, mask.shape[0]-1))
            bottom_pix = max(0, min(bottom_pix, mask.shape[0]-1))
            if left_pix < right_pix and top_pix < bottom_pix:
                cv2.rectangle(vis_img, (left_pix, top_pix), (right_pix, bottom_pix), (0, 0, 255), 1)
            # 绘制中心点
            center_pix_x, center_pix_y = phys_to_pixel(cx, cy)
            cv2.circle(vis_img, (center_pix_x, center_pix_y), 3, (255, 0, 0), -1)

    # 保存验证图像
    cv2.imwrite(str(output_vis), vis_img)
    print(f"验证图像已保存至：{output_vis}")

    return samples_info, all_centers

# ========== 使用示例 ==========
if __name__ == "__main__":
    mask_file = "~/Documents/LineScanOCT/mask3.png"
    # 假设原图与mask同目录，文件名类似但后缀为.jpg，请根据实际情况修改
    image_file = "~/Documents/LineScanOCT/mask3.jpg"  # 可选，如果不需要可设为None
    pixel_size = 0.05   # 每个像素0.01 mm，需根据实际调整
    fov_w = 2.0         # 视场宽度2.5 mm
    fov_h = 3.0         # 视场高度2.0 mm

    # 选择方法：'bbox' 或 'exact'
    method = 'bbox'  # 尝试第一种方法

    samples_info, all_centers = compute_cover_fields(
        mask_file,
        pixel_size_mm=pixel_size,
        fov_width_mm=fov_w,
        fov_height_mm=fov_h,
        reference_point=(0,0),
        image_path=image_file,
        output_vis_path="coverage_check.png",  # 相对路径，会自动保存在mask所在目录
        method=method
    )

    # 打印每个样本的信息
    for sample, info in samples_info.items():
        print(f"{sample}: 需要 {info['num_fields']} 个视场")
        for i, (x, y, w, h) in enumerate(info['centers'], 1):
            print(f"  视场 {i}: 中心 ({x:.3f} mm, {y:.3f} mm), 尺寸 {w:.3f} x {h:.3f} mm")

    # 全局拍摄顺序列表
    print("\n全局拍摄顺序（按样本依次）:")
    for idx, (x, y, w, h) in enumerate(all_centers, 1):
        print(f"步骤 {idx}: 移动到 ({x:.3f} mm, {y:.3f} mm), 视场尺寸 {w:.3f} x {h:.3f} mm")

    # 保存为CSV（包含尺寸信息）
    import csv
    with open('shooting_sequence.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'center_x_mm', 'center_y_mm', 'fov_width_mm', 'fov_height_mm'])
        for step, (x, y, w, h) in enumerate(all_centers, 1):
            writer.writerow([step, x, y, w, h])
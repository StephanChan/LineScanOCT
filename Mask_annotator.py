#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 17:41:32 2026

@author: stephanchang
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk

class PolygonAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("多孔板轮廓标注工具")

        # 图像相关变量
        self.image_path = None
        self.original_image = None          # PIL Image 格式
        self.image_array = None              # numpy 数组 (用于保存mask)
        self.img_width = 0
        self.img_height = 0

        # 视图变换参数
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.min_scale = 0.1
        self.max_scale = 5.0

        # 标注数据
        self.polygons = []                    # 所有已完成的多边形 (每个多边形为 [[x,y],...] 列表)
        self.current_polygon = []              # 当前正在绘制的多边形顶点 (图像坐标)
        self.temp_line_ids = []                 # 临时线条 (当前多边形线段和跟随鼠标的线) 的canvas对象ID

        # 画布和按钮框架
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg='gray')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 垂直滚动条（可选，但缩放平移后可能不需要，保留以备扩展）
        v_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=v_scroll.set)

        h_scroll = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.configure(xscrollcommand=h_scroll.set)

        # 按钮框架
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.btn_load = tk.Button(self.button_frame, text="加载图片", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_finish = tk.Button(self.button_frame, text="完成当前多边形", command=self.finish_polygon)
        self.btn_finish.pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(self.button_frame, text="保存Mask", command=self.save_mask)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(self.button_frame, text="清除所有", command=self.clear_all)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # 绑定鼠标事件
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)        # 左键点击添加顶点
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)            # 左键移动时更新临时线（可选，也可以只在点击时画）
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)    # 左键释放停止绘制临时线（可选）
        self.canvas.bind("<ButtonPress-2>", self.on_middle_press)      # 中键按下启动平移
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)           # 中键拖动平移
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)       # 右键点击完成当前多边形（备用）
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)           # Windows 滚轮
        self.canvas.bind("<Button-4>", self.on_mousewheel)             # Linux 滚轮向上
        self.canvas.bind("<Button-5>", self.on_mousewheel)             # Linux 滚轮向下

        # 初始化状态
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.is_dragging = False

    def load_image(self):
        """打开文件对话框加载图片，并初始化显示"""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if not file_path:
            return
        self.image_path = file_path
        self.original_image = Image.open(file_path).convert("RGB")
        self.img_width, self.img_height = self.original_image.size
        self.image_array = np.array(self.original_image)  # 用于保存mask时参考尺寸

        # 重置视图参数
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.polygons = []
        self.current_polygon = []
        self._clear_temp_lines()
        self._redraw_canvas()

    def _clear_temp_lines(self):
        """删除所有临时线条"""
        for line_id in self.temp_line_ids:
            self.canvas.delete(line_id)
        self.temp_line_ids = []

    def _redraw_canvas(self):
        """根据当前 scale/offset 重新绘制背景图像和所有多边形"""
        if self.original_image is None:
            return

        # 清空画布
        self.canvas.delete("all")

        # 计算缩放后的图像尺寸
        new_width = int(self.img_width * self.scale)
        new_height = int(self.img_height * self.scale)

        # 缩放图像
        resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized_image)

        # 在画布上显示图像 (图像左上角坐标为 (offset_x, offset_y))
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo)

        # 绘制所有已完成的多边形轮廓 (仅轮廓)
        for poly in self.polygons:
            self._draw_polygon(poly, outline='green', width=2, fill='')

        # 绘制当前正在绘制的多边形顶点和线段
        if self.current_polygon:
            # 绘制顶点之间的线段
            points = self.current_polygon
            for i in range(len(points)-1):
                x1, y1 = self._img_to_canvas(points[i][0], points[i][1])
                x2, y2 = self._img_to_canvas(points[i+1][0], points[i+1][1])
                line_id = self.canvas.create_line(x1, y1, x2, y2, fill='red', width=2)
                self.temp_line_ids.append(line_id)
            # 绘制顶点
            for (x, y) in points:
                cx, cy = self._img_to_canvas(x, y)
                r = 3
                oval_id = self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill='red')
                self.temp_line_ids.append(oval_id)

        # 更新画布滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _img_to_canvas(self, x, y):
        """图像坐标 -> 画布坐标"""
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def _canvas_to_img(self, cx, cy):
        """画布坐标 -> 图像坐标"""
        x = (cx - self.offset_x) / self.scale
        y = (cy - self.offset_y) / self.scale
        return x, y

    def _draw_polygon(self, poly, outline='green', width=2, fill=''):
        """在画布上绘制一个多边形（给定顶点图像坐标）"""
        if len(poly) < 3:
            return
        canvas_points = []
        for (x, y) in poly:
            cx, cy = self._img_to_canvas(x, y)
            canvas_points.extend([cx, cy])
        self.canvas.create_polygon(canvas_points, outline=outline, width=width, fill=fill)

    # 鼠标事件处理
    def on_left_press(self, event):
        """左键点击：添加一个顶点"""
        if self.original_image is None:
            return
        # 获取画布坐标
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        # 转换为图像坐标
        ix, iy = self._canvas_to_img(cx, cy)
        # 边界限制
        ix = max(0, min(ix, self.img_width-1))
        iy = max(0, min(iy, self.img_height-1))
        self.current_polygon.append([ix, iy])

        # 重新绘制以显示新的顶点和线段
        self._clear_temp_lines()
        self._redraw_canvas()

    def on_mouse_move(self, event):
        """鼠标移动（左键按住时）：显示从最后一个顶点到鼠标当前位置的临时线"""
        if self.original_image is None or not self.current_polygon:
            return
        # 获取鼠标画布坐标
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        # 转换为图像坐标（仅用于显示，不实际添加点）
        ix, iy = self._canvas_to_img(cx, cy)
        ix = max(0, min(ix, self.img_width-1))
        iy = max(0, min(iy, self.img_height-1))

        # 删除旧的临时线条（除了顶点和已有线段，我们只更新最后一段临时线）
        # 简单起见，每次移动时重新绘制当前多边形（包括顶点和所有线段）和额外的鼠标线
        self._clear_temp_lines()
        self._redraw_canvas()

        # 再画一条从最后一个顶点到鼠标位置的线（作为临时线）
        if self.current_polygon:
            last = self.current_polygon[-1]
            x1, y1 = self._img_to_canvas(last[0], last[1])
            x2, y2 = self._img_to_canvas(ix, iy)
            line_id = self.canvas.create_line(x1, y1, x2, y2, fill='yellow', width=2, dash=(4,2))
            self.temp_line_ids.append(line_id)

    def on_left_release(self, event):
        """左键释放：清除跟随鼠标的临时线（已经由 on_mouse_move 更新，释放后应删除最后那条线）"""
        # 重新绘制，去掉临时鼠标线
        self._clear_temp_lines()
        self._redraw_canvas()

    def on_middle_press(self, event):
        """中键按下：记录起始位置，准备平移"""
        self.drag_start_x = self.canvas.canvasx(event.x)
        self.drag_start_y = self.canvas.canvasy(event.y)
        self.is_dragging = True
        self.canvas.config(cursor="fleur")

    def on_middle_drag(self, event):
        """中键拖动：平移视图"""
        if not self.is_dragging:
            return
        # 当前画布坐标
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        # 计算偏移变化
        dx = cx - self.drag_start_x
        dy = cy - self.drag_start_y
        self.offset_x += dx
        self.offset_y += dy
        # 更新拖动起点
        self.drag_start_x = cx
        self.drag_start_y = cy
        # 重绘
        self._redraw_canvas()

    def on_middle_release(self, event):
        """中键释放：结束平移"""
        self.is_dragging = False
        self.canvas.config(cursor="")

    def on_right_click(self, event):
        """右键点击：快速完成当前多边形"""
        self.finish_polygon()

    def on_mousewheel(self, event):
        """鼠标滚轮缩放，保持鼠标指向的图像点不变"""
        if self.original_image is None:
            return

        # 获取鼠标在画布上的位置
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        # 计算缩放前鼠标对应的图像坐标
        img_x, img_y = self._canvas_to_img(cx, cy)

        # 根据滚轮方向调整缩放因子
        if event.num == 4 or event.delta > 0:   # 向上滚，放大
            self.scale *= 1.1
        elif event.num == 5 or event.delta < 0: # 向下滚，缩小
            self.scale /= 1.1

        # 限制缩放范围
        self.scale = max(self.min_scale, min(self.max_scale, self.scale))

        # 调整偏移量，使鼠标指向的图像点不变
        new_cx, new_cy = self._img_to_canvas(img_x, img_y)
        self.offset_x += cx - new_cx
        self.offset_y += cy - new_cy

        # 重绘
        self._redraw_canvas()

    def finish_polygon(self):
        """完成当前多边形：封闭并添加到 polygons 列表"""
        if len(self.current_polygon) < 3:
            messagebox.showwarning("警告", "多边形至少需要3个顶点")
            return
        # 封闭多边形（其实不一定要添加最后一个点到第一个点的线段，但为了保存完整多边形，直接添加即可）
        # 注意：我们存储的顶点列表就是闭合的（第一个和最后一个不重复），绘制时会自动闭合（使用create_polygon）
        self.polygons.append(self.current_polygon.copy())
        self.current_polygon = []
        self._clear_temp_lines()
        self._redraw_canvas()

    def save_mask(self):
        """生成并保存掩码图像"""
        if not self.polygons:
            messagebox.showwarning("警告", "没有标注任何多边形")
            return
        if self.image_array is None:
            return

        # 创建空白掩码 (单通道，黑色背景)
        mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        # 将每个多边形填充为白色 (255)
        for poly in self.polygons:
            # 多边形顶点需要是整数坐标，且格式为 numpy 数组
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)

        # 保存文件对话框
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if save_path:
            cv2.imwrite(save_path, mask)
            messagebox.showinfo("完成", f"掩码已保存至: {save_path}")

    def clear_all(self):
        """清除所有多边形，重新开始"""
        if messagebox.askyesno("确认", "清除所有已标注的多边形？"):
            self.polygons = []
            self.current_polygon = []
            self._clear_temp_lines()
            self._redraw_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    app = PolygonAnnotator(root)
    root.mainloop()
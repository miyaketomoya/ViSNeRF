from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, Range1d, Label
from bokeh.plotting import figure

from opt import config_parser
from viewer import ImageSynthetizer

import numpy as np

from bokeh.models import Button
from PIL import Image
import os
from datetime import datetime





def pack_rgba_uint32(img: np.ndarray) -> np.ndarray:
    """
    Bokeh の image_rgba が期待する 32bit 整数配列 (h, w) に変換する。
    - img: shape (h, w, 3) or (h, w, 4), dtype=uint8, order=RGB(A)
    - return: shape (h, w), dtype=uint32, 各ピクセルは 0xRRGGBBAA
    """
    h, w, c = img.shape
    # チャンネル数が 3 の場合は α=255 を追加
    if c == 3:
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)
    # 各チャネルを 32bit 内の位置にビットシフト
    r = img[:, :, 0].astype(np.uint32)
    g = img[:, :, 1].astype(np.uint32)
    b = img[:, :, 2].astype(np.uint32)
    a = img[:, :, 3].astype(np.uint32)
    rgba32 = (a << 24) | (b << 16) | (g << 8) | r  # 0xAABBGGRR
    # Bokeh は左下が原点なので上下反転して渡す
    return np.flipud(rgba32)

# --- 引数パース & インスタンス化 ---
args = config_parser()
image_synthesizer = ImageSynthetizer(args)

# --- Bokeh Figure のセットアップ ---
w, h = image_synthesizer.w, image_synthesizer.h

p_rgb = figure(x_range=Range1d(0, w), y_range=Range1d(0, h),
               tools="zoom_in,zoom_out,reset", width=w, height=h)
p_rgb.xaxis.visible = False
p_rgb.yaxis.visible = False

p_depth = figure(x_range=Range1d(0, w), y_range=Range1d(0, h),
                 tools="zoom_in,zoom_out,reset", width=w, height=h)
p_depth.xaxis.visible = False
p_depth.yaxis.visible = False

# --- 初期画像を描画 ---
t0, th0, ph0 = 50, 23, -63
img0, depth0, x0, y0, z0 = image_synthesizer.visualize(th0, ph0, t0)

rgba0 = pack_rgba_uint32(img0)
r_rgb = p_rgb.image_rgba(image=[rgba0], x=0, y=0, dw=w, dh=h)

depth_rgba0 = pack_rgba_uint32(depth0)
r_depth = p_depth.image_rgba(image=[depth_rgba0], x=0, y=0, dw=w, dh=h)

label = Label(x=10, y=h-20,
              text=f"x: {x0:.2f}, y: {y0:.2f}, z: {z0:.2f}, t: {t0}",
              text_font_size="10pt", text_color="white")
p_rgb.add_layout(label)

# --- スライダー ---
slider_t   = Slider(start=image_synthesizer.min_params[0],
                    end=image_synthesizer.max_params[0],
                    value=t0, step=1, title="Time step", width=w)
slider_th  = Slider(start=-180., end=180., value=th0, step=1., title="θ", width=w)
slider_ph  = Slider(start=-180., end=180., value=ph0, step=1., title="φ", width=w)

# 保存ボタンのコールバック関数
def save_rgb_image():
    t = slider_t.value
    th = slider_th.value
    ph = slider_ph.value
    
    # 現在の画像を取得
    img, _, x, y, z = image_synthesizer.visualize(th, ph, t)
    
    # ファイル名を生成（タイムスタンプ付き）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rgb_t{t:.0f}_th{th:.1f}_ph{ph:.1f}_{timestamp}.png"
    
    # 画像を保存
    Image.fromarray(img).save(filename)
    print(f"RGB画像を保存しました: {filename}")

# 保存ボタンを作成
save_button = Button(label="RGB画像を保存", width=200, button_type="success")
save_button.on_click(save_rgb_image)

# --- コールバック ---
def update_data(attr, old, new):
    t   = slider_t.value
    th  = slider_th.value
    ph  = slider_ph.value
    img, depth_img, x, y, z = image_synthesizer.visualize(th, ph, t)

    # RGB 更新
    rgba = pack_rgba_uint32(img)
    r_rgb.data_source.data["image"] = [rgba]

    # 深度（カラー）更新
    depth_rgba = pack_rgba_uint32(depth_img)
    r_depth.data_source.data["image"] = [depth_rgba]

    # ラベル更新
    label.text = f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}, t: {t:.2f}"

for s in (slider_t, slider_th, slider_ph):
    s.on_change("value", update_data)

# # --- レイアウト & ドキュメント登録 ---
# layout = column(slider_t, slider_th, slider_ph,
#                 row(p_rgb, p_depth))
# curdoc().add_root(layout)

# --- レイアウト & ドキュメント登録 ---
layout = column(
    row(slider_t, save_button),  # スライダーとボタンを横並び
    slider_th, 
    slider_ph,
    row(p_rgb, p_depth)
)
curdoc().add_root(layout)

# python -m bokeh serve --show bokeh_viewer.py --args --config /data/data2/tomoya/for_visverf/configs/cube_for_test.txt
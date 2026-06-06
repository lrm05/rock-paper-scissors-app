# -*- coding: utf-8 -*-
from ultralytics import YOLO  # 引入 YOLOv8 模型库，负责 AI 手势识别核心算法
import cv2  # OpenCV 库，负责图像和视频的高级处理（读写、缩放、颜色转换）
import numpy as np  # 数学计算库，图像在计算机眼里就是一堆由数字组成的 NumPy 矩阵
import os  # 系统操作库，用来检查文件是否存在、删除文件等

import streamlit as st  # 网页开发框架，用来把 Python 代码快速变成高大上的网页
from PIL import Image  # Python 图像处理标准库，用来打开和处理网页上传的静态图片
import base64  # 编码库，用来把背景图片转换成字符串嵌入到网页样式中
import tempfile  # 临时文件库，用来在服务器后台安全地暂存用户上传的视频

# ⚠️ 必须放在代码最上面：设置网页在浏览器标签页上的标题，并将布局设为占满全屏的“宽屏”模式
st.set_page_config(page_title="石头剪刀布识别", layout="wide")


# ==============================================================================
# 1. 加载 AI 模型（核心大脑）
# ==============================================================================
@st.cache_resource
def load_model():
    model = YOLO('best.pt')  # 加载你训练好的目标检测模型权重文件 'best.pt'
    model.model.names = {0: '石头', 1: '剪刀', 2: '布'}
    return model


model = load_model()  # 实例化模型
gesture_dict = {0: '石头', 1: '剪刀', 2: '布'}  # 建立一个数字到汉字的快捷字典


# ==============================================================================
# 2. 读取背景图并生成自定义 CSS 样式（视觉美化）
# ==============================================================================
def set_bg_and_css(image_path):
    if not os.path.exists(image_path):
        st.warning(f"⚠️ 找不到背景图片 {image_path}，将使用纯色背景。")
        return

    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

    css = f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpeg;base64,{encoded_string}'); 
        background-size: cover;             
        background-position: center bottom;  
        background-attachment: fixed;        
    }}
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.6); 
        padding: 2rem;                              
        border-radius: 15px;                        
        margin-top: 2rem;                           
        margin-bottom: 180px;                       
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


set_bg_and_css('bg.jpg')

# ==============================================================================
# 3. 页面标题区
# ==============================================================================
st.markdown("<h1 style='text-align: center; color: #333;'>✨ ✊✌️✋ 终极石头剪刀布对决 ✨</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>单人挑战 AI，双人开启裁判模式！</h3>", unsafe_allow_html=True)
st.write("---") 

# ==============================================================================
# 4. 操作输入区（提供三种出招方式）
# ==============================================================================
input_mode = st.radio("请选择你的出招方式：", ["📂 上传本地图片", "📷 开启摄像头拍照", "🎬 上传视频文件"], horizontal=True)

img_file_buffer = None  
video_file_buffer = None  

if input_mode == "📂 上传本地图片":
    img_file_buffer = st.file_uploader("📂 放入比赛截图 (支持 jpg, png)", type=['jpg', 'jpeg', 'png'])
elif input_mode == "📷 开启摄像头拍照":
    img_file_buffer = st.camera_input("📷 点击下方按钮拍摄你的手势")
else:
    video_file_buffer = st.file_uploader("🎬 放入视频文件 (支持 mp4, avi, mov)", type=['mp4', 'avi', 'mov'])

# ==============================================================================
# 5. 识别与判断逻辑
# ==============================================================================

# ------------------------------------------------------------------------------
# 情况一：用户输入的是【图片】或者【摄像头拍照】
# ------------------------------------------------------------------------------
if img_file_buffer is not None:
    image_pil = Image.open(img_file_buffer)  
    img_np = np.array(image_pil)  

    if img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)  
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  

    spacer_left, img_col1, img_col2, spacer_right = st.columns([1, 2, 2, 1])

    with img_col1:
        st.image(image_pil, caption="你的招式", use_column_width=True)

    with st.spinner('🤖 AI裁判正在火眼金睛识别中...'):
        results = model(img_bgr, conf=0.5, iou=0.4)
        res_image_bgr = results[0].plot()  
        res_image_rgb = cv2.cvtColor(res_image_bgr, cv2.COLOR_BGR2RGB)  

        boxes = results[0].boxes  
        num_hands = len(boxes)  

        if num_hands == 0:
            msg = "🤔 哎呀，我好像没看清画面里有什么招式，要不要重拍一张？"
            msg_type = "warning"
        elif num_hands == 1:
            top_class = int(boxes.cls[0].item())
            if top_class == 0:  
                msg = "🪨 你出了【石头】！那我出【布】吧，哈哈，我赢啦！🎉"
            elif top_class == 1:  
                msg = "✂️ 你出了【剪刀】！那我出【石头】吧，我又赢啦！😎"
            elif top_class == 2:  
                msg = "🖐️ 你出了【布】！那我出【剪刀】，哈哈，我赢啦！🎉"
            msg_type = "info"
        elif num_hands == 2:
            box1, box2 = boxes[0], boxes[1]
            x1_1, x1_2 = box1.xyxy[0][0].item(), box2.xyxy[0][0].item()

            if x1_1 < x1_2:
                left_class, right_class = int(box1.cls[0].item()), int(box2.cls[0].item())
            else:
                left_class, right_class = int(box2.cls[0].item()), int(box1.cls[0].item())

            left_name = gesture_dict.get(left_class, "未知")
            right_name = gesture_dict.get(right_class, "未知")

            if left_class == right_class:
                result = "🤝 哎呀，是平局，心有灵犀！"
            elif (left_class == 0 and right_class == 1) or \
                    (left_class == 1 and right_class == 2) or \
                    (left_class == 2 and right_class == 0):
                result = "🏆 裁判宣布：【左方】玩家获得胜利！"
            else:
                result = "🏆 裁判宣布：【右方】玩家获得胜利！"

            msg = f"⚔️ 巅峰对决！左方出了【{left_name}】，右方出了【{right_name}】。\n\n{result}"
            msg_type = "success"
        else:
            msg = f"😱 哇塞，画面里检测到了 {num_hands} 只手！裁判看不过来了，一次最多两人对战哦！"
            msg_type = "error"

    with img_col2:
        st.image(res_image_rgb, caption="AI 识别结果", use_column_width=True)

    st.write("---")
    st.subheader("🤖 裁判播报")
    if msg_type == "warning":
        st.warning(msg)  
    elif msg_type == "error":
        st.error(msg)  
    elif msg_type == "success":
        st.success(msg)  
    else:
        st.info(msg)  

# ------------------------------------------------------------------------------
# 情况二：用户输入的是【视频文件】（🌟 离线高性能同步转码版-彻底消灭卡顿）
# ------------------------------------------------------------------------------
elif video_file_buffer is not None:
    st.write("---")

    video_title_placeholder = st.empty()
    video_title_placeholder.markdown(
        "<h3 style='text-align: center; color: #333;'>🎬 AI 裁判正在全速解析并导出带框视频流...</h3>", unsafe_allow_html=True)

    # 1. 建立服务器后台临时输入文件
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file_buffer.read())  
    tfile.close()  

    # 2. 使用 OpenCV 打开这个原视频并获取它的关键属性
    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 24  # 防止部分异常视频获取不到帧率导致程序卡死
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. 建立一个临时的【带框输出视频文件】
    out_tfile_path = tfile.name + "_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 使用标准 MP4 格式在后台默默写入
    out = cv2.VideoWriter(out_tfile_path, fourcc, fps, (width, height))

    # 4. 在前端生成高级、直观的百分比进度条，安抚用户情绪
    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0  

    # 5. 后台全速闭眼运行（不跳帧、不缩放，保证画质完好无损）
    while cap.isOpened():
        ret, frame = cap.read()  
        if not ret:
            break  

        frame_count += 1

        # 让 AI 默默在后台识别当前帧，不把图发往前端网页（零网络开销，速度飞起）
        results = model(frame, conf=0.5, iou=0.4, verbose=False)

        # 绘制彩色检测框
        res_frame_bgr = results[0].plot()

        # 把这帧已经画好框的完美画面，默默写进新的视频文件里
        out.write(res_frame_bgr)

        # 实时刷新网页上的进度条百分比
        percent = min(int((frame_count / total_frames) * 100), 100)
        progress_bar.progress(percent)
        status_text.markdown(f"<p style='text-align: center; color: #666;'>⚡ 正在精细追踪第 <b>{frame_count}</b> / {total_frames} 帧 ({percent}%)</p>", unsafe_allow_html=True)

    # 释放原视频和新视频的文件锁
    cap.release()
    out.release()

    # 6. 🌟 核心魔法：由于 OpenCV 直接导出的视频编码，网页多半会黑屏拒绝播放。
    # 我们用系统内置的 ffmpeg 命令，一秒钟将其转码为网页通用的 H.264 编码和 yuv420p 色彩格式。
    web_ready_path = tfile.name + "_ready.mp4"
    # 🌟 新增：-movflags +faststart 让视频变成真正的“流媒体”，支持边下边播，拒绝转圈！
    # 同时加上 -preset veryfast 稍微压缩一下体积，传输更快！
    cmd = f"ffmpeg -y -i {out_tfile_path} -vcodec libx264 -preset veryfast -pix_fmt yuv420p -movflags +faststart {web_ready_path}"
    os.system(cmd)

    # 7. 清除已经跑满的进度条组件，还网页一片清净
    progress_bar.empty()
    status_text.empty()

    # 8. 居中排版，将最终处理好的丝滑视频完美展出
    # 🌟 关键修改：把 [1, 2, 1] 改成 [1, 1, 1]
    v_spacer_l, v_col, v_spacer_r = st.columns([2, 1, 2])

    with v_col:
        # 安全机制：如果 ffmpeg 转换顺利成功了
        if os.path.exists(web_ready_path) and os.path.getsize(web_ready_path) > 0:
            video_title_placeholder.markdown("<h3 style='text-align: center; color: #4CAF50;'>✅ AI 裁判全片解析渲染完毕！</h3>", unsafe_allow_html=True)
            with open(web_ready_path, "rb") as f:
                video_bytes = f.read()
            # 调用 HTML5 顶级流媒体播放器，彻底解决网络卡顿
            st.video(video_bytes)
            st.balloons()  # 欢呼放气球
            st.success("🎉 奇迹见证！快点击上方播放按钮看看丝滑视频吧！")
        else:
            # 如果转换失败（通常是没安装 ffmpeg 依赖），给用户一个保底播放和提示
            video_title_placeholder.markdown("<h3 style='text-align: center; color: #FF5722;'>⚠️ 视频转码遇到了一点小插曲</h3>", unsafe_allow_html=True)
            with open(out_tfile_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.error("由于你没有在 GitHub 的 packages.txt 里添加 ffmpeg 驱动，导致视频黑屏或无法正常解码。请立即检查 packages.txt 配置文件。")

    # 9. 彻底粉碎删除服务器上产生的所有临时文件，不占内存
    try:
        os.unlink(tfile.name)
        os.unlink(out_tfile_path)
        os.unlink(web_ready_path)
    except:
        pass

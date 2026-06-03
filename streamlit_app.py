import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
import os
import tempfile  # 🌟 新增：用于在服务器后台处理视频流的临时文件

# ⚠️ 必须放在代码最上面：设置页面标题和布局设为宽屏
st.set_page_config(page_title="石头剪刀布识别", layout="wide")


# 1. 加载模型
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    model.model.names = {0: '石头', 1: '剪刀', 2: '布'}
    return model


model = load_model()
gesture_dict = {0: '石头', 1: '剪刀', 2: '布'}


# 2. 读取背景图和生成自定义 CSS
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

# 3. 页面标题区
st.markdown("<h1 style='text-align: center; color: #333;'>✨ ✊✌️✋ 终极石头剪刀布对决 ✨</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>单人挑战 AI，双人开启裁判模式！</h3>", unsafe_allow_html=True)
st.write("---")

# 4. 操作区（🌟 修改：增加了“上传视频文件”的单选按钮选项）
input_mode = st.radio("请选择你的出招方式：", ["📂 上传本地图片", "📷 开启摄像头拍照", "🎬 上传视频文件"], horizontal=True)

img_file_buffer = None
video_file_buffer = None

if input_mode == "📂 上传本地图片":
    img_file_buffer = st.file_uploader("📂 放入比赛截图 (支持 jpg, png)", type=['jpg', 'jpeg', 'png'])
elif input_mode == "📷 开启摄像头拍照":
    img_file_buffer = st.camera_input("📷 点击下方按钮拍摄你的手势")
else:
    # 🌟 新增：视频上传组件
    video_file_buffer = st.file_uploader("🎬 放入视频文件 (支持 mp4, avi, mov)", type=['mp4', 'avi', 'mov'])

# 5. 识别与判断逻辑 
# ==================== 情况一：处理图片或拍照输入 ====================
if img_file_buffer is not None:
    image_pil = Image.open(img_file_buffer)
    img_np = np.array(image_pil)
    
    if img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 左右各加一个空白列 (占位比例: 左空白1, 原图2, AI图2, 右空白1)
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

    # 将 AI 识别结果图片放在右边那一列
    with img_col2:
        st.image(res_image_rgb, caption="AI 识别结果", use_column_width=True)

    # 裁判播报单独占用下面一行，居中显示
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

# ==================== 情况二：处理视频文件输入（🌟 全新新增） ====================
elif video_file_buffer is not None:
    st.write("---")
    st.markdown("<h3 style='text-align: center; color: #333;'>🎬 AI 裁判正在火眼金睛解析视频中...</h3>", unsafe_allow_html=True)
    
    # 1. 建立临时文件，将用户上传的视频字节写入其中，以便 OpenCV 能够用路径读取
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file_buffer.read())
    tfile.close()  # 写入完毕，关闭文件句柄
    
    # 2. 使用 OpenCV 打开这个视频文件
    cap = cv2.VideoCapture(tfile.name)
    
    # 3. 页面排版：左右留白，视频框居中缩小显示
    v_spacer_l, v_col, v_spacer_r = st.columns([1, 2, 1])
    
    with v_col:
        # 创建一个空容器占位符，用来在循环中不断刷新视频帧，形成播放动画
        video_placeholder = st.empty()
        # 创建一个空容器占位符，用来实时刷新当前的裁判播报状态
        status_placeholder = st.empty()
    
    # 4. 逐帧读取并进行 YOLO 实时检测
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 视频读完时自动退出循环
            
        # 运行 YOLO 模型对当前帧进行推理 (verbose=False 可以关闭控制台的大量日志打印，让运行更平滑)
        results = model(frame, conf=0.5, iou=0.4, verbose=False)
        
        # 绘制检测框和标签 (复用你原本的 plot 功能)
        res_frame_bgr = results[0].plot()
        res_frame_rgb = cv2.cvtColor(res_frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 🌟 核心魔法：将动态图片丢进占位符，实现视频播放效果
        video_placeholder.image(res_frame_rgb, use_column_width=True, caption="AI 实时视频追踪处理")
        
        # 5. 在视频下方实时更新“裁判的碎碎念”
        boxes = results[0].boxes
        num_hands = len(boxes)
        
        if num_hands == 0:
            status_placeholder.info("🤔 当前帧：还没抓捕到有效手势，快把手伸出来呀...")
        elif num_hands == 1:
            cls_id = int(boxes.cls[0].item())
            status_placeholder.success(f"🤖 AI 裁判盯着你：画面中只有 1 人，你当前出的是【{gesture_dict[cls_id]}】！")
        elif num_hands == 2:
            # 简单按照画面中的左右水平位置排序
            box1, box2 = boxes[0], boxes[1]
            if box1.xyxy[0][0].item() < box2.xyxy[0][0].item():
                l_cls, r_cls = int(box1.cls[0].item()), int(box2.cls[0].item())
            else:
                l_cls, r_cls = int(box2.cls[0].item()), int(box1.cls[0].item())
            
            status_placeholder.success(f"⚔️ 视频巅峰对决：左边选手【{gesture_dict[l_cls]}】 VS 右边选手【{gesture_dict[r_cls]}】")
        else:
            status_placeholder.warning(f"😱 警告：当前画面检测到了 {num_hands} 只手！裁判眼花了！")
            
    # 5. 视频播放完毕，释放内存资源并清理临时文件
    cap.release()
    try:
        os.unlink(tfile.name)
    except:
        pass
        
    st.balloons()  # 播放完成时，全屏燃放成功气球！
    st.success("🎉 视频分析播放完成！")

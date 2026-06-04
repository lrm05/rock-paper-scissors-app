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
# ==================== 情况二：处理视频文件输入（🌟 完美终结黑屏版） ====================
elif video_file_buffer is not None:
    import time  
    st.write("---")
    
    # 1. 创建动态标题魔术盒
    video_title_placeholder = st.empty()
    video_title_placeholder.markdown("<h3 style='text-align: center; color: #333;'>🎬 AI 裁判正在全速解析并导出视频流...</h3>", unsafe_allow_html=True)
    
    # 2. 建立临时输入文件
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file_buffer.read())
    tfile.close()  
    
    # 3. 使用 OpenCV 打开视频并获取参数
    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 24  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 4. 建立临时输出视频文件
    out_tfile_path = tfile.name + "_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(out_tfile_path, fourcc, fps, (width, height))
    
    # 5. 创建动态进度条看板
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    
    # 6. 后台全速无损渲染
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  
            
        frame_count += 1
        
        results = model(frame, conf=0.5, iou=0.4, verbose=False)
        res_frame_bgr = results[0].plot()
        out.write(res_frame_bgr)
        
        # 实时刷新进度条
        percent = min(int((frame_count / total_frames) * 100), 100)
        progress_bar.progress(percent)
        status_text.markdown(f"<p style='text-align: center; color: #666;'>⚡ 正在精细追踪第 <b>{frame_count}</b> / {total_frames} 帧 ({percent}%)</p>", unsafe_allow_html=True)
            
    cap.release()
    out.release()
    
    # 7. 🌟 核心修复：加上 -pix_fmt yuv420p 彻底治好浏览器的黑屏傲娇病！
    web_ready_path = tfile.name + "_ready.mp4"
    import os
    # 核心注入：-pix_fmt yuv420p 转换为网页通用的标准色彩像素点排列格式
    cmd = f"ffmpeg -y -i {out_tfile_path} -vcodec libx264 -pix_fmt yuv420p {web_ready_path}"
    os.system(cmd)
    
    # 8. 清理进度条组件
    progress_bar.empty()
    status_text.empty()
    
    # 居中显示原生播放器
    v_spacer_l, v_col, v_spacer_r = st.columns([1, 2, 1])
    with v_col:
        # 🌟 增加安全防护机制：确认转换成功的文件存在，再丢给网页播放
        if os.path.exists(web_ready_path) and os.path.getsize(web_ready_path) > 0:
            video_title_placeholder.markdown("<h3 style='text-align: center; color: #4CAF50;'>✅ AI 裁判全片解析渲染完毕！</h3>", unsafe_allow_html=True)
            with open(web_ready_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.balloons()  # 成功时放气球
            st.success("🎉 奇迹见证！快点击上方播放按钮看看吧！")
        else:
            # 如果文件没生成，多半是服务器后台还没把 ffmpeg 依赖完全装好
            video_title_placeholder.markdown("<h3 style='text-align: center; color: #FF5722;'>⚠️ 视频格式转换出现了一点小插曲</h3>", unsafe_allow_html=True)
            st.error("由于你刚刚在 GitHub 的 packages.txt 里增加了 ffmpeg，云端服务器需要 1~2 分钟来下载和部署这个底层工具。请稍等两分钟，然后刷新网页重新上传视频试试看！")
            
    # 9. 彻底清理临时文件
    try:
        os.unlink(tfile.name)
        os.unlink(out_tfile_path)
        os.unlink(web_ready_path)
    except:
        pass

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
import os

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

# 4. 创建左右两列布局
col1, col2 = st.columns(2)

with col1:
    # --- 🌟 本次核心修改：增加输入模式选择 🌟 ---
    input_mode = st.radio("请选择你的出招方式：", ["📂 上传本地图片", "📷 开启摄像头拍照"], horizontal=True)

    img_file_buffer = None
    if input_mode == "📂 上传本地图片":
        img_file_buffer = st.file_uploader("📂 放入比赛截图 (支持 jpg, png)", type=['jpg', 'jpeg', 'png'])
    else:
        # 只需要这一行，Streamlit 就会自动调用电脑摄像头！
        img_file_buffer = st.camera_input("📷 点击下方按钮拍摄你的手势")

# 5. 识别与判断逻辑
# 不管是上传的还是拍的，最后都会变成 img_file_buffer 传给模型
if img_file_buffer is not None:
    image_pil = Image.open(img_file_buffer)
    img_np = np.array(image_pil)

    if img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with col1:
        st.image(image_pil, caption="你的招式", width=400)

    with st.spinner('🤖 AI裁判正在火眼金睛识别中...'):
        # 加上了 conf=0.5 过滤掉桌子角等假阳性目标
        # 加入 iou=0.4 参数，强行清理重叠的多余框
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
                msg = "🖐️ 你出了【布】！那我出【剪刀】... 等等，你赢了！😭"
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

    with col2:
        st.image(res_image_rgb, caption="AI 识别结果", width=400)

    st.subheader("🤖 裁判播报")
    if msg_type == "warning":
        st.warning(msg)
    elif msg_type == "error":
        st.error(msg)
    elif msg_type == "success":
        st.success(msg)
    else:
        st.info(msg)



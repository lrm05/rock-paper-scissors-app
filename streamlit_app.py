# -*- coding: utf-8 -*-
from ultralytics import YOLO 
import cv2
import numpy as np
import os
import streamlit as st
from PIL import Image
import base64
import tempfile
import random 

# ---------------------- 页面配置 ----------------------
# 设置页面配置：标题、宽屏布局
st.set_page_config(page_title="石头剪刀布识别", layout="wide")

# ----------------------模型加载（带缓存）----------------------
# 使用 Streamlit 的缓存装饰器，避免重复加载模型（提升性能）
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    model.model.names = {0: '石头', 1: '剪刀', 2: '布'}
    return model


# 加载模型（只执行一次）
model = load_model()
# 定义手势字典，便于快速获取名称
gesture_dict = {0: '石头', 1: '剪刀', 2: '布'}

# ---------------------- 游戏状态初始化（RPG 对战小游戏） ----------------------
# 用循环一行初始化所有 RPG 变量
# 检查 session_state 中是否存在战斗相关变量，若不存在则赋初始值
for k, v in [('rpg_player_hp', 100), ('rpg_boss_hp', 100), ('rpg_log', "⚔️ 战斗开始！深渊魔王发出咆哮！")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------- 通用工具函数 --------------------------------
def process_image_buffer(buffer):
    image_pil = Image.open(buffer) 
    img_np = np.array(image_pil)
    # 如果图片有透明度通道（RGBA），转换为 RGB（丢弃 alpha）
    if img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    # YOLO 需要 BGR 格式，所以再次转换
    return image_pil, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def ai_referee_judge(boxes):
    num_hands = len(boxes)  # 检测到的手的数量
    if num_hands == 0:
        return "🤔 没看清画面，要不要重拍一张？", "warning"

    if num_hands == 1:  # 单人手势：AI 故意给出相反手势，表示 AI 赢
        # 根据手势类别返回不同消息
        msgs = {
            0: "🪨 你出【石头】！我出【布】，我赢啦！🎉",
            1: "✂️ 你出【剪刀】！我出【石头】，我又赢啦！😎",
            2: "🖐️ 你出【布】！我出【剪刀】，哈哈，我赢啦！🎉"
        }
        cls_id = int(boxes.cls[0].item())
        return msgs.get(cls_id, "未知"), "info"

    if num_hands == 2:  # 双手对战：判断左右双方胜负
        # 根据 x 坐标（左侧 x 较小）确定左右手
        x1 = boxes[0].xyxy[0][0].item()
        x2 = boxes[1].xyxy[0][0].item()
        if x1 < x2:
            l_cls = int(boxes[0].cls[0].item())
            r_cls = int(boxes[1].cls[0].item())
        else:
            l_cls = int(boxes[1].cls[0].item())
            r_cls = int(boxes[0].cls[0].item())

        l_name = gesture_dict.get(l_cls, "未知")
        r_name = gesture_dict.get(r_cls, "未知")

        # 判断胜负规则：石头(0)赢剪刀(1)，剪刀(1)赢布(2)，布(2)赢石头(0)
        if l_cls == r_cls:
            res = "🤝 平局！"
        elif (l_cls == 0 and r_cls == 1) or (l_cls == 1 and r_cls == 2) or (l_cls == 2 and r_cls == 0):
            res = "🏆 【左方】胜利！"
        else:
            res = "🏆 【右方】胜利！"

        return f"⚔️ 左方【{l_name}】 VS 右方【{r_name}】。\n\n{res}", "success"

    # 超过两只手，返回错误提示
    return f"😱 检测到 {num_hands} 只手！一次最多两人对战哦！", "error"


def set_bg_and_css(image_path):
    if not os.path.exists(image_path):
        st.warning(f"⚠️ 找不到背景图片 {image_path}")
        return
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')
    st.markdown(f"""
    <style>
    .stApp {{ background-image: url('data:image/jpeg;base64,{encoded_string}'); background-size: cover; background-position: center bottom; background-attachment: fixed; }}
    .main .block-container {{ background-color: rgba(255, 255, 255, 0.6); padding: 2rem; border-radius: 15px; margin-top: 2rem; margin-bottom: 180px; }}
    </style>
    """, unsafe_allow_html=True)


# 设置背景图片（确保 'bg.jpg' 存在于当前目录）
set_bg_and_css('bg.jpg')

# ----------------------------- 页面布局与输入方式选择 ----------------------------
# 标题
st.markdown("<h1 style='text-align: center; color: #333;'>✨ ✊✌️✋ 终极石头剪刀布对决 ✨</h1>", unsafe_allow_html=True)
st.write("---")

# 选择输入模式：图片/视频上传、摄像头拍照、对战小游戏
input_mode = st.radio("出招方式：", ["📂 上传本地文件 (图片/视频)", "📷 开启摄像头拍照", "对战小游戏"], horizontal=True)
img_file_buffer = None  # 存储图片上传对象
video_file_buffer = None  # 存储视频上传对象

if input_mode == "📂 上传本地文件 (图片/视频)":
    # 允许同时上传图片或视频，根据 MIME 类型区分
    mixed_buffer = st.file_uploader("📂 放入截图或视频", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
    if mixed_buffer:
        if mixed_buffer.type.startswith('image'):
            img_file_buffer = mixed_buffer
        elif mixed_buffer.type.startswith('video'):
            video_file_buffer = mixed_buffer

elif input_mode == "📷 开启摄像头拍照":
    # 调用摄像头拍照组件
    img_file_buffer = st.camera_input("📷 拍摄手势")

# ----------------------------- 图片输入处理 ---------------------------------
if img_file_buffer:
    # 转换图片为 PIL 和 BGR 格式
    image_pil, img_bgr = process_image_buffer(img_file_buffer)
    # 布局：左右两列显示原图和识别结果
    _, col1, col2, _ = st.columns([1, 2, 2, 1])
    with col1:
        st.image(image_pil, caption="你的招式", use_column_width=True)

    with st.spinner('🤖 识别中...'):
        # 调用 YOLO 模型进行检测，置信度阈值 0.5，IoU 阈值 0.4
        results = model(img_bgr, conf=0.5, iou=0.4)
        # 根据检测框生成 AI 解说消息
        msg, msg_type = ai_referee_judge(results[0].boxes)

    with col2:
        # 显示带有检测框和标签的图片（plot() 返回 BGR，需转 RGB）
        st.image(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB), caption="AI 识别结果", use_column_width=True)
    st.write("---")
    # 根据消息类型显示不同样式的消息（info/success/error/warning）
    getattr(st, msg_type)(msg)

# ----------------------------- 视频输入处理 ---------------------------------
elif video_file_buffer:
    st.write("---")
    title_ph = st.empty()  # 占位符用于动态更新标题
    title_ph.markdown("<h3 style='text-align: center;'>🎬 AI 正在解析视频...</h3>", unsafe_allow_html=True)

    # 将上传的视频保存到临时文件
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file_buffer.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)  # 打开视频
    fps = max(24, int(cap.get(5)))  # 帧率，至少24
    width = int(cap.get(3))
    height = int(cap.get(4))
    total = int(cap.get(7))  # 总帧数
    out_path = tfile.name + "_out.mp4"  # 临时输出文件
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    p_bar = st.progress(0)  # 进度条
    s_text = st.empty()  # 显示帧处理文字
    count = 0

    # 逐帧处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        # 对当前帧进行检测，并绘制结果（plot()）
        result_frame = model(frame, conf=0.5, iou=0.4, verbose=False)[0].plot()
        out.write(result_frame)
        pct = min(int((count / total) * 100), 100)
        p_bar.progress(pct)
        s_text.markdown(f"<p style='text-align: center;'>⚡ 渲染第 {count} / {total} 帧 ({pct}%)</p>",
                        unsafe_allow_html=True)

    cap.release()
    out.release()

    # 使用 ffmpeg 重新编码为网页兼容的 h.264 格式（带 faststart 便于流式播放）
    web_ready = tfile.name + "_ready.mp4"
    os.system(
        f"ffmpeg -y -i {out_path} -vcodec libx264 -preset veryfast -pix_fmt yuv420p -movflags +faststart {web_ready}"
    )
    p_bar.empty()
    s_text.empty()

    # 显示最终视频
    _, v_col, _ = st.columns([2, 1, 2])
    with v_col:
        if os.path.exists(web_ready) and os.path.getsize(web_ready) > 0:
            title_ph.markdown("<h3 style='text-align: center; color: #4CAF50;'>✅ 渲染完毕！</h3>",
                              unsafe_allow_html=True)
            # 读取视频文件二进制并嵌入播放器
            st.video(open(web_ready, "rb").read())
            st.balloons()
            st.success("🎉 播放丝滑视频！")
        else:
            title_ph.markdown("<h3 style='text-align: center; color: #FF5722;'>⚠️ 转码失败</h3>",
                              unsafe_allow_html=True)
            # 如果 ffmpeg 不可用，回退播放原始临时视频（可能部分浏览器不支持）
            st.video(open(out_path, "rb").read())
            st.error("由于缺少 ffmpeg，无法解码。")

    # 🌟 极简优化 2：用一行循环强行粉碎 3 个临时文件，消灭原来的 try-except 堆砌
    for f in [tfile.name, out_path, web_ready]:
        try:
            os.unlink(f)  # 删除临时文件
        except:
            pass  # 忽略删除失败（如文件不存在）

# ----------------------------- 对战小游戏（RPG 模式） ---------------------------
elif input_mode == "对战小游戏":
    st.write("---")
    st.markdown("<h2 style='text-align: center; color: #8A2BE2;'>⚡ 人类 vs 怪兽 ⚡</h2>", unsafe_allow_html=True)

    # 居中放置摄像头组件
    _, cam_col, _ = st.columns([1, 1.5, 1])
    with cam_col:
        spell_buffer = st.camera_input("📷 结印（石头/剪刀/布），点击拍照！")

    # 仅在双方都存活且用户拍照后进行处理
    if spell_buffer and st.session_state.rpg_player_hp > 0 and st.session_state.rpg_boss_hp > 0:
        _, img_bgr = process_image_buffer(spell_buffer)
        with st.spinner('🔮 魔法吟唱中...'):
            boxes = model(img_bgr, conf=0.5, iou=0.4, verbose=False)[0].boxes
            if len(boxes) == 1:
                # 随机生成怪物的反击伤害（10/15/20）
                b_dmg = random.choice([10, 15, 20])
                # 🌟 极简优化 3：把 20 多行的连环 if-else 战斗逻辑压缩成一个数据字典映射！
                # 键：手势类别，值：(对BOSS伤害, 对玩家伤害, 日志消息)
                actions = {
                    0: (0, 0, f"🛡️ 【绝对防御】！挡下魔王 {b_dmg} 伤害！"),
                    1: (15, b_dmg, f"⚔️ 【极光斩】！魔王-15血。你被反击 -{b_dmg}血！"),
                    2: (25, b_dmg, f"🔥 【爆裂火球】！魔王-25血。你被反击 -{b_dmg}血！")
                }
                boss_hit, player_hit, log_msg = actions.get(int(boxes.cls[0].item()), (0, 0, "⚠️ 未知魔法！"))

                # 更新血量，不低于0
                st.session_state.rpg_boss_hp = max(0, st.session_state.rpg_boss_hp - boss_hit)
                st.session_state.rpg_player_hp = max(0, st.session_state.rpg_player_hp - player_hit)
                st.session_state.rpg_log = log_msg
            else:
                st.session_state.rpg_log = "⚠️ 魔法失效或暴走（只能单手结印）！"

    # 显示双方血量条和日志
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<h3 style='color: #4CAF50;'>🧙‍♂️ 你的血量: {st.session_state.rpg_player_hp}/100</h3>",
                    unsafe_allow_html=True)
        st.progress(st.session_state.rpg_player_hp / 100.0)
    with c2:
        st.markdown(f"<h3 style='color: #FF5722;'>👹 深渊魔王: {st.session_state.rpg_boss_hp}/100</h3>",
                    unsafe_allow_html=True)
        st.progress(st.session_state.rpg_boss_hp / 100.0)
    st.info(st.session_state.rpg_log)

    # 🌟 极简优化 4：把胜负判断和重置游戏代码揉合（利用高级三元表达式）
    # 如果任意一方血量为0，则结束游戏并显示结果
    if st.session_state.rpg_player_hp == 0 or st.session_state.rpg_boss_hp == 0:
        is_win = st.session_state.rpg_boss_hp == 0  # 魔王血量为0表示玩家胜利
        getattr(st, 'success' if is_win else 'error')("🏆 奇迹发生！击杀魔王！" if is_win else "💀 被魔王击败！")
        if is_win:
            st.balloons()
        # 重置游戏按钮
        if st.button("🔄 再玩一次" if is_win else "🔄 重新复活"):
            # 重新初始化血量与日志
            for k, v in [('rpg_player_hp', 100), ('rpg_boss_hp', 100), ('rpg_log', "⚔️ 战斗重新开始！")]:
                st.session_state[k] = v
            st.rerun()  # 刷新页面，重置状态

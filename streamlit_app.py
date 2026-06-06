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

st.set_page_config(page_title="石头剪刀布识别", layout="wide")

# ==============================================================================
# 1. 初始化模型与记忆变量 (极致压缩版)
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    model.model.names = {0: '石头', 1: '剪刀', 2: '布'}
    return model

model = load_model()
gesture_dict = {0: '石头', 1: '剪刀', 2: '布'}

# 🌟 极简优化 1：用循环一行初始化所有 RPG 变量，消灭 6 行啰嗦的 if
for k, v in [('rpg_player_hp', 100), ('rpg_boss_hp', 100), ('rpg_log', "⚔️ 战斗开始！深渊魔王发出咆哮！")]:
    if k not in st.session_state: st.session_state[k] = v

# ==============================================================================
# 2. 通用工具函数区
def process_image_buffer(buffer):
    image_pil = Image.open(buffer)
    img_np = np.array(image_pil)
    if img_np.shape[-1] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    return image_pil, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def ai_referee_judge(boxes):
    num_hands = len(boxes)
    if num_hands == 0: return "🤔 没看清画面，要不要重拍一张？", "warning"
    if num_hands == 1:
        msgs = {0: "🪨 你出【石头】！我出【布】，我赢啦！🎉", 1: "✂️ 你出【剪刀】！我出【石头】，我又赢啦！😎", 2: "🖐️ 你出【布】！我出【剪刀】，哈哈，我赢啦！🎉"}
        return msgs.get(int(boxes.cls[0].item()), "未知"), "info"
    if num_hands == 2:
        x1, x2 = boxes[0].xyxy[0][0].item(), boxes[1].xyxy[0][0].item()
        l_cls, r_cls = (int(boxes[0].cls[0].item()), int(boxes[1].cls[0].item())) if x1 < x2 else (int(boxes[1].cls[0].item()), int(boxes[0].cls[0].item()))
        l_name, r_name = gesture_dict.get(l_cls, "未知"), gesture_dict.get(r_cls, "未知")
        res = "🤝 平局！" if l_cls == r_cls else ("🏆 【左方】胜利！" if (l_cls==0 and r_cls==1) or (l_cls==1 and r_cls==2) or (l_cls==2 and r_cls==0) else "🏆 【右方】胜利！")
        return f"⚔️ 左方【{l_name}】 VS 右方【{r_name}】。\n\n{res}", "success"
    return f"😱 检测到 {num_hands} 只手！一次最多两人对战哦！", "error"

def set_bg_and_css(image_path):
    if not os.path.exists(image_path): return st.warning(f"⚠️ 找不到背景图片 {image_path}")
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')
    st.markdown(f"""
    <style>
    .stApp {{ background-image: url('data:image/jpeg;base64,{encoded_string}'); background-size: cover; background-position: center bottom; background-attachment: fixed; }}
    .main .block-container {{ background-color: rgba(255, 255, 255, 0.6); padding: 2rem; border-radius: 15px; margin-top: 2rem; margin-bottom: 180px; }}
    </style>
    """, unsafe_allow_html=True)

set_bg_and_css('bg.jpg')

# ==============================================================================
# 3. 页面布局与输入路由
st.markdown("<h1 style='text-align: center; color: #333;'>✨ ✊✌️✋ 终极石头剪刀布对决 ✨</h1>", unsafe_allow_html=True)
st.write("---")

input_mode = st.radio("出招方式：", ["📂 上传本地文件 (图片/视频)", "📷 开启摄像头拍照", "对战小游戏"], horizontal=True)
img_file_buffer, video_file_buffer = None, None

if input_mode == "📂 上传本地文件 (图片/视频)":
    mixed_buffer = st.file_uploader("📂 放入截图或视频", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
    if mixed_buffer:
        if mixed_buffer.type.startswith('image'): img_file_buffer = mixed_buffer
        elif mixed_buffer.type.startswith('video'): video_file_buffer = mixed_buffer
elif input_mode == "📷 开启摄像头拍照":
    img_file_buffer = st.camera_input("📷 拍摄手势")

# ==============================================================================
# 4. 核心逻辑执行区
# ---- 处理图片 ----
if img_file_buffer:
    image_pil, img_bgr = process_image_buffer(img_file_buffer)
    _, col1, col2, _ = st.columns([1, 2, 2, 1])
    with col1: st.image(image_pil, caption="你的招式", use_column_width=True)

    with st.spinner('🤖 识别中...'):
        results = model(img_bgr, conf=0.5, iou=0.4)
        msg, msg_type = ai_referee_judge(results[0].boxes)
        
    with col2: st.image(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB), caption="AI 识别结果", use_column_width=True)
    st.write("---")
    getattr(st, msg_type)(msg)

# ---- 处理视频 ----
elif video_file_buffer:
    st.write("---")
    title_ph = st.empty()
    title_ph.markdown("<h3 style='text-align: center;'>🎬 AI 正在解析视频...</h3>", unsafe_allow_html=True)

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file_buffer.read()); tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps, width, height, total = max(24, int(cap.get(5))), int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    out_path = tfile.name + "_out.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    p_bar, s_text, count = st.progress(0), st.empty(), 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        count += 1
        out.write(model(frame, conf=0.5, iou=0.4, verbose=False)[0].plot())
        pct = min(int((count / total) * 100), 100)
        p_bar.progress(pct); s_text.markdown(f"<p style='text-align: center;'>⚡ 渲染第 {count} / {total} 帧 ({pct}%)</p>", unsafe_allow_html=True)

    cap.release(); out.release()
    web_ready = tfile.name + "_ready.mp4"
    os.system(f"ffmpeg -y -i {out_path} -vcodec libx264 -preset veryfast -pix_fmt yuv420p -movflags +faststart {web_ready}")
    p_bar.empty(); s_text.empty()

    _, v_col, _ = st.columns([2, 1, 2])
    with v_col:
        if os.path.exists(web_ready) and os.path.getsize(web_ready) > 0:
            title_ph.markdown("<h3 style='text-align: center; color: #4CAF50;'>✅ 渲染完毕！</h3>", unsafe_allow_html=True)
            st.video(open(web_ready, "rb").read())
            st.balloons(); st.success("🎉 播放丝滑视频！")
        else:
            title_ph.markdown("<h3 style='text-align: center; color: #FF5722;'>⚠️ 转码失败</h3>", unsafe_allow_html=True)
            st.video(open(out_path, "rb").read())
            st.error("由于缺少 ffmpeg，无法解码。")

    # 🌟 极简优化 2：用一行循环强行粉碎 3 个临时文件，消灭原来的 try-except 堆砌
    for f in [tfile.name, out_path, web_ready]:
        try: os.unlink(f)
        except: pass

# ---- 处理对战小游戏 ----
elif input_mode == "对战小游戏":
    st.write("---")
    st.markdown("<h2 style='text-align: center; color: #8A2BE2;'>⚡ 人类 vs 怪兽 ⚡</h2>", unsafe_allow_html=True)
    
    _, cam_col, _ = st.columns([1, 1.5, 1])
    with cam_col: spell_buffer = st.camera_input("📷 结印（石头/剪刀/布），点击拍照！")

    if spell_buffer and st.session_state.rpg_player_hp > 0 and st.session_state.rpg_boss_hp > 0:
        _, img_bgr = process_image_buffer(spell_buffer)
        with st.spinner('🔮 魔法吟唱中...'):
            boxes = model(img_bgr, conf=0.5, iou=0.4, verbose=False)[0].boxes
            if len(boxes) == 1:
                b_dmg = random.choice([10, 15, 20])
                # 🌟 极简优化 3：把 20 多行的连环 if-else 战斗逻辑，压缩成一个数据字典映射！
                actions = {
                    0: (0, 0, f"🛡️ 【绝对防御】！挡下魔王 {b_dmg} 伤害！"),
                    1: (15, b_dmg, f"⚔️ 【极光斩】！魔王-15血。你被反击 -{b_dmg}血！"),
                    2: (25, b_dmg, f"🔥 【爆裂火球】！魔王-25血。你被反击 -{b_dmg}血！")
                }
                boss_hit, player_hit, log_msg = actions.get(int(boxes.cls[0].item()), (0, 0, "⚠️ 未知魔法！"))
                
                # 瞬间结算血量（限制最低为0）
                st.session_state.rpg_boss_hp = max(0, st.session_state.rpg_boss_hp - boss_hit)
                st.session_state.rpg_player_hp = max(0, st.session_state.rpg_player_hp - player_hit)
                st.session_state.rpg_log = log_msg
            else:
                st.session_state.rpg_log = "⚠️ 魔法失效或暴走（只能单手结印）！"

    # UI 渲染
    c1, c2 = st.columns(2)
    with c1: st.markdown(f"<h3 style='color: #4CAF50;'>🧙‍♂️ 你的血量: {st.session_state.rpg_player_hp}/100</h3>", unsafe_allow_html=True); st.progress(st.session_state.rpg_player_hp / 100.0)
    with c2: st.markdown(f"<h3 style='color: #FF5722;'>👹 深渊魔王: {st.session_state.rpg_boss_hp}/100</h3>", unsafe_allow_html=True); st.progress(st.session_state.rpg_boss_hp / 100.0)
    st.info(st.session_state.rpg_log)

    # 🌟 极简优化 4：把胜负判断和重置游戏代码强行揉合（利用高级三元表达式）
    if st.session_state.rpg_player_hp == 0 or st.session_state.rpg_boss_hp == 0:
        is_win = st.session_state.rpg_boss_hp == 0
        getattr(st, 'success' if is_win else 'error')("🏆 奇迹发生！击杀魔王！" if is_win else "💀 被魔王击败！")
        if is_win: st.balloons()
        if st.button("🔄 再玩一次" if is_win else "🔄 重新复活"):
            for k, v in [('rpg_player_hp', 100), ('rpg_boss_hp', 100), ('rpg_log', "⚔️ 战斗重新开始！")]: st.session_state[k] = v
            st.rerun()

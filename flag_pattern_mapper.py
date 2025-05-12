import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Flag Pattern Mapper", layout="centered")
st.title("🏳️ Flag Pattern Mapper")

st.write("""
Upload a **pattern/texture** image and a **white flag with folds** image. The app will map the pattern onto the flag, simulating realistic cloth folds and lighting.
""")

def segment_flag_area(flag_img):
    mask = np.zeros(flag_img.shape[:2], np.uint8)
    h, w = flag_img.shape[:2]
    rect = (int(0.08*w), int(0.05*h), int(0.85*w), int(0.9*h))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(flag_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    return mask2

def resize_and_pad(pattern, target_shape):
    target_h, target_w = target_shape
    ph, pw = pattern.shape[:2]
    scale = min(target_w / pw, target_h / ph)
    new_w, new_h = int(pw * scale), int(ph * scale)
    pattern_resized = cv2.resize(pattern, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pattern_padded = cv2.copyMakeBorder(pattern_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255,255,255])
    return pattern_padded

def blend_pattern_on_flag(flag, pattern, flag_mask):
    # 1. Find bounding box of flag mask
    ys, xs = np.where(flag_mask)
    min_y, max_y = ys.min(), ys.max()
    min_x, max_x = xs.min(), xs.max()
    box_h, box_w = max_y - min_y, max_x - min_x
    # 2. Resize pattern to fit bounding box, maintaining aspect ratio
    pattern_resized = resize_and_pad(pattern, (box_h, box_w))
    # 3. Place pattern onto flag area
    pattern_on_flag = flag.copy()
    pattern_on_flag[min_y:max_y, min_x:max_x] = pattern_resized
    # 4. Extract shadow map from flag (1 - normalized grayscale)
    flag_gray = cv2.cvtColor(flag, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    shadow_map = 1 - (flag_gray * 0.5 + 0.5)  # range ~[0,1], adjust as needed
    # 5. Apply shadow map to pattern_on_flag (multiply)
    for c in range(3):
        pattern_on_flag[..., c] = pattern_on_flag[..., c] * (1 - shadow_map)
    # 6. Composite onto original image using mask
    final = flag.copy()
    mask3 = np.repeat(flag_mask[:, :, np.newaxis], 3, axis=2)
    final[mask3 == 1] = pattern_on_flag[mask3 == 1]
    return final

with st.expander("Upload Images", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        pattern_file = st.file_uploader("Upload Pattern.jpg", type=["jpg", "jpeg", "png"], key="pattern")
    with col2:
        flag_file = st.file_uploader("Upload Flag.jpg (white flag with folds)", type=["jpg", "jpeg", "png"], key="flag")

if pattern_file and flag_file:
    # Read images
    pattern_img = Image.open(pattern_file).convert("RGB")
    flag_img = Image.open(flag_file).convert("RGB")

    # Show previews
    with st.expander("Preview Uploaded Images", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.image(pattern_img, caption="Pattern.jpg", use_container_width=True)
        with col2:
            st.image(flag_img, caption="Flag.jpg", use_container_width=True)

    # Convert to OpenCV format
    pattern_np = np.array(pattern_img)
    flag_np = np.array(flag_img)

    # --- Advanced Segmentation: GrabCut ---
    with st.spinner("Segmenting flag area (GrabCut)..."):
        flag_mask = segment_flag_area(flag_np)

    # --- Improved Pattern Mapping ---
    result = blend_pattern_on_flag(flag_np, pattern_np, flag_mask)

    # Show output
    st.subheader("Output: Pattern Mapped to Flag (Cloth Only)")
    st.image(result, caption="Output.jpg", use_container_width=True)

    # Download button
    output_pil = Image.fromarray(result)
    buf = BytesIO()
    output_pil.save(buf, format="PNG")
    st.download_button("Download Output Image", data=buf.getvalue(), file_name="Output.png", mime="image/png")
else:
    st.info("Please upload both Pattern.jpg and Flag.jpg to begin.") 
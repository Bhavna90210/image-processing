import cv2
import numpy as np

# Function to load an image from a given path
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img

# Function to enhance shading of the flag image
def enhance_shading(flag_img):
    # Convert flag image to grayscale and normalize
    gray = cv2.cvtColor(flag_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Apply Gaussian blur to smooth the shading
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    # Normalize the blurred image to enhance contrast
    return cv2.normalize(blur, None, 0.6, 1.3, cv2.NORM_MINMAX)

# Function to warp the pattern image using the flag's folds
def warp_pattern(pattern, flag):
    # Get dimensions of the flag image
    h, w = flag.shape[:2]
    # Resize pattern to match flag dimensions
    pattern = cv2.resize(pattern, (w, h))
    # Convert flag to grayscale and apply Gaussian blur for displacement
    disp = cv2.GaussianBlur(cv2.cvtColor(flag, cv2.COLOR_BGR2GRAY), (45, 45), 15)
    # Normalize displacement map
    disp = cv2.normalize(disp.astype(np.float32), None, -10, 10, cv2.NORM_MINMAX)
    # Create meshgrid for pixel coordinates
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    # Apply displacement to coordinates
    map_x += disp / 3.0
    map_y += disp / 3.0
    # Remap pattern using displacement
    return cv2.remap(pattern, map_x, map_y, cv2.INTER_LINEAR)

# Advanced segmentation using GrabCut to isolate the flag cloth area
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

# Helper to resize and pad pattern to fit bounding box, maintaining aspect ratio
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

# Improved blending logic for realistic flag mapping
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
    flag_gray = cv2.cvtColor(flag, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow_map = 1 - (flag_gray * 0.5 + 0.5)  # range ~[0,1], adjust as needed
    # 5. Apply shadow map to pattern_on_flag (multiply)
    for c in range(3):
        pattern_on_flag[..., c] = pattern_on_flag[..., c] * (1 - shadow_map)
    # 6. Composite onto original image using mask
    final = flag.copy()
    mask3 = np.repeat(flag_mask[:, :, np.newaxis], 3, axis=2)
    final[mask3 == 1] = pattern_on_flag[mask3 == 1]
    return final

# Main function to process images and save the output
def main():
    try:
        # Load pattern and flag images
        pattern = load_image("Pattern.jpg")
        flag = load_image("Flag.jpg")
        # Segment the flag area using GrabCut
        flag_mask = segment_flag_area(flag)
        # Blend the pattern onto the flag using improved logic
        result = blend_pattern_on_flag(flag, pattern, flag_mask)
        # Save the result as Output.jpg
        cv2.imwrite("Output.jpg", result)
        print("✅ Output saved as Output.jpg")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
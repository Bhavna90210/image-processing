# Flag Pattern Mapper

## Approach
This project implements a pattern mapping algorithm that realistically maps a texture onto a curved flag surface. The process involves:
1. **Advanced segmentation (OpenCV GrabCut)** to automatically isolate the flag cloth area, excluding background and pole.
2. **Pattern warping** using displacement mapping based on flag folds.
3. **Pattern rescaling** to fit the flag cloth area, preserving the pattern's color and design.
4. **Subtle shading enhancement** to simulate cloth lighting and folds, without distorting the pattern's colors.
5. **Alpha blending** for realistic pattern integration, applied only to the flag area.

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the script:
```bash
python script.py
```

3. For interactive demo:
```bash
streamlit run flag_pattern_mapper.py
```

## Files
- `script.py`: Core image processing logic
- `flag_pattern_mapper.py`: Interactive Streamlit app
- `requirements.txt`: Project dependencies
- `Output.jpg`: Generated output image

## Requirements
- Python 3.7+
- OpenCV
- NumPy
- Streamlit (for interactive demo)

## Notes
- Place your `Pattern.jpg` and `Flag.jpg` in the project directory, or upload them via the Streamlit app.
- The output image will show the pattern mapped only to the flag cloth, preserving the background and pole.
- The pattern's color and design are preserved, with only subtle shading for realism. 
# Furikanji

Adds furigana to kanji in images.

**Input Image:**

![Input](example/sample.png)

**Output Image:**

![Output](example/result_overlay.jpg)

## Setup

1. **Clone the repository:**
   ```bash
   git clone --recurse-submodules https://github.com/tiagopvianna/furikanji.git
   cd furikanji
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

## Usage

```bash
python -m src.furikanji.process_image <image_path> --output_path <output_path>
```

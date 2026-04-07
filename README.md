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

3. **Choose dictionary mode (recommended):**
   - Small install (default quality, fastest setup):
   ```bash
   pip install -e ".[unidic-lite]"
   ```
   - Better reading quality (large one-time download, kept outside this repo):
   ```bash
   pip install -e ".[unidic]"
   python -m unidic download
   ```
   - Sudachi backend (alternative tokenizer + core dictionary):
   ```bash
   pip install -e ".[sudachi]"
   ```
   The UniDic download is stored in your Python environment/cache, not in this git repository.

## Usage

```bash
python -m src.furikanji.main <image_path> --output_path <output_path>
```

Reading backend selection:
```bash
python -m src.furikanji.main <image_path> --reading_backend fugashi --output_path <output_path>
python -m src.furikanji.main <image_path> --reading_backend sudachi --output_path <output_path>
```

# Furikanji

Adds furigana to kanji in images.

## Examples

### Sample 1

| Input | Output |
|---|---|
| ![sample input](example/sample.png) | ![sample output](example/sample_result.png) |

### Sample 2

| Input | Output |
|---|---|
| ![sample 2 input](example/sample_2.png) | ![sample 2 output](example/sample_2_result.png) |

### Full page

| Input | Output |
|---|---|
| ![full page input](example/full_page.jpg) | ![full page output](example/full_page_result.png) |

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

3. **Install Sudachi (default backend):**
   ```bash
   pip install -e ".[sudachi]"
   ```

4. **Optional Fugashi dictionaries (fallback/experimentation):**
   - Small install:
   ```bash
   pip install -e ".[unidic-lite]"
   ```
   - Better UniDic quality (large one-time download, kept outside this repo):
   ```bash
   pip install -e ".[unidic]"
   python -m unidic download
   ```
   The UniDic download is stored in your Python environment/cache, not in this git repository.

## Usage

```bash
python -m src.furikanji.main ./example/full_page.jpg --output_path ./example/full_page_result.png --draw_target_boxes False --draw_overlay_text True --reading_backend sudachi
```

Reading backend:
- `sudachi` is the default and recommended backend.
- You can still select Fugashi explicitly when needed.

Fugashi example:
```bash
python -m src.furikanji.main <image_path> --reading_backend fugashi --output_path <output_path>
```

## Vertical Ruby Fit

Vertical rendering now constrains each furigana token to its own base-token span budget.

Relevant `FuriganaRenderConfig.vertical` options:
- `ruby_fit_policy`: currently `"shrink_then_skip"` (default)
- `ruby_min_size`: minimum per-token ruby font size during fit
- `ruby_min_spacing`: minimum per-token ruby spacing during fit
- `ruby_align`: `"top"` (default) or `"center"`

## Sudachi Reading Overrides

Sudachi overrides are defined in:

`src/furikanji/adapters/sudachi_reading_overrides.json`

Each rule can override a token reading with optional context constraints:

- `kanji`: token surface to match
- `reading`: replacement reading (hiragana recommended)
- `pos_contains`: optional substring match on Sudachi POS fields
- `next_surfaces`: optional allow-list for next token surface
- `prev_surfaces`: optional allow-list for previous token surface
- `exception_prefixes`: skip rule when left context ends with any value
- `exception_suffixes`: skip rule when right context starts with any value

Example:

```json
{
  "rules": [
    {
      "kanji": "私",
      "reading": "わたし",
      "pos_contains": ["代名詞"],
      "next_surfaces": [],
      "prev_surfaces": [],
      "exception_prefixes": [],
      "exception_suffixes": []
    }
  ]
}
```

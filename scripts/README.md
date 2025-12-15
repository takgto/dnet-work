# YOLOv3/YOLOv3-tiny 自動学習スクリプト (Python版)

Darknetを使用したYOLOv3/YOLOv3-tinyの学習を自動化するスクリプトです。

## ファイル構成

```
├── train_yolo.py       # メイン学習スクリプト
├── config.yaml         # 実験設定ファイル
└── README.md           # このファイル
```

## 必要なパッケージ

```bash
pip install pyyaml
```

## 必要なディレクトリ構造

```
workspace/
├── darknet/
│   └── darknet                  # Darknet実行ファイル
├── scripts/
│   ├── train_yolo.py            # メインスクリプト
│   └── config.yaml              # 設定ファイル
├── darknet53.conv.74            # YOLOv3用pre-trained weights
├── darknet19_448.conv.23        # YOLOv3-tiny用pre-trained weights
└── work_dir/                    # --work-dir で指定
    ├── log/                     # ログファイル保存先（自動作成）
    └── work/
        ├── obj.data             # データ設定ファイル
        ├── val.txt              # 検証用ファイルリスト
        ├── train.txt            # 学習用ファイルリスト
        ├── yolov3-416.cfg       # YOLOv3ベースCFG
        ├── yolov3-tiny-416.cfg  # YOLOv3-tinyベースCFG
        └── backup/              # 学習済みweights保存先（自動作成）
```

## 使い方

### 単一実験

```bash
# 基本形（カレントディレクトリのwork/を使用）
python train_yolo.py -r 512 -m yolov3-tiny

# 作業ディレクトリを指定
python train_yolo.py -r 512 -m yolov3-tiny -w /path/to/work

# 最初から学習
python train_yolo.py -r 512 -m yolov3-tiny --clear

# 特定のweightsから再開
python train_yolo.py -r 512 -m yolov3-tiny --resume work/backup/yolov3-tiny-512_10000.weights

# mAP計算のみ
python train_yolo.py -r 512 -m yolov3-tiny --skip-train

# GPUを指定
python train_yolo.py -r 512 -m yolov3-tiny --gpus-train 0,1,2,3 --gpus-map 0,1
```

### バッチ実験

```bash
# config.yamlを使用
python train_yolo.py --batch

# 別の設定ファイルを使用
python train_yolo.py --batch --config my_config.yaml
```

## コマンドラインオプション

| オプション | 短縮形 | 説明 |
|-----------|--------|------|
| `--resolution` | `-r` | 入力解像度（32の倍数） |
| `--model` | `-m` | モデルタイプ（yolov3/yolov3-tiny） |
| `--work-dir` | `-w` | 作業ディレクトリ（work/の位置） |
| `--batch` | - | バッチモードで実行 |
| `--config` | `-c` | 設定ファイルパス |
| `--clear` | - | 最初から学習し直す |
| `--resume` | - | 指定weightsから再開 |
| `--skip-train` | - | 学習スキップ（mAP計算のみ） |
| `--gpus-train` | - | 学習用GPU |
| `--gpus-map` | - | mAP計算用GPU |
| `--docker-image` | - | Dockerイメージ名 |

## config.yaml の設定

```yaml
# 実験リスト
experiments:
  - resolution: 416
    model: yolov3-tiny
    clear: true           # 最初から学習

  - resolution: 512
    model: yolov3-tiny
    clear: true

# GPU設定
gpus:
  train: "4,5,6,7"
  map: "1,2,3,4,5"

# Docker設定
docker:
  image: darknet_compose:ubuntu
  workspace_mount: /workspace
  data_mount: /data2:/data2

# パス設定
# work_dirを指定すれば、以下が自動設定されます：
#   - log_dir:    work_dir/log
#   - backup_dir: work_dir/work/backup
#   - data_file:  work_dir/work/obj.data
#   - val_file:   work_dir/work/val.txt
paths:
  work_dir: work_dir                          # または絶対パス: /path/to/work_dir
  darknet_path: /workspace/darknet/darknet
```

## 出力ファイル

学習完了後、以下のファイルが生成されます：

```
work_dir/
├── log/
│   ├── yolov3-tiny-512_20231201_123456_train.log   # 学習ログ
│   └── yolov3-tiny-512_20231201_123456_results.txt # mAP結果
│   └── batch_20231201_123456/
│       └── summary.txt                              # バッチ実行サマリー
└── work/
    ├── yolov3-tiny-512.cfg                          # 生成されたCFG
    └── backup/
        └── yolov3-tiny-512_last.weights             # 最終weights
```

## 処理フロー

```
┌─────────────────────────────────────────────────────────┐
│  1. 設定読み込み (config.yaml / コマンドライン引数)      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  2. CFGファイル生成                                      │
│     - ベースCFG (work_dir/work/yolov3-416.cfg等) を読込  │
│     - width/height を指定解像度に変更                    │
│     - work_dir/work/yolov3-tiny-512.cfg 等を生成        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  3. Docker実行スクリプト生成                             │
│     - 学習コマンド / mAP計算コマンドを含むシェルスクリプト │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  4. Docker起動 → 学習実行 → mAP計算                     │
│     - darknet_compose:ubuntu イメージを使用             │
│     - 適切なpre-trained weightsを使用                   │
│       (yolov3: darknet53.conv.74)                      │
│       (yolov3-tiny: darknet19_448.conv.23)             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  5. ログ・結果保存                                       │
│     - work_dir/log/ に学習ログとmAP結果を保存            │
│     - work_dir/work/backup/ に学習済みweightsを保存     │
└─────────────────────────────────────────────────────────┘
```

## 注意事項

- **解像度は32の倍数**である必要があります（例: 416, 448, 512, 608）
- CFGファイルのwidth/heightは自動的に指定解像度に変更されます
- 学習中断後に再開する場合は、`--clear`オプションを**付けない**でください
- mAP計算には`work/val.txt`が必要です

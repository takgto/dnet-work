# dnet-work
python scripts and configurable files for executing darknet training on nedo server.

## 作業手順（サーバーでのDarkent学習方法）
1. windows power shellからssh loginする。
   ```bash
   ssh guest@<ip address>:/home/guest
   ```
2. 自分の作業ディレクトリを/data2/goto_data/darknet/dnet-worの下に作る。
   ```bash
   cp -r /data2/goto_data/darknet/dnet-work/work_dir /data2/goto_data/darknet/dnet-work/{自分の作業ディレクトリ名、(your name)_dirとか}
   ```
3. 学習をdnet-workの下で実行する。
   まず、
   ```bash
   cd /data2/goto_data/darknet
   ```
  ### 作業ディレクトリを必ず指定、ここではwork_dirとする。-rで解像度を指定できる、gpu番号は0, mAP計算用gpuも0、yolov3のタイプはtiny, -tinyを付けなければ通常のyolov3　
  ```bash
  python ./dnet-work/scripts/train_yolo.py -r 512 -m yolov3-tiny -w ./work_dir --gpus-train 0 --gpus-map 0
  ```
  ### 最初から学習させる場合
  ```bash
  python ./dnet-work/scripts/train_yolo.py -r 512 -m yolov3-tiny -w ./work_dir --clear
  ```
  ### 特定のweightsから再開させる
  ```bash
  python ./dnet-work/scripts/train_yolo.py -r 512 -m yolov3-tiny -w ./work_dir --resume work/backup/yolov3-tiny-512_10000.weights
  ```
  ### mAP計算のみ実行
  ```bash
  python ./dnet-work/scripts/train_yolo.py -r 512 -m yolov3-tiny -w ./work_dir --skip-train
  ```
  ### GPUを指定する
  ```bash
  python ./dnet-work/scripts/train_yolo.py -r 512 -m yolov3-tiny -w ./work_dir --gpus-train 0,1,2,3 --gpus-map 0,1
  ```
### コマンドラインオプション
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

これらオプションのデフォルト値は、次のconfig.yamlに記述されている。
```bash
/data2/goto_data/darknet/dnet-work/scripts/config.yaml
```
コマンドラインオプションで指定しなくともこのファイルの記述を変えればオプション値は変更可能である。

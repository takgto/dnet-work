#!/usr/bin/env python3
"""
YOLOv3 / YOLOv3-tiny 自動学習スクリプト

Usage:
    # 単一実験
    python train_yolo.py --resolution 512 --model yolov3-tiny
    
    # バッチ実験（config.yamlを使用）
    python train_yolo.py --batch
    
    # mAP計算のみ
    python train_yolo.py --resolution 512 --model yolov3-tiny --skip-train

    # 独自CFGで学習（バックボーンはyolov3と同じ → darknet53.conv.74を使用）
    python train_yolo.py -r 512 --cfg my-yolov3-head.cfg

    # 独自CFGで学習（バックボーンを変更 → 事前学習重みの先頭N層だけ使用）
    python train_yolo.py -r 512 --cfg my-yolov3-small.cfg --partial 36

    # 独自CFGをスクラッチで学習
    python train_yolo.py -r 512 --cfg my-net.cfg --pretrained none
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

try:
    import yaml
except ImportError:
    print("PyYAMLが必要です: pip install pyyaml")
    sys.exit(1)


# =============================================================================
# 定数
# =============================================================================

NUM_CLASSES = 3  # クラス数（固定）


# =============================================================================
# 設定クラス
# =============================================================================

@dataclass
class ModelConfig:
    """モデル固有の設定"""
    name: str
    base_cfg: str
    pretrained_weights: str


@dataclass
class Experiment:
    """実験設定"""
    resolution: int
    model: str
    clear: bool = False
    resume_weights: Optional[str] = None
    skip_train: bool = False
    cfg: Optional[str] = None          # 独自ベースCFG（指定時はmodelの代わりに使用）
    pretrained: Optional[str] = None   # 事前学習重みの上書き（"none"でスクラッチ学習）
    partial: Optional[int] = None      # 事前学習重みの先頭N層だけを使う（darknet partial）
    no_weight_check: bool = False      # 初期重みと cfg の対応検査を省く（旧条件の再現用）


@dataclass
class GPUConfig:
    """GPU設定"""
    train: str = "4,5,6,7"
    map: str = "1,2,3,4,5"


@dataclass
class DockerConfig:
    """Docker設定"""
    image: str = "darknet_compose:ubuntu"
    workspace_mount: str = "/workspace"
    data_mount: str = "/data2:/data2"


@dataclass
class PathConfig:
    """パス設定"""
    work_dir: Path = Path("work_dir")
    log_dir: Path = None
    backup_dir: Path = None
    work_subdir: Path = None  # work_dir/work
    data_file: str = None
    val_file: str = None
    train_file: str = None
    names_file: str = None
    darknet_path: str = "/workspace/darknet/darknet"
    
    def __post_init__(self):
        """work_dirを基準に他のパスを自動設定"""
        self.work_dir = Path(self.work_dir)
        self.work_subdir = self.work_dir / "work"
        if self.log_dir is None:
            self.log_dir = self.work_dir / "log"
        if self.backup_dir is None:
            self.backup_dir = self.work_subdir / "backup"
        if self.data_file is None:
            self.data_file = str(self.work_subdir / "obj.data")
        if self.val_file is None:
            self.val_file = str(self.work_subdir / "val.txt")
        if self.train_file is None:
            self.train_file = str(self.work_subdir / "train.txt")
        if self.names_file is None:
            self.names_file = str(self.work_subdir / "obj.names")


# モデル定義
MODELS = {
    "yolov3": ModelConfig(
        name="yolov3",
        base_cfg="yolov3-416.cfg",
        pretrained_weights="/workspace/darknet53.conv.74"
    ),
    "yolov3-tiny": ModelConfig(
        name="yolov3-tiny",
        base_cfg="yolov3-tiny-416.cfg",
        # yolov3-tiny.weights（COCO 学習済み）の先頭15層（層0〜14）。作り方:
        #   ./darknet partial cfg/yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.15 15
        # 以前の darknet19_448.conv.23 は YOLOv2 の Darknet-19 用で、tiny とは層構成が違うため
        # 全層がずれた値で初期化されていた（tools/check_init_weights.py で確認可能）
        pretrained_weights="/workspace/yolov3-tiny.conv.15"
    )
}


# =============================================================================
# ユーティリティ関数
# =============================================================================

def validate_resolution(resolution: int) -> bool:
    """解像度が32の倍数かチェック"""
    if resolution % 32 != 0:
        print(f"Error: 解像度は32の倍数である必要があります (入力: {resolution})")
        return False
    if resolution < 32 or resolution > 2048:
        print(f"Error: 解像度は32〜2048の範囲で指定してください (入力: {resolution})")
        return False
    return True


def validate_model(model: str) -> bool:
    """モデルタイプをチェック"""
    if model not in MODELS:
        print(f"Error: モデルは {list(MODELS.keys())} のいずれかを指定してください")
        return False
    return True


def generate_cfg(
    base_cfg_path: Path,
    output_cfg_path: Path,
    resolution: int
) -> bool:
    """CFGファイルを生成（width/heightを変更）"""
    
    if not base_cfg_path.exists():
        print(f"Error: ベースCFGファイルが見つかりません: {base_cfg_path}")
        return False
    
    content = base_cfg_path.read_text()
    
    # width/heightを置換
    content = re.sub(r'^width=\d+', f'width={resolution}', content, flags=re.MULTILINE)
    content = re.sub(r'^height=\d+', f'height={resolution}', content, flags=re.MULTILINE)
    
    output_cfg_path.write_text(content)
    print(f"Generated CFG: {output_cfg_path}")
    
    return True


def resolve_cfg_path(cfg: str, work_subdir: Path) -> Optional[Path]:
    """--cfgのパスを解決（カレントディレクトリ基準 → work_dir/work/ の順に探す）"""
    candidates = [Path(cfg), work_subdir / cfg]
    for c in candidates:
        if c.exists():
            return c
    print(f"Error: CFGファイルが見つかりません: {cfg}")
    print(f"  探した場所: {', '.join(str(c) for c in candidates)}")
    return None


def cfg_model_name(cfg_path: Path) -> str:
    """CFGファイル名から実験名を作る（末尾の解像度 -416 などは除く）
    例: yolov3-head-416.cfg → yolov3-head"""
    return re.sub(r'-\d+$', '', cfg_path.stem)


def conv_float_boundaries(cfg_path: Path) -> list:
    """cfg の conv 層ごとに、そこまでの累積 float 数を [(層番号, 累積数)] で返す。
    Darknet の .weights は conv 層の biases / BN(scales, mean, var) / weights を
    この順に並べただけなので、正しい初期重みの float 数はどこかの層の区切りに一致する。"""
    sections = []
    for raw in cfg_path.read_text().splitlines():
        line = raw.split('#')[0].strip()
        if line.startswith('[') and line.endswith(']'):
            sections.append((line[1:-1].strip(), {}))
        elif '=' in line and sections:
            k, v = line.split('=', 1)
            sections[-1][1][k.strip()] = v.strip()
    net = sections[0][1] if sections and sections[0][0] in ('net', 'network') else {}
    layers = [s for s in sections if s[0] not in ('net', 'network')]
    c = int(net.get('channels', 3))
    out_c, bounds, total = [], [(0, 0)], 0
    for i, (t, kv) in enumerate(layers):
        if t == 'convolutional':
            f, k = int(kv['filters']), int(kv['size'])
            g = int(kv.get('groups', 1))
            bn = int(kv.get('batch_normalize', 0))
            total += f * (4 if bn else 1) + f * (c // g) * k * k
            c = f
            bounds.append((i + 1, total))
        elif t == 'route':
            src = [int(x) for x in kv['layers'].split(',')]
            src = [i + x if x < 0 else x for x in src]
            c = sum(out_c[j] for j in src)
            if 'groups' in kv:
                c //= int(kv['groups'])
        out_c.append(c)
    return bounds


def weights_float_count(path: Path) -> int:
    """.weights のヘッダー（major, minor, revision, seen）を除いた float 数"""
    import struct
    with open(path, 'rb') as f:
        major, minor, _ = struct.unpack('3i', f.read(12))
    hdr = 20 if (major * 10 + minor >= 2 and major < 1000 and minor < 1000) else 16
    return (path.stat().st_size - hdr) // 4


def check_pretrained_fit(cfg_path: Path, weights_path: Path, partial: Optional[int]) -> bool:
    """初期重みが cfg に対応しているかを float 数で検査する。
    別ネットワーク用の重みを渡しても Darknet はエラーにせず、ずれた値で初期化するため。"""
    if not weights_path.exists():
        print(f"Warning: {weights_path} がホスト側に見つからないため、初期重みの検査をスキップ")
        return True
    n = weights_float_count(weights_path)
    bounds = conv_float_boundaries(cfg_path)
    if partial is not None:
        need = max(b for l, b in bounds if l <= partial)
        if n < need:
            print(f"Error: {weights_path} の float 数 {n:,} が、先頭{partial}層に必要な {need:,} より少ない")
            return False
        print(f"初期重みの検査: float 数 {n:,}（先頭{partial}層で {need:,} を使用）"
              "※ partial は先頭層の構成が元の重みと同じであることが前提")
        return True
    hit = [l for l, b in bounds if b == n]
    if hit:
        print(f"初期重みの検査: OK（float 数 {n:,} が cfg の先頭{hit[-1]}層の区切りと一致）")
        return True
    print(f"Error: 初期重み {weights_path}（float 数 {n:,}）が cfg {cfg_path.name} のどの層の区切りとも一致しません。")
    print(f"       cfg の conv 全体の float 数は {bounds[-1][1]:,}。別のネットワーク用の重みの可能性が高いです。")
    print("       （例: yolov3-tiny に darknet19_448.conv.23 を渡すと全層がずれた値で初期化される）")
    print("       意図した重みなら --partial N で先頭N層だけを使ってください。")
    return False


def validate_cfg(cfg_path: Path) -> bool:
    """[yolo]層のclasses / mask / anchorsと、直前の[convolutional]のfiltersの整合性をチェック"""

    sections = []  # [(セクション名, {key: value}, 開始行番号)]
    for lineno, raw in enumerate(cfg_path.read_text().splitlines(), 1):
        line = raw.split('#')[0].strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith(']'):
            sections.append((line[1:-1].strip(), {}, lineno))
        elif '=' in line and sections:
            key, value = line.split('=', 1)
            sections[-1][1][key.strip()] = value.strip()

    errors = []
    yolo_count = 0
    for i, (name, opts, lineno) in enumerate(sections):
        if name != 'yolo':
            continue
        yolo_count += 1
        where = f"[yolo] ({lineno}行目)"

        classes = int(opts.get('classes', -1))
        if classes != NUM_CLASSES:
            errors.append(f"{where}: classes={classes} → classes={NUM_CLASSES} にしてください")

        masks = [int(m) for m in opts.get('mask', '').split(',') if m.strip()]
        num = int(opts.get('num', 0))
        anchors = [a for a in opts.get('anchors', '').split(',') if a.strip()]
        if len(anchors) != num * 2:
            errors.append(f"{where}: anchorsの数({len(anchors)})が num={num} の2倍になっていません")
        if any(m >= num for m in masks):
            errors.append(f"{where}: mask={masks} に num={num} 以上の値があります")

        # 直前の[convolutional]のfilters
        prev = next((s for s in reversed(sections[:i]) if s[0] in ('convolutional', 'conv')), None)
        expected = (NUM_CLASSES + 5) * len(masks)
        if prev is None:
            errors.append(f"{where}: 直前に[convolutional]がありません")
        elif int(prev[1].get('filters', -1)) != expected:
            errors.append(
                f"{where}: 直前の[convolutional] ({prev[2]}行目) のfilters={prev[1].get('filters')}"
                f" → ({NUM_CLASSES}+5)×{len(masks)} = {expected} にしてください"
            )

    if yolo_count == 0:
        errors.append("[yolo]層が1つもありません")

    net = next((s for s in sections if s[0] in ('net', 'network')), None)
    if net is None or 'width' not in net[1] or 'height' not in net[1]:
        errors.append("[net]にwidth/heightがありません（行頭に width=, height= を書いてください）")

    if errors:
        print(f"Error: CFGファイルに問題があります: {cfg_path}")
        for e in errors:
            print(f"  - {e}")
        return False

    print(f"CFG check OK: [yolo]層 {yolo_count}個, classes={NUM_CLASSES}")
    return True


def generate_obj_data(path_config: PathConfig) -> bool:
    """obj.dataファイルを自動生成（Docker内の絶対パスを使用）"""
    
    ws = "/workspace"
    obj_data_content = f"""classes = {NUM_CLASSES}
train = {ws}/{path_config.train_file}
valid = {ws}/{path_config.val_file}
names = {ws}/{path_config.names_file}
backup = {ws}/{path_config.backup_dir}
"""
    
    obj_data_path = Path(path_config.data_file)
    obj_data_path.write_text(obj_data_content)
    print(f"Generated obj.data: {obj_data_path}")
    
    return True


def generate_docker_script(
    script_path: Path,
    train_cmd: str,
    map_cmd: str,
    train_log: str,
    results_file: str,
    val_file: str,
    final_weights: str,
    log_dir: str,
    work_dir: str,
    config_name: str,
    skip_train: bool = False,
    partial_cmd: Optional[str] = None
) -> None:
    """Docker内で実行するスクリプトを生成"""
    
    script_content = f'''#!/bin/bash
set -e

# 実行ディレクトリをwork_dirに変更（チャートファイルがここに出力される）
cd /workspace/{work_dir}

SCRIPT_START=$(date +%s)
echo "=========================================="
echo "Training started at: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

'''
    
    if partial_cmd and not skip_train:
        script_content += f'''
# 事前学習重みの先頭N層を切り出し（darknet partial）
echo "[Partial Command]"
echo "{partial_cmd}"
{partial_cmd}
echo ""
'''

    if not skip_train:
        script_content += f'''
# 学習実行
TRAIN_START=$(date +%s)

echo ""
echo "[Training Command]"
echo "{train_cmd}"
echo ""
{train_cmd} 2>&1 | tee /workspace/{train_log}

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$((TRAIN_END - TRAIN_START))
TRAIN_HOURS=$((TRAIN_ELAPSED / 3600))
TRAIN_MINUTES=$(((TRAIN_ELAPSED % 3600) / 60))
TRAIN_SECONDS=$((TRAIN_ELAPSED % 60))

echo ""
echo "Training completed at: $(date)"
echo "Training time: ${{TRAIN_HOURS}}h ${{TRAIN_MINUTES}}m ${{TRAIN_SECONDS}}s (${{TRAIN_ELAPSED}} seconds)"
echo ""

# チャートファイルをログディレクトリにコピー
if [ -f "/workspace/{work_dir}/chart.png" ]; then
    cp /workspace/{work_dir}/chart.png /workspace/{log_dir}/
    echo "Copied chart.png to {log_dir}/"
fi
if [ -f "/workspace/{work_dir}/chart_{config_name}.png" ]; then
    cp /workspace/{work_dir}/chart_{config_name}.png /workspace/{log_dir}/
    echo "Copied chart_{config_name}.png to {log_dir}/"
fi

'''
    
    script_content += f'''
# mAP計算
echo "=========================================="
echo "mAP Calculation started at: $(date)"
echo "=========================================="

if [ -f "/workspace/{final_weights}" ]; then
    echo "[mAP Command]"
    echo "{map_cmd}"
    {map_cmd} < /workspace/{val_file} 2>&1 | tee /workspace/{results_file}
    echo ""
    echo "mAP calculation completed at: $(date)"
else
    echo "Error: Weights file not found: {final_weights}"
    exit 1
fi

SCRIPT_END=$(date +%s)
TOTAL_ELAPSED=$((SCRIPT_END - SCRIPT_START))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

echo ""
echo "=========================================="
echo "All tasks completed at: $(date)"
echo "Total time: ${{TOTAL_HOURS}}h ${{TOTAL_MINUTES}}m ${{TOTAL_SECONDS}}s (${{TOTAL_ELAPSED}} seconds)"
echo "=========================================="
'''
    
    script_path.write_text(script_content)
    script_path.chmod(0o755)


# =============================================================================
# メイン処理クラス
# =============================================================================

class YOLOTrainer:
    """YOLO学習管理クラス"""
    
    def __init__(
        self,
        gpu_config: GPUConfig,
        docker_config: DockerConfig,
        path_config: PathConfig
    ):
        self.gpu = gpu_config
        self.docker = docker_config
        self.paths = path_config
        
        # ディレクトリ作成
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.paths.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # obj.data自動生成
        generate_obj_data(self.paths)
    
    def run_experiment(self, exp: Experiment) -> bool:
        """単一の実験を実行"""
        
        # バリデーション
        if not validate_resolution(exp.resolution):
            return False
        if not validate_model(exp.model):
            return False

        model_config = MODELS[exp.model]
        ws = "/workspace"

        def to_container_path(p: str) -> str:
            """Docker内のパスに変換（絶対パスはそのまま、相対パスは/workspace基準）"""
            return p if p.startswith("/") else f"{ws}/{p}"

        # ベースCFGと実験名の決定（--cfg指定時は独自CFG、それ以外は-mのモデル）
        if exp.cfg:
            base_cfg_path = resolve_cfg_path(exp.cfg, self.paths.work_subdir)
            if base_cfg_path is None:
                return False
            name = cfg_model_name(base_cfg_path)
        else:
            base_cfg_path = self.paths.work_subdir / model_config.base_cfg
            name = exp.model

        # 事前学習重みの決定（--pretrained指定時は上書き、"none"でスクラッチ）
        pretrained = exp.pretrained if exp.pretrained else model_config.pretrained_weights
        if pretrained.lower() == "none":
            pretrained = None
        if exp.partial is not None and pretrained is None:
            print("Error: --partial を使うには事前学習重みが必要です（--pretrained none とは併用不可）")
            return False

        config_name = f"{name}-{exp.resolution}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{config_name}_{timestamp}"

        print("=" * 50)
        print("YOLOv3 Training Script")
        print("=" * 50)
        print(f"Experiment:   {experiment_name}")
        print(f"Base CFG:     {base_cfg_path}")
        print(f"Resolution:   {exp.resolution}x{exp.resolution}")
        print(f"Classes:      {NUM_CLASSES}")
        pre_desc = pretrained if pretrained else "なし（スクラッチ学習）"
        if exp.partial is not None:
            pre_desc += f"（先頭{exp.partial}層のみ）"
        print(f"Pre-trained:  {pre_desc}")
        print(f"Train GPUs:   {self.gpu.train}")
        print(f"mAP GPUs:     {self.gpu.map}")
        print(f"Clear:        {exp.clear}")
        print(f"Skip Train:   {exp.skip_train}")
        print("=" * 50)

        # Step 1: CFGファイル生成
        print("\n[Step 1] CFGファイルの生成...")
        output_cfg = self.paths.work_subdir / f"{config_name}.cfg"

        if not generate_cfg(base_cfg_path, output_cfg, exp.resolution):
            return False
        if exp.cfg and not validate_cfg(output_cfg):
            return False

        abs_data_file = f"{ws}/{self.paths.data_file}"
        abs_output_cfg = f"{ws}/{output_cfg}"

        # Step 2: Weightsファイルの決定
        print("\n[Step 2] Weightsファイルの決定...")
        final_weights = self.paths.backup_dir / f"{config_name}_last.weights"
        partial_cmd = None

        if exp.resume_weights:
            start_weights = to_container_path(exp.resume_weights)
            print(f"Resuming from: {start_weights}")
        elif final_weights.exists() and not exp.clear:
            start_weights = to_container_path(str(final_weights))
            print(f"Resuming from: {start_weights}")
        elif pretrained is None:
            start_weights = None
            print("Starting from: スクラッチ（事前学習重みなし）")
        elif exp.partial is not None:
            # 事前学習重みの先頭N層だけを切り出してから学習する
            partial_weights = self.paths.backup_dir / f"{name}.conv.{exp.partial}"
            start_weights = to_container_path(str(partial_weights))
            partial_cmd = (
                f"{self.paths.darknet_path} partial "
                f"{abs_output_cfg} {to_container_path(pretrained)} {start_weights} {exp.partial}"
            )
            print(f"Starting from: {start_weights}（{pretrained} の先頭{exp.partial}層）")
        else:
            start_weights = to_container_path(pretrained)
            print(f"Starting from: {start_weights}")

        # 事前学習重みが cfg に合っているかを検査（再開時は対象外）
        if start_weights and not exp.resume_weights and not (final_weights.exists() and not exp.clear):
            host_pre = Path(pretrained)
            if str(host_pre).startswith(ws + "/"):
                host_pre = Path.cwd() / str(host_pre)[len(ws) + 1:]
            if exp.no_weight_check:
                print("Warning: --no-weight-check 指定のため初期重みの検査を省略（cfg に合わない重みでも学習します）")
            elif not check_pretrained_fit(output_cfg, host_pre, exp.partial):
                return False

        # Step 3: コマンド構築（Docker内の絶対パスを使用）
        clear_flag = "-clear" if exp.clear else ""
        abs_final_weights = f"{ws}/{final_weights}"

        # チャートは実行ディレクトリ(/workspace)に自動出力され、学習後にlog_dirにコピー
        train_cmd = (
            f"{self.paths.darknet_path} detector train "
            f"{abs_data_file} {abs_output_cfg} {start_weights or ''} "
            f"-dont_show -gpus {self.gpu.train} {clear_flag}"
        ).strip()
        
        map_cmd = (
            f"{self.paths.darknet_path} detector map "
            f"{abs_data_file} {abs_output_cfg} {abs_final_weights} "
            f"-dont_show -ext_output -gpus {self.gpu.map}"
        )
        
        # Step 4: Docker実行スクリプト生成
        print("\n[Step 3] Docker実行スクリプトの生成...")
        train_log = f"{self.paths.log_dir}/{experiment_name}_train.log"
        results_file = f"{self.paths.log_dir}/{experiment_name}_results.txt"
        docker_script = self.paths.work_dir / f"run_{experiment_name}.sh"
        
        generate_docker_script(
            script_path=docker_script,
            train_cmd=train_cmd,
            map_cmd=map_cmd,
            train_log=train_log,
            results_file=results_file,
            val_file=self.paths.val_file,
            final_weights=str(final_weights),
            log_dir=str(self.paths.log_dir),
            work_dir=str(self.paths.work_dir),
            config_name=config_name,
            skip_train=exp.skip_train,
            partial_cmd=partial_cmd
        )
        
        # Step 5: Docker実行
        print("\n[Step 4] Docker実行...")
        docker_cmd = [
            "docker", "run",
            "--rm",  # コンテナ終了時に自動削除
            "-v", f"{Path.cwd()}:{self.docker.workspace_mount}",
            "-v", self.docker.data_mount,
            "-w", self.docker.workspace_mount,
            "--gpus", "all",
            "-it", self.docker.image,
            "/bin/bash", f"{self.docker.workspace_mount}/{docker_script}"
        ]
        
        print(f"Command: {' '.join(docker_cmd)}\n")
        
        try:
            result = subprocess.run(docker_cmd, check=True)
            success = result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"Error: Docker実行に失敗しました: {e}")
            success = False
        except KeyboardInterrupt:
            print("\n中断されました")
            success = False
        
        # 完了メッセージ
        print("\n" + "=" * 50)
        print("実行完了" if success else "実行失敗")
        print("=" * 50)
        print(f"Train Log:     {train_log}")
        print(f"mAP Results:   {results_file}")
        print(f"Final Weights: {final_weights}")
        print("=" * 50)
        
        return success
    
    def run_batch(self, experiments: list[Experiment]) -> dict:
        """複数の実験を一括実行"""
        
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_log_dir = self.paths.log_dir / f"batch_{batch_timestamp}"
        batch_log_dir.mkdir(parents=True, exist_ok=True)
        
        results = {"success": [], "failed": []}
        
        print("=" * 50)
        print("Batch Training Script")
        print("=" * 50)
        print(f"Total experiments: {len(experiments)}")
        print(f"Log directory: {batch_log_dir}")
        print("=" * 50)
        
        for i, exp in enumerate(experiments, 1):
            label = f"{exp.cfg or exp.model}-{exp.resolution}"
            print(f"\n[{i}/{len(experiments)}] {exp.cfg or exp.model} @ {exp.resolution}x{exp.resolution}")

            if self.run_experiment(exp):
                results["success"].append(label)
            else:
                results["failed"].append(label)
        
        # サマリー出力
        summary_file = batch_log_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Batch Training Summary - {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Success:\n")
            for s in results["success"]:
                f.write(f"  - {s}\n")
            f.write("\nFailed:\n")
            for s in results["failed"]:
                f.write(f"  - {s}\n")
        
        print("\n" + "=" * 50)
        print("Batch Training Complete")
        print("=" * 50)
        print(f"Success: {len(results['success'])}")
        print(f"Failed:  {len(results['failed'])}")
        print(f"Summary: {summary_file}")
        print("=" * 50)
        
        return results


# =============================================================================
# 設定ファイル読み込み
# =============================================================================

def load_config(config_path: Path) -> dict:
    """YAML設定ファイルを読み込み"""
    
    if not config_path.exists():
        print(f"Error: 設定ファイルが見つかりません: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_config(config: dict) -> tuple[list[Experiment], GPUConfig, DockerConfig, PathConfig]:
    """設定ファイルをパース"""
    
    # GPU設定
    gpu_cfg = config.get("gpus", {})
    gpu_config = GPUConfig(
        train=gpu_cfg.get("train", "4,5,6,7"),
        map=gpu_cfg.get("map", "1,2,3,4,5")
    )
    
    # Docker設定
    docker_cfg = config.get("docker", {})
    docker_config = DockerConfig(
        image=docker_cfg.get("image", "darknet_compose:ubuntu"),
        workspace_mount=docker_cfg.get("workspace_mount", "/workspace"),
        data_mount=docker_cfg.get("data_mount", "/data2:/data2")
    )
    
    # パス設定（work_dirを基準に自動設定）
    paths_cfg = config.get("paths", {})
    work_dir = Path(paths_cfg.get("work_dir", "work_dir"))
    path_config = PathConfig(
        work_dir=work_dir,
        darknet_path=paths_cfg.get("darknet_path", "/workspace/darknet/darknet")
    )
    
    # 実験設定
    experiments = []
    for exp_cfg in config.get("experiments", []):
        experiments.append(Experiment(
            resolution=exp_cfg["resolution"],
            model=exp_cfg.get("model", "yolov3"),
            clear=exp_cfg.get("clear", False),
            resume_weights=exp_cfg.get("resume_weights"),
            skip_train=exp_cfg.get("skip_train", False),
            cfg=exp_cfg.get("cfg"),
            pretrained=exp_cfg.get("pretrained"),
            partial=exp_cfg.get("partial"),
            no_weight_check=exp_cfg.get("no_weight_check", False)
        ))
    
    return experiments, gpu_config, docker_config, path_config


# =============================================================================
# メイン
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv3/YOLOv3-tiny 自動学習スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 単一実験
  python train_yolo.py -r 512 -m yolov3-tiny
  
  # 作業ディレクトリを指定
  python train_yolo.py -r 512 -m yolov3-tiny -w /path/to/work
  
  # 最初から学習
  python train_yolo.py -r 512 -m yolov3-tiny --clear
  
  # バッチ実験
  python train_yolo.py --batch --config config.yaml
  
  # mAP計算のみ
  python train_yolo.py -r 512 -m yolov3-tiny --skip-train
        """
    )
    
    # 実行モード
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--batch", action="store_true",
        help="バッチモードで実行（config.yamlを使用）"
    )
    
    # 単一実験用オプション
    parser.add_argument(
        "--resolution", "-r", type=int,
        help="入力解像度（32の倍数）"
    )
    parser.add_argument(
        "--model", "-m", type=str, choices=["yolov3", "yolov3-tiny"],
        help="モデルタイプ"
    )
    parser.add_argument(
        "--cfg", type=str, metavar="CFG",
        help="独自のベースCFGファイル（カレントディレクトリ基準、無ければwork_dir/work/から探す）。"
             "指定時は-mの代わりにこのCFGを使い、出力名はCFGファイル名から決まる"
    )
    parser.add_argument(
        "--pretrained", type=str, metavar="WEIGHTS",
        help="事前学習重み（デフォルトは-mのモデルのもの: yolov3→darknet53.conv.74）。"
             "noneでスクラッチ学習"
    )
    parser.add_argument(
        "--partial", type=int, metavar="N",
        help="事前学習重みの先頭N層だけを使う（バックボーンを変更したCFG用、darknet partialを自動実行）"
    )
    parser.add_argument(
        "--no-weight-check", action="store_true",
        help="初期重みと cfg の対応検査を省く（cfg に合わない旧初期重みでの学習を再現するとき用）"
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="最初から学習し直す"
    )
    parser.add_argument(
        "--resume", type=str, metavar="WEIGHTS",
        help="指定weightsから再開"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="学習をスキップしてmAP計算のみ"
    )
    
    # 共通オプション
    parser.add_argument(
        "--config", "-c", type=str, default="config.yaml",
        help="設定ファイルパス (デフォルト: config.yaml)"
    )
    parser.add_argument(
        "--work-dir", "-w", type=str,
        help="作業ディレクトリ（work/の位置）"
    )
    parser.add_argument(
        "--gpus-train", type=str,
        help="学習用GPU (例: 4,5,6,7)"
    )
    parser.add_argument(
        "--gpus-map", type=str,
        help="mAP計算用GPU (例: 1,2,3,4,5)"
    )
    parser.add_argument(
        "--docker-image", type=str,
        help="Dockerイメージ名"
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    config_path = Path(args.config)
    
    if args.batch:
        # バッチモード
        if not config_path.exists():
            print(f"Error: 設定ファイルが必要です: {config_path}")
            sys.exit(1)
        
        config = load_config(config_path)
        experiments, gpu_config, docker_config, path_config = parse_config(config)
        
        # コマンドライン引数で上書き（darknet_pathは設定ファイルの値を引き継ぐ）
        if args.work_dir:
            path_config = PathConfig(work_dir=Path(args.work_dir), darknet_path=path_config.darknet_path)

        trainer = YOLOTrainer(gpu_config, docker_config, path_config)
        results = trainer.run_batch(experiments)
        
        sys.exit(0 if not results["failed"] else 1)
    
    else:
        # 単一実験モード
        if not args.resolution:
            parser.error("--resolution は必須です（--batch を使用しない場合）")
        if not args.model and not args.cfg:
            parser.error("--model か --cfg のどちらかが必要です（--batch を使用しない場合）")
        if args.partial is not None and args.partial <= 0:
            parser.error("--partial には1以上の層数を指定してください")
        
        # デフォルト設定またはconfig.yamlから読み込み
        if config_path.exists():
            config = load_config(config_path)
            _, gpu_config, docker_config, path_config = parse_config(config)
        else:
            gpu_config = GPUConfig()
            docker_config = DockerConfig()
            path_config = PathConfig()
        
        # コマンドライン引数で上書き
        if args.work_dir:
            path_config = PathConfig(work_dir=Path(args.work_dir), darknet_path=path_config.darknet_path)
        if args.gpus_train:
            gpu_config.train = args.gpus_train
        if args.gpus_map:
            gpu_config.map = args.gpus_map
        if args.docker_image:
            docker_config.image = args.docker_image
        
        experiment = Experiment(
            resolution=args.resolution,
            model=args.model or "yolov3",  # --cfgのみ指定時はyolov3の事前学習重みを使う
            clear=args.clear,
            resume_weights=args.resume,
            skip_train=args.skip_train,
            cfg=args.cfg,
            pretrained=args.pretrained,
            partial=args.partial,
            no_weight_check=args.no_weight_check
        )
        
        trainer = YOLOTrainer(gpu_config, docker_config, path_config)
        success = trainer.run_experiment(experiment)
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

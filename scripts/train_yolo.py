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
        pretrained_weights="darknet53.conv.74"
    ),
    "yolov3-tiny": ModelConfig(
        name="yolov3-tiny",
        base_cfg="yolov3-tiny-416.cfg",
        pretrained_weights="darknet19_448.conv.23"
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


def generate_obj_data(path_config: PathConfig) -> bool:
    """obj.dataファイルを自動生成"""
    
    obj_data_content = f"""classes = {NUM_CLASSES}
train = {path_config.train_file}
valid = {path_config.val_file}
names = {path_config.names_file}
backup = {path_config.backup_dir}
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
    config_name: str,
    skip_train: bool = False
) -> None:
    """Docker内で実行するスクリプトを生成"""
    
    script_content = f'''#!/bin/bash
set -e

cd /workspace

SCRIPT_START=$(date +%s)
echo "=========================================="
echo "Training started at: $(date)"
echo "=========================================="

'''
    
    if not skip_train:
        script_content += f'''
# 学習実行
TRAIN_START=$(date +%s)

# log_dirが存在することを確認
mkdir -p /workspace/{log_dir}

# チャートファイルのシンボリックリンクを作成（学習中からlog_dirに保存されるように）
rm -f /workspace/chart.png /workspace/chart_{config_name}.png
ln -s /workspace/{log_dir}/chart.png /workspace/chart.png
ln -s /workspace/{log_dir}/chart_{config_name}.png /workspace/chart_{config_name}.png
echo "Chart files will be saved to {log_dir}/"

echo ""
echo "[Training Command]"
echo "{train_cmd}"
echo ""
{train_cmd} 2>&1 | tee /workspace/{train_log}

# シンボリックリンクを削除
rm -f /workspace/chart.png /workspace/chart_{config_name}.png

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$((TRAIN_END - TRAIN_START))
TRAIN_HOURS=$((TRAIN_ELAPSED / 3600))
TRAIN_MINUTES=$(((TRAIN_ELAPSED % 3600) / 60))
TRAIN_SECONDS=$((TRAIN_ELAPSED % 60))

echo ""
echo "Training completed at: $(date)"
echo "Training time: ${{TRAIN_HOURS}}h ${{TRAIN_MINUTES}}m ${{TRAIN_SECONDS}}s (${{TRAIN_ELAPSED}} seconds)"
echo ""

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{exp.model}-{exp.resolution}_{timestamp}"
        
        print("=" * 50)
        print("YOLOv3 Training Script")
        print("=" * 50)
        print(f"Experiment:   {experiment_name}")
        print(f"Model:        {exp.model}")
        print(f"Resolution:   {exp.resolution}x{exp.resolution}")
        print(f"Classes:      {NUM_CLASSES}")
        print(f"Pre-trained:  {model_config.pretrained_weights}")
        print(f"Train GPUs:   {self.gpu.train}")
        print(f"mAP GPUs:     {self.gpu.map}")
        print(f"Clear:        {exp.clear}")
        print(f"Skip Train:   {exp.skip_train}")
        print("=" * 50)
        
        # Step 1: CFGファイル生成
        print("\n[Step 1] CFGファイルの生成...")
        base_cfg_path = self.paths.work_subdir / model_config.base_cfg
        output_cfg = self.paths.work_subdir / f"{exp.model}-{exp.resolution}.cfg"
        
        if not generate_cfg(base_cfg_path, output_cfg, exp.resolution):
            return False
        
        # Step 2: Weightsファイルの決定
        print("\n[Step 2] Weightsファイルの決定...")
        final_weights = self.paths.backup_dir / f"{exp.model}-{exp.resolution}_last.weights"
        
        if exp.resume_weights:
            start_weights = exp.resume_weights
            print(f"Resuming from: {start_weights}")
        elif final_weights.exists() and not exp.clear:
            start_weights = str(final_weights)
            print(f"Resuming from: {start_weights}")
        else:
            start_weights = model_config.pretrained_weights
            print(f"Starting from: {start_weights}")
        
        # Step 3: コマンド構築
        clear_flag = "-clear" if exp.clear else ""
        chart_path = f"{self.paths.log_dir}/chart_{exp.model}-{exp.resolution}.png"
        
        train_cmd = (
            f"{self.paths.darknet_path} detector train "
            f"{self.paths.data_file} {output_cfg} {start_weights} "
            f"-dont_show -gpus {self.gpu.train} -chart {chart_path} {clear_flag}"
        ).strip()
        
        map_cmd = (
            f"{self.paths.darknet_path} detector map "
            f"{self.paths.data_file} {output_cfg} {final_weights} "
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
            config_name=f"{exp.model}-{exp.resolution}",
            skip_train=exp.skip_train
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
            print(f"\n[{i}/{len(experiments)}] {exp.model} @ {exp.resolution}x{exp.resolution}")
            
            if self.run_experiment(exp):
                results["success"].append(f"{exp.model}-{exp.resolution}")
            else:
                results["failed"].append(f"{exp.model}-{exp.resolution}")
        
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
            model=exp_cfg["model"],
            clear=exp_cfg.get("clear", False),
            resume_weights=exp_cfg.get("resume_weights"),
            skip_train=exp_cfg.get("skip_train", False)
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
        
        # コマンドライン引数で上書き
        if args.work_dir:
            path_config = PathConfig(work_dir=Path(args.work_dir))
        
        trainer = YOLOTrainer(gpu_config, docker_config, path_config)
        results = trainer.run_batch(experiments)
        
        sys.exit(0 if not results["failed"] else 1)
    
    else:
        # 単一実験モード
        if not args.resolution or not args.model:
            parser.error("--resolution と --model は必須です（--batch を使用しない場合）")
        
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
            path_config = PathConfig(work_dir=Path(args.work_dir))
        if args.gpus_train:
            gpu_config.train = args.gpus_train
        if args.gpus_map:
            gpu_config.map = args.gpus_map
        if args.docker_image:
            docker_config.image = args.docker_image
        
        experiment = Experiment(
            resolution=args.resolution,
            model=args.model,
            clear=args.clear,
            resume_weights=args.resume,
            skip_train=args.skip_train
        )
        
        trainer = YOLOTrainer(gpu_config, docker_config, path_config)
        success = trainer.run_experiment(experiment)
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

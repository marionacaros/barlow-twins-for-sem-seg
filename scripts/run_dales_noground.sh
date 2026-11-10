#!/usr/bin/env bash
# DALES without ground points, z min-max scaled to [0, 1] (dataset 'dales_noground'):
# supervised training -> Barlow Twins pre-training initialized from the supervised encoder -> fine-tuning from
# the Barlow Twins encoder -> test of both models, logged to MLflow. The supervised models use ISNS class weights
# (1/sqrt of the training points per class) by default.
#
# Usage (from anywhere; start the MLflow server first):
#   MLFLOW_TRACKING_URI=http://localhost:5000 scripts/run_dales_noground.sh
# Environment variables: MLFLOW_TRACKING_URI (default http://localhost:5000), PYTHON (default: prod_pytorch10),
# DALES_DIR (default: the float64 DALES windows), WEIGHTING (isns or fixed, default isns), SUPERVISED_CKPT (existing supervised .pth: skips supervised training).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}
export MLFLOW_DISABLE_AGENT_HINT=1
# prod_pytorch10 mixes pip protobuf 7 with a stale compiled protobuf 3.20 extension (google/protobuf/pyext), which
# breaks the MLflow REST client; use protobuf's own upb backend instead
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-upb}
P=${PYTHON:-/home/m.caros/anaconda3/envs/prod_pytorch10/bin/python}
D=${DALES_DIR:-/dades/LIDAR/vehicle_detection/DALES/dales_25x25_f64}
W=${WEIGHTING:-isns}
BT_DIR=src/bt/checkpoints_dales_noground_$W
CKPT_DIR=src/checkpoints/dales_noground_$W

if [[ $MLFLOW_TRACKING_URI == http* ]] && ! curl -sf "$MLFLOW_TRACKING_URI/health" > /dev/null; then
    echo "MLflow server not reachable at $MLFLOW_TRACKING_URI" >&2
    exit 1
fi

TEST_CKPTS=()
if [[ -n ${SUPERVISED_CKPT:-} ]]; then
    echo "[$(date)] using supervised checkpoint $SUPERVISED_CKPT (already tested, not tested again)"
else
    echo "[$(date)] supervised training"
    $P src/bt/train_seg_pointnet.py --dataset dales_noground --dataset_folder "$D/train" --weighting $W \
        --checkpoint_dir $CKPT_DIR/scratch
    SUPERVISED_CKPT=$(ls -t $CKPT_DIR/scratch/*.pth | head -1)
    TEST_CKPTS+=("$SUPERVISED_CKPT")
fi

echo "[$(date)] Barlow Twins pre-training, encoder initialized from $SUPERVISED_CKPT"
$P src/bt/train_BT_pointnet.py --dataset dales_noground --dataset_folder "$D/train" --output_dir $BT_DIR \
    --init_checkpoint "$SUPERVISED_CKPT"

# checkpoint with the lowest val_loss, as recorded by the ModelCheckpoint callback in the last checkpoint
BT_CKPT=$($P -c "
import glob, os, torch
last = max(glob.glob('$BT_DIR/*.ckpt'), key=os.path.getmtime)
callbacks = torch.load(last, map_location='cpu', weights_only=False)['callbacks']
print(next(v for k, v in callbacks.items() if k.startswith('ModelCheckpoint'))['best_model_path'])")
echo "[$(date)] best Barlow Twins checkpoint: $BT_CKPT"

echo "[$(date)] fine-tuning from Barlow Twins"
$P src/bt/train_seg_pointnet.py --dataset dales_noground --dataset_folder "$D/train" \
    --model_checkpoint "$BT_CKPT" --weighting $W --checkpoint_dir $CKPT_DIR/barlow

TEST_CKPTS=("$(ls -t $CKPT_DIR/barlow/*.pth | head -1)" "${TEST_CKPTS[@]}")
for CK in "${TEST_CKPTS[@]}"; do
    echo "[$(date)] test: $CK"
    $P src/bt/test_seg_pointnet.py --dataset dales_noground --dataset_folder "$D/test" --model_checkpoint "$CK" \
        --number_of_workers 8
done
echo "[$(date)] done"

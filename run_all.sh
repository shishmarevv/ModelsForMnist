set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOGS="./output/logs"
mkdir -p "$LOGS"

echo "Launching all experiments in parallel..."

python main.py --model mlp --mode retries \
    --hidden_dims 256 128 --dropout 0.0 0.0 \
    --epochs 30 --lr 0.001 --batch_size 512 -o mlp_arch1 \
    > "$LOGS/mlp_arch1.log" 2>&1 &

python main.py --model mlp --mode retries \
    --hidden_dims 512 256 128 --dropout 0.0 0.0 0.0 \
    --epochs 30 --lr 0.001 --batch_size 512 -o mlp_arch2 \
    > "$LOGS/mlp_arch2.log" 2>&1 &

python main.py --model cnn --mode retries \
    --hidden_dims 32 64 --dropout 0.0 0.0 \
    --epochs 30 --lr 0.001 --batch_size 512 -o cnn_arch1 \
    > "$LOGS/cnn_arch1.log" 2>&1 &

python main.py --model cnn --mode retries \
    --hidden_dims 64 128 --dropout 0.0 0.0 \
    --epochs 30 --lr 0.001 --batch_size 512 -o cnn_arch2 \
    > "$LOGS/cnn_arch2.log" 2>&1 &

python main.py --model cnn --mode retries \
    --hidden_dims 32 64 128 --dropout 0.0 0.0 0.0 \
    --epochs 30 --lr 0.001 --batch_size 512 -o cnn_arch3 \
    > "$LOGS/cnn_arch3.log" 2>&1 &

python main.py --model cnn --mode retries \
    --hidden_dims 64 128 --dropout 0.2 0.2 \
    --epochs 30 --lr 0.001 --batch_size 512 -o cnn_dropout02 \
    > "$LOGS/cnn_dropout02.log" 2>&1 &

python main.py --model cnn --mode retries \
    --hidden_dims 64 128 --dropout 0.5 0.5 \
    --epochs 30 --lr 0.001 --batch_size 512 -o cnn_dropout05 \
    > "$LOGS/cnn_dropout05.log" 2>&1 &

python main.py --model cnn --mode retries --device cpu \
    --hidden_dims 32 64 128 --dropout 0.0 0.0 0.0 \
    --epochs 30 --lr 0.001 --batch_size 512 -o cnn_arch3_cpu \
    > "$LOGS/cnn_arch3_cpu.log" 2>&1 &

python main.py --model cnn --mode retries --device cuda \
    --hidden_dims 32 64 128 --dropout 0.0 0.0 0.0 \
    --epochs 30 --lr 0.001 --batch_size 512 -o cnn_arch3_gpu \
    > "$LOGS/cnn_arch3_gpu.log" 2>&1 &

echo "All 9 experiments running. Waiting for completion..."
wait
echo "All done. Results in ./output/. Logs in $LOGS/"

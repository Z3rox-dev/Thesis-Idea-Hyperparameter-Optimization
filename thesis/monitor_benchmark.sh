#!/bin/bash
# Monitor GPU and CPU during benchmark execution

# Create monitoring directory
mkdir -p monitoring_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="monitoring_logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Benchmark Monitor"
echo "=========================================="
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping monitoring..."
    kill $GPU_PID 2>/dev/null
    kill $CPU_PID 2>/dev/null
    wait 2>/dev/null
    echo "Monitoring stopped. Logs saved to: $LOG_DIR"
}
trap cleanup EXIT INT TERM

# Start GPU monitoring in background
echo "Starting GPU monitoring..."
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 1 > "$LOG_DIR/gpu_monitor.csv" &
GPU_PID=$!

# Start CPU monitoring in background
echo "Starting CPU monitoring..."
{
    echo "timestamp,cpu_percent,mem_percent,mem_used_gb,mem_total_gb"
    while true; do
        TIMESTAMP=$(date +%Y-%m-%d\ %H:%M:%S)
        # Get CPU usage
        CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
        # Get memory usage
        MEM=$(free -g | awk '/^Mem:/ {printf "%.1f,%.1f,%d,%d", ($3/$2)*100, $3/$2*100, $3, $2}')
        echo "$TIMESTAMP,$CPU,$MEM"
        sleep 1
    done
} > "$LOG_DIR/cpu_monitor.csv" &
CPU_PID=$!

echo "GPU monitoring PID: $GPU_PID"
echo "CPU monitoring PID: $CPU_PID"
echo ""
echo "=========================================="
echo "Running benchmark..."
echo "=========================================="
echo ""

# Run the benchmark
# Reduced runtime: single seed, smaller budget configurable via env vars
# Allow overrides: BENCH_BUDGET, BENCH_SEED, BENCH_METHODS
: ${BENCH_BUDGET:=30}
: ${BENCH_SEED:=11}
: ${BENCH_METHODS:=curv,optuna,random}

echo "Using budget=$BENCH_BUDGET seed=$BENCH_SEED methods=$BENCH_METHODS"
python3 thesis/advanced_benchmark.py --budget "$BENCH_BUDGET" --seeds "$BENCH_SEED" --methods "$BENCH_METHODS" --gpu --verbose --test-topk 5 2>&1 | tee "$LOG_DIR/benchmark_output.txt"

BENCHMARK_EXIT=$?

echo ""
echo "=========================================="
echo "Benchmark completed with exit code: $BENCHMARK_EXIT"
echo "=========================================="

# Cleanup will be called automatically by trap
exit $BENCHMARK_EXIT

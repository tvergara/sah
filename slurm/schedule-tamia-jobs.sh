#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH -o /project/aip-sreddy/tvergara/slurm-logs/parallel-%j.out
#SBATCH --account=aip-sreddy


# Schedule jobs in parallel across 4 H100 GPUs using GNU Parallel
# Automatically chains to next 12hr window if jobs remain
# Usage: sbatch slurm/schedule-tamia-jobs.sh

# Chain tracking to prevent infinite loops
CHAIN_COUNT=${CHAIN_COUNT:-0}
MAX_CHAINS=10

echo "========================================="
echo "Parallel Job Scheduler (12hr window)"
echo "========================================="
echo "Chain: $((CHAIN_COUNT + 1))/$MAX_CHAINS"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

source ~/.bashrc
cd $SCRATCH/sah

# Load required modules for tamia cluster
module load gcc
module load arrow
module load python/3.12.4
module load cuda/12.6
module load httpproxy/1.0

# Disable hydra-auto-schema to prevent YAML file corruption
export HYDRA_AUTO_SCHEMA=0

# Activate virtual environment
. .venv/bin/activate

# Job list and log paths
JOB_LIST="jobs.txt"
LOG_FILE="slurm/parallel.log"

# Generate job list if it doesn't exist
if [ ! -f $JOB_LIST ]; then
  echo "Generating job list..."
  bash slurm/generate-job-list.sh
fi

TOTAL_JOBS=$(wc -l < $JOB_LIST)
echo "Total jobs in queue: $TOTAL_JOBS"

# Count completed jobs from log
if [ -f $LOG_FILE ]; then
  COMPLETED=$(awk 'NR>1 && $7==0 {count++} END {print count+0}' $LOG_FILE)
  FAILED=$(awk 'NR>1 && $7!=0 {count++} END {print count+0}' $LOG_FILE)
  REMAINING=$((TOTAL_JOBS - COMPLETED))
  echo "Completed: $COMPLETED"
  echo "Failed: $FAILED"
  echo "Remaining: $REMAINING"
else
  COMPLETED=0
  FAILED=0
  REMAINING=$TOTAL_JOBS
  echo "Starting fresh (no previous log found)"
fi

echo ""
echo "Running 4 jobs in parallel (one per GPU)"
echo "Time limit: 12 hours (will stop at 11.5 hrs to chain)"
echo ""

# Export CUDA devices so they're visible to parallel jobs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Time management: stop 30 minutes before time limit to allow cleanup
# 12 hours = 43200 seconds, stop at 11.5 hours = 41400 seconds
TIMEOUT=41400

# Run jobs with GNU Parallel with timeout
# Each job uses run-single-job.sh wrapper for proper environment setup
# --timeout: Kill jobs after timeout (in seconds)
# --joblog: Append to same log file (enables true resume)
# --resume: Skip already completed jobs
# --resume-failed: Retry failed jobs

timeout $TIMEOUT parallel \
  --jobs 4 \
  --joblog $LOG_FILE \
  --resume \
  --resume-failed \
  --line-buffer \
  --will-cite \
  'CUDA_VISIBLE_DEVICES=$((({%} - 1))) bash slurm/run-single-job.sh {}' \
  :::: $JOB_LIST

PARALLEL_EXIT=$?

echo ""
echo "========================================="
echo "Batch Complete"
echo "========================================="
echo "End time: $(date)"
echo "Parallel exit code: $PARALLEL_EXIT"
echo ""

# Print summary
if [ -f $LOG_FILE ]; then
  echo "Current Progress:"
  echo "-----------------"
  COMPLETED=$(awk 'NR>1 && $7==0 {count++} END {print count+0}' $LOG_FILE)
  FAILED=$(awk 'NR>1 && $7!=0 {count++} END {print count+0}' $LOG_FILE)
  REMAINING=$((TOTAL_JOBS - COMPLETED))

  echo "Total jobs: $TOTAL_JOBS"
  echo "Completed: $COMPLETED"
  echo "Failed: $FAILED"
  echo "Remaining: $REMAINING"
  echo ""
fi

# Check if we need to chain another job
if [ $REMAINING -gt 0 ]; then
  # Check if we've reached the maximum chain limit
  if [ $CHAIN_COUNT -ge $MAX_CHAINS ]; then
    echo "========================================="
    echo "WARNING: Maximum chain limit reached!"
    echo "========================================="
    echo "Completed $CHAIN_COUNT chains ($(($CHAIN_COUNT * 12)) hours total)"
    echo "Stopping to prevent infinite loop."
    echo ""
    echo "Remaining jobs: $REMAINING"
    echo "To continue, manually run:"
    echo "  sbatch slurm/schedule-tamia-jobs.sh"
    echo ""
    echo "Or to reset the chain counter and continue:"
    echo "  CHAIN_COUNT=0 sbatch slurm/schedule-tamia-jobs.sh"
  else
    echo "Jobs remaining! Submitting next 12hr batch..."
    NEXT_CHAIN=$((CHAIN_COUNT + 1))
    NEXT_JOB=$(sbatch --export=ALL,CHAIN_COUNT=$NEXT_CHAIN slurm/schedule-tamia-jobs.sh)
    echo "Chained job: $NEXT_JOB (Chain $((NEXT_CHAIN + 1))/$MAX_CHAINS)"
    echo ""
    echo "This is job chaining - jobs will continue until all are complete."
    echo "You can monitor progress in: $LOG_FILE"
  fi
else
  echo "All jobs complete! 🎉"
  echo ""
  echo "Final summary:"
  echo "--------------"
  echo "Total: $TOTAL_JOBS jobs"
  echo "Successful: $COMPLETED jobs"
  echo "Failed: $FAILED jobs"

  if [ $FAILED -gt 0 ]; then
    echo ""
    echo "WARNING: Some jobs failed. Check the log:"
    echo "  $LOG_FILE"
    echo ""
    echo "To retry failed jobs, run:"
    echo "  sbatch slurm/schedule-tamia-jobs.sh"
  fi
fi

exit 0

#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH -o /project/aip-sreddy/tvergara/slurm-logs/parallel-%A_%a.out
#SBATCH --account=aip-sreddy


# Schedule jobs in parallel across 4 H100 GPUs using GNU Parallel
# Supports parallelization via SLURM job arrays
#
# Usage:
#   Single job (process all jobs.txt):
#     sbatch slurm/schedule-tamia-jobs.sh
#
#   With custom job file:
#     JOB_FILE=missing-jobs.txt sbatch slurm/schedule-tamia-jobs.sh
#
#   Parallel jobs (each processes a subset):
#     sbatch --array=0-9 slurm/schedule-tamia-jobs.sh
#     JOB_FILE=missing-jobs.txt sbatch --array=0-9 slurm/schedule-tamia-jobs.sh

echo "========================================="
echo "Parallel Job Scheduler (24hr window)"
echo "========================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job file: ${JOB_FILE:-jobs.txt}"
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
fi
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

# Job list and log paths (can be overridden via environment variable)
JOB_LIST="${JOB_FILE:-jobs.txt}"

if [ ! -f $JOB_LIST ]; then
  echo "ERROR: $JOB_LIST not found!"
  exit 1
fi

TOTAL_JOBS=$(wc -l < $JOB_LIST)

# Determine which subset of jobs to process
if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ -n "$SLURM_ARRAY_TASK_COUNT" ]; then
  JOBS_PER_TASK=$(( (TOTAL_JOBS + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
  START_LINE=$(( SLURM_ARRAY_TASK_ID * JOBS_PER_TASK + 1 ))
  END_LINE=$(( START_LINE + JOBS_PER_TASK - 1 ))

  if [ $END_LINE -gt $TOTAL_JOBS ]; then
    END_LINE=$TOTAL_JOBS
  fi

  if [ $START_LINE -gt $TOTAL_JOBS ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID has no jobs to process"
    exit 0
  fi

  SUBSET_JOB_LIST="jobs_subset_${SLURM_ARRAY_TASK_ID}.txt"
  sed -n "${START_LINE},${END_LINE}p" $JOB_LIST > $SUBSET_JOB_LIST
  JOB_LIST=$SUBSET_JOB_LIST
  LOG_FILE="slurm/parallel_${SLURM_ARRAY_TASK_ID}.log"

  echo "Array parallelization: Task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
  echo "Processing jobs ${START_LINE}-${END_LINE} of ${TOTAL_JOBS}"
  TOTAL_JOBS=$(wc -l < $JOB_LIST)
else
  LOG_FILE="slurm/parallel.log"
  echo "Processing all jobs: $TOTAL_JOBS"
fi

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
echo "Time limit: 24 hours (will stop at 23.5 hrs for cleanup)"
echo ""

# Export CUDA devices so they're visible to parallel jobs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Time management: stop 30 minutes before time limit to allow cleanup
# 24 hours = 86400 seconds, stop at 23.5 hours = 84600 seconds
TIMEOUT=84600

# Run jobs with GNU Parallel with timeout
# Each job uses run-single-job.sh wrapper for proper environment setup
# --timeout: Kill jobs after timeout (in seconds)
# --joblog: Append to same log file (enables true resume)
# --resume: Skip already completed jobs
# --resume-failed: Retry failed jobs

timeout $TIMEOUT parallel \
  --jobs 4 \
  --joblog $LOG_FILE \
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

if [ $REMAINING -gt 0 ]; then
  echo "Jobs remaining: $REMAINING"
  echo ""
  if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    echo "To continue this subset, rerun the same array task"
  else
    echo "To continue, manually run:"
    if [ "$JOB_LIST" != "jobs.txt" ]; then
      echo "  JOB_FILE=$JOB_LIST sbatch slurm/schedule-tamia-jobs.sh"
    else
      echo "  sbatch slurm/schedule-tamia-jobs.sh"
    fi
  fi
else
  echo "All jobs complete!"
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
  fi
fi

if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ -f "jobs_subset_${SLURM_ARRAY_TASK_ID}.txt" ]; then
  rm "jobs_subset_${SLURM_ARRAY_TASK_ID}.txt"
fi

exit 0

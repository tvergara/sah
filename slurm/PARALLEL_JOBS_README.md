# Parallel Job Scheduling for Multi-GPU Nodes

This system efficiently schedules single-GPU jobs across a 4-GPU node using GNU Parallel with **automatic job chaining** for 12-hour time limits.

## Quick Start

```bash
# 1. Generate the job list (or edit jobs.txt manually)
bash slurm/generate-job-list.sh

# 2. Submit to SLURM (only need to do this ONCE!)
sbatch slurm/schedule-tamia-jobs.sh
```

That's it! The script will automatically chain to new 12hr windows until all jobs complete.

## How Job Chaining Works

1. **First 12hr window**: Runs as many jobs as possible
2. **Before timeout**: Stops at 11.5 hours to cleanup
3. **Auto-submit**: If jobs remain, automatically submits next 12hr batch
4. **Repeat**: Continues until all jobs complete

You only submit once - the system handles everything!

## Files

- **schedule-tamia-jobs.sh**: Main SLURM script (12hr windows, auto-chains)
- **generate-job-list.sh**: Generates jobs.txt from your experiment configs
- **jobs.txt**: One job command per line (editable)
- **parallel.log**: Single persistent log file tracking ALL jobs across all chains

## How GNU Parallel Works

Automatically:

- Runs 4 jobs simultaneously (one per GPU)
- Starts new jobs as soon as a GPU becomes free
- Logs all job details (start time, duration, exit status)
- Resumes from interruptions (never redoes completed work)

## Features

### GPU Assignment

Jobs are automatically assigned to GPUs 0-3 in round-robin as they start.

### Job Logging

All jobs are logged to `slurm/parallel-YYYYMMDD-HHMMSS.log` with:

- Start/end time
- Duration
- Exit code
- Command executed

### Resume Failed Jobs

If a job fails or you want to retry failures:

```bash
sbatch slurm/schedule-tamia-jobs.sh
```

It will:

- Skip all completed jobs (reads from parallel.log)
- Retry all failed jobs
- Continue any remaining jobs

### Progress Monitoring

While running, you can check progress:

```bash
# Watch the SLURM output (shows current 12hr batch)
tail -f ~/scratch/slurm-logs/parallel-<job-id>.out

# Check overall progress (across all chains)
tail -f slurm/parallel.log

# Count completed jobs
awk 'NR>1 && $7==0 {count++} END {print count}' slurm/parallel.log

# See active SLURM jobs
squeue -u $USER
```

## Customization

### Edit Job List

Manually edit `slurm/jobs.txt` to:

- Remove specific jobs
- Add new jobs
- Reorder jobs

### Adjust Resources

Edit `schedule-tamia-jobs.sh` to change:

- `--time`: Maximum walltime (currently 12:00:00)
- `TIMEOUT`: Stop time in seconds (currently 41400 = 11.5hrs)
- `--mem`: Memory per node (currently 192G)
- `--cpus-per-task`: CPUs (currently 16 = 4 per GPU)

### Run Subset of Jobs

Create a custom job list:

```bash
# Only run first 20 jobs
head -20 slurm/jobs.txt > slurm/jobs-subset.txt

# Edit schedule-tamia-jobs.sh to use jobs-subset.txt instead
```

## Current Job Configuration

Based on your scripts, this generates:

- **15 models** (SmolLM3 variants + OLMo3 variants)
- **2 learning rates** (1e-4, 1e-5)
- **6 max_examples** (1024, 2048, 4096, 8192, 16384, 32768)
- **IFEval dataset only**
- **Online-coding strategy only**

Total: ~216 jobs

Expected runtime:

- Varies by model and max_examples
- Small jobs (1024 examples): ~30min
- Large jobs (32768 examples): ~4hrs
- Total walltime: ~2-4 days (via automatic job chaining)
    - Each 12hr window completes ~40-50 jobs
    - System auto-chains until all ~216 jobs complete

## Troubleshooting

### GNU Parallel not found

```bash
# Load module (if available)
module load parallel

# Or install locally
pip install parallel  # or conda install -c conda-forge parallel
```

### Jobs fail with CUDA errors

Check that CUDA_VISIBLE_DEVICES is being set correctly:

```bash
# Add to jobs.txt for debugging
echo $CUDA_VISIBLE_DEVICES && nvidia-smi
```

### Need to cancel

```bash
# Cancel current batch only (more will auto-submit)
scancel <job-id>

# Cancel ALL your jobs (stops the chain)
scancel -u $USER

# To prevent auto-chaining before canceling:
# Remove or rename slurm/parallel.log so next chain sees no progress
mv slurm/parallel.log slurm/parallel.log.backup
```

### Start fresh

```bash
# Remove progress log to start from scratch
rm slurm/parallel.log

# Regenerate job list
bash slurm/generate-job-list.sh

# Submit
sbatch slurm/schedule-tamia-jobs.sh
```

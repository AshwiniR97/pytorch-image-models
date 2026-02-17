#!/usr/bin/env python3
"""
MLflow Logger for timm Training

Monitors timm summary.csv output and logs metrics to MLflow in real-time.
Runs as a background process alongside training.

Usage:
    python mlflow_logger.py --summary-file output/exp/summary.csv --run-name my_exp
"""
import argparse
import csv
import os
import time
import yaml
from pathlib import Path

try:
    import mlflow
    has_mlflow = True
except ImportError:
    has_mlflow = False
    print("WARNING: mlflow not installed. Install with: pip install mlflow")


class MLflowTrainingLogger:
    """Watch timm summary.csv and log metrics to MLflow."""
    
    def __init__(
        self,
        summary_file,
        run_name,
        tracking_uri=None,
        experiment_name='cifar10_vit_training',
        poll_interval=5.0,
    ):
        self.summary_file = Path(summary_file)
        self.run_name = run_name
        self.poll_interval = poll_interval
        self.last_row_count = 0
        self.run_id = None
        
        if not has_mlflow:
            print("MLflow not available. Metrics will not be logged.")
            return
        
        # Set tracking URI (defaults to ./mlruns)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        mlflow.start_run(run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id
        print(f"Started MLflow run: {run_name} (ID: {self.run_id})")
        
        # Log hyperparameters if args.yaml exists
        self._log_hyperparameters()
    
    def _log_hyperparameters(self):
        """Log training hyperparameters from args.yaml if it exists."""
        if not has_mlflow:
            return
        
        args_file = self.summary_file.parent / 'args.yaml'
        if args_file.exists():
            try:
                with open(args_file, 'r') as f:
                    args = yaml.safe_load(f)
                
                # Log key hyperparameters
                params = {
                    'model': args.get('model', 'unknown'),
                    'dataset': args.get('dataset', 'unknown'),
                    'epochs': args.get('epochs', 0),
                    'batch_size': args.get('batch_size', 0),
                    'lr': args.get('lr', 0),
                    'optimizer': args.get('opt', 'unknown'),
                    'weight_decay': args.get('weight_decay', 0),
                    'img_size': args.get('img_size', 0),
                    'num_classes': args.get('num_classes', 0),
                    'mixup': args.get('mixup', 0),
                    'cutmix': args.get('cutmix', 0),
                    'aa': args.get('aa', 'none'),
                    'amp': args.get('amp', False),
                }
                mlflow.log_params(params)
                print(f"Logged hyperparameters from {args_file}")
            except Exception as e:
                print(f"Warning: Could not log hyperparameters: {e}")
    
    def _read_new_rows(self):
        """Read new rows from summary.csv since last check."""
        if not self.summary_file.exists():
            return []
        
        try:
            with open(self.summary_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Get only new rows
            new_rows = rows[self.last_row_count:]
            self.last_row_count = len(rows)
            return new_rows
        except Exception as e:
            print(f"Warning: Error reading summary file: {e}")
            return []
    
    def _log_metrics(self, row):
        """Log metrics from a single row to MLflow."""
        if not has_mlflow:
            return
        
        try:
            epoch = int(row['epoch'])
            metrics = {}
            
            # Log all metrics from the row
            for key, value in row.items():
                if key == 'epoch':
                    continue
                try:
                    metrics[key] = float(value)
                except (ValueError, TypeError):
                    pass
            
            if metrics:
                mlflow.log_metrics(metrics, step=epoch)
                print(f"Epoch {epoch}: {metrics}")
        except Exception as e:
            print(f"Warning: Could not log metrics: {e}")
    
    def watch(self):
        """Watch summary.csv and log metrics in real-time."""
        if not has_mlflow:
            print("MLflow not available. Running in dry-run mode (monitoring only).")
        
        print(f"Watching {self.summary_file}...")
        print(f"Will poll every {self.poll_interval} seconds")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                new_rows = self._read_new_rows()
                for row in new_rows:
                    self._log_metrics(row)
                
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\nStopping MLflow logger...")
        finally:
            self.cleanup()
    
    def log_final_artifacts(self, checkpoint_path=None, log_file=None):
        """Log final artifacts to MLflow."""
        if not has_mlflow:
            return
        
        try:
            # Log checkpoint if it exists
            if checkpoint_path and Path(checkpoint_path).exists():
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
                print(f"Logged checkpoint: {checkpoint_path}")
            
            # Log training log if it exists
            if log_file and Path(log_file).exists():
                mlflow.log_artifact(log_file, artifact_path="logs")
                print(f"Logged training log: {log_file}")
            
            # Log args.yaml
            args_file = self.summary_file.parent / 'args.yaml'
            if args_file.exists():
                mlflow.log_artifact(str(args_file), artifact_path="config")
        except Exception as e:
            print(f"Warning: Could not log artifacts: {e}")
    
    def cleanup(self):
        """End MLflow run and cleanup."""
        if has_mlflow and mlflow.active_run():
            print(f"Ending MLflow run: {self.run_id}")
            mlflow.end_run()


def parse_args():
    parser = argparse.ArgumentParser(description='MLflow logger for timm training')
    parser.add_argument(
        '--summary-file',
        type=str,
        required=True,
        help='Path to summary.csv file to monitor'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        required=True,
        help='Name for the MLflow run'
    )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (default: ./mlruns)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='cifar10_vit_training',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--poll-interval',
        type=float,
        default=5.0,
        help='Seconds between checks for new metrics (default: 5.0)'
    )
    parser.add_argument(
        '--log-artifacts',
        action='store_true',
        help='Log final artifacts (checkpoint, logs) after watching completes'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to log as artifact'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to training log to log as artifact'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger = MLflowTrainingLogger(
        summary_file=args.summary_file,
        run_name=args.run_name,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        poll_interval=args.poll_interval,
    )
    
    # Watch and log metrics
    logger.watch()
    
    # Log artifacts if requested
    if args.log_artifacts:
        logger.log_final_artifacts(
            checkpoint_path=args.checkpoint,
            log_file=args.log_file,
        )


if __name__ =='__main__':
    main()

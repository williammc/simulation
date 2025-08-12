"""
Evaluation orchestrator for running comprehensive SLAM benchmarks.
"""

import yaml
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.comparison import ComparisonResult, EstimatorRunner
from src.evaluation.metrics import TrajectoryMetrics
from src.plotting.evaluation_dashboard import create_evaluation_dashboard


class EvaluationOrchestrator:
    """Orchestrates the complete evaluation pipeline."""
    
    def __init__(self, config_path: str):
        """
        Initialize the evaluation orchestrator.
        
        Args:
            config_path: Path to the global evaluation config YAML
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.output_dir = Path(self.config['evaluation']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.results = {}
        
    def _load_config(self) -> Dict:
        """Load the evaluation configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        handlers = []
        if log_config.get('console', True):
            handlers.append(logging.StreamHandler())
        
        if log_config.get('file'):
            log_file = self.output_dir / log_config['file']
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting evaluation pipeline: {self.config['evaluation']['name']}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.
        
        Returns:
            Dictionary containing all evaluation results
        """
        self.logger.info("=" * 80)
        self.logger.info("SLAM EVALUATION PIPELINE")
        self.logger.info("=" * 80)
        
        # Step 1: Prepare datasets
        datasets = self._prepare_datasets()
        
        # Step 2: Run estimators on all datasets
        estimation_results = self._run_estimators(datasets)
        
        # Step 3: Compute metrics and comparisons
        comparison_results = self._compute_comparisons(estimation_results)
        
        # Step 4: Extract KPIs
        kpis = self._extract_kpis(comparison_results)
        
        # Step 5: Generate dashboard
        dashboard_path = self._generate_dashboard(comparison_results, kpis)
        
        # Step 6: Save results
        self._save_results(comparison_results, kpis, dashboard_path)
        
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION COMPLETE")
        self.logger.info(f"Dashboard: {dashboard_path}")
        self.logger.info("=" * 80)
        
        return {
            'datasets': datasets,
            'results': comparison_results,
            'kpis': kpis,
            'dashboard': str(dashboard_path)
        }
    
    def _prepare_datasets(self) -> Dict[str, Path]:
        """
        Prepare all datasets for evaluation.
        
        Returns:
            Dictionary mapping dataset names to their paths
        """
        self.logger.info("Preparing datasets...")
        datasets = {}
        
        # Handle simulated datasets
        if 'simulated' in self.config['datasets']:
            sim_config = self.config['datasets']['simulated']
            if sim_config.get('generate_if_missing', True):
                for dataset in sim_config['types']:
                    dataset_name = dataset['name']
                    dataset_path = self._generate_simulated_dataset(dataset_name, dataset['config'])
                    datasets[f"sim_{dataset_name}"] = dataset_path
        
        # Handle TUM-VI datasets
        if 'tum_vi' in self.config['datasets']:
            tum_config = self.config['datasets']['tum_vi']
            for sequence in tum_config['sequences']:
                seq_name = sequence['name']
                dataset_path = self._prepare_tumvi_dataset(seq_name, tum_config)
                datasets[f"tumvi_{seq_name}"] = dataset_path
        
        self.logger.info(f"Prepared {len(datasets)} datasets")
        return datasets
    
    def _generate_simulated_dataset(self, name: str, config: Dict) -> Path:
        """
        Generate a simulated dataset if it doesn't exist.
        
        Args:
            name: Dataset name
            config: Dataset configuration
            
        Returns:
            Path to the generated dataset
        """
        dataset_dir = self.output_dir / "datasets" / "simulated" / name
        dataset_file = dataset_dir / f"simulation_{name}.json"
        
        if dataset_file.exists() and not self.config['evaluation'].get('overwrite', False):
            self.logger.info(f"Using existing simulated dataset: {name}")
            # If it's a directory, find the actual JSON file inside
            if dataset_file.is_dir():
                json_files = list(dataset_file.glob("*.json"))
                if json_files:
                    return json_files[0]
                else:
                    raise FileNotFoundError(f"No JSON file found in existing dataset directory: {dataset_file}")
            return dataset_file
        
        self.logger.info(f"Generating simulated dataset: {name}")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simulation config file
        traj_config = config['trajectory']
        noise_config = config.get('noise', {})
        landmark_config = config.get('landmarks', {})
        
        # Build simulation config
        sim_config = {
            'trajectory': {
                'type': traj_config['type'],
                'duration': traj_config.get('duration', 60),
                'params': {}
            },
            'sensors': {
                'imu': {
                    'rate': traj_config.get('rate', 200.0),
                    'noise': {
                        'enabled': noise_config.get('add_noise', False),
                        'accel_noise_density': noise_config.get('imu_noise_level', 0.01),
                        'gyro_noise_density': noise_config.get('imu_noise_level', 0.001) * 0.1
                    }
                },
                'camera': {
                    'rate': 30.0,
                    'noise': {
                        'enabled': noise_config.get('add_noise', False),
                        'pixel_std': noise_config.get('camera_noise_level', 1.0)
                    }
                }
            },
            'landmarks': {
                'count': landmark_config.get('num_landmarks', 200),
                'distribution': landmark_config.get('distribution', 'uniform')
            }
        }
        
        # Add trajectory-specific parameters
        if traj_config['type'] == 'circle':
            sim_config['trajectory']['params']['radius'] = traj_config.get('radius', 10.0)
        elif traj_config['type'] == 'figure8':
            sim_config['trajectory']['params']['scale_x'] = traj_config.get('scale_x', 5.0)
            sim_config['trajectory']['params']['scale_y'] = traj_config.get('scale_y', 3.0)
        elif traj_config['type'] == 'spiral':
            sim_config['trajectory']['params']['initial_radius'] = traj_config.get('initial_radius', 1.0)
            sim_config['trajectory']['params']['final_radius'] = traj_config.get('final_radius', 5.0)
        elif traj_config['type'] == 'line':
            sim_config['trajectory']['params']['start'] = traj_config.get('start', [0, 0, 0])
            sim_config['trajectory']['params']['end'] = traj_config.get('end', [20, 0, 0])
        
        # Save config to temp file
        config_file = dataset_dir / f"sim_config_{name}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sim_config, f)
        
        # Build simulation command
        cmd = [
            "./run.sh", "simulate",
            traj_config['type'],
            "--config", str(config_file),
            "--duration", str(traj_config.get('duration', 60)),
            "--output", str(dataset_file)
        ]
        
        # Add noise flag if configured
        if noise_config.get('add_noise', False):
            cmd.append("--add-noise")
        
        # Run simulation
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # The simulate command creates a directory with timestamped file
            # Find the actual generated file
            if dataset_file.is_dir():
                json_files = list(dataset_file.glob("*.json"))
                if json_files:
                    actual_file = json_files[0]  # Get the first (should be only) JSON file
                    self.logger.info(f"Generated dataset: {actual_file}")
                    return actual_file
                else:
                    raise FileNotFoundError(f"No JSON file found in {dataset_file}")
            else:
                self.logger.info(f"Generated dataset: {dataset_file}")
                return dataset_file
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to generate dataset {name}: {e.stderr}")
            raise
    
    def _prepare_tumvi_dataset(self, sequence: str, config: Dict) -> Path:
        """
        Prepare a TUM-VI dataset for evaluation.
        
        Args:
            sequence: Sequence name
            config: TUM-VI configuration
            
        Returns:
            Path to the prepared dataset
        """
        base_path = Path(config['base_path'])
        sequence_dir = base_path / sequence
        output_file = self.output_dir / "datasets" / "tumvi" / f"{sequence}.json"
        
        if output_file.exists() and not self.config['evaluation'].get('overwrite', False):
            self.logger.info(f"Using existing TUM-VI dataset: {sequence}")
            return output_file
        
        self.logger.info(f"Converting TUM-VI dataset: {sequence}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset exists
        if not sequence_dir.exists():
            if config.get('download_if_missing', True):
                self._download_tumvi_dataset(sequence, base_path)
            else:
                raise FileNotFoundError(f"TUM-VI dataset not found: {sequence_dir}")
        
        # Convert dataset using the new convert command
        cmd = [
            "./run.sh", "convert",
            "tumvi",
            str(sequence_dir),
            str(output_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"Converted dataset: {output_file}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to convert TUM-VI dataset {sequence}: {e.stderr}")
            raise
        
        return output_file
    
    def _download_tumvi_dataset(self, sequence: str, base_path: Path):
        """Download a TUM-VI dataset."""
        self.logger.info(f"Downloading TUM-VI dataset: {sequence}")
        base_path.mkdir(parents=True, exist_ok=True)
        
        # TUM-VI dataset URLs
        urls = {
            "mocap-desk": "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
            "mocap-desk2": "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz",
            "running-easy": "https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam.tgz",
            "running-hard": "https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam2.tgz"
        }
        
        if sequence not in urls:
            raise ValueError(f"Unknown TUM-VI sequence: {sequence}")
        
        # Download and extract
        import urllib.request
        import tarfile
        
        url = urls[sequence]
        tar_path = base_path / f"{sequence}.tgz"
        
        self.logger.info(f"Downloading from {url}")
        urllib.request.urlretrieve(url, tar_path)
        
        self.logger.info(f"Extracting to {base_path}")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(base_path)
        
        tar_path.unlink()  # Remove tar file
        self.logger.info(f"Downloaded and extracted {sequence}")
    
    def _run_estimators(self, datasets: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
        """
        Run all enabled estimators on all datasets.
        
        Args:
            datasets: Dictionary of dataset paths
            
        Returns:
            Nested dictionary of results [dataset][estimator]
        """
        self.logger.info("Running estimators on all datasets...")
        results = {}
        
        # Get enabled estimators
        enabled_estimators = [
            name for name, config in self.config['estimators'].items()
            if config.get('enabled', True)
        ]
        
        total_runs = len(datasets) * len(enabled_estimators)
        parallel_jobs = self.config['evaluation'].get('parallel_jobs', 1)
        
        with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
            # Submit all jobs
            futures = {}
            for dataset_name, dataset_path in datasets.items():
                for estimator_name in enabled_estimators:
                    future = executor.submit(
                        self._run_single_estimation,
                        dataset_name,
                        dataset_path,
                        estimator_name,
                        self.config['estimators'][estimator_name]
                    )
                    futures[future] = (dataset_name, estimator_name)
            
            # Process results
            with tqdm(total=total_runs, desc="Running estimations") as pbar:
                for future in as_completed(futures):
                    dataset_name, estimator_name = futures[future]
                    try:
                        result = future.result()
                        if dataset_name not in results:
                            results[dataset_name] = {}
                        results[dataset_name][estimator_name] = result
                        pbar.update(1)
                    except Exception as e:
                        import traceback
                        self.logger.error(f"Failed {estimator_name} on {dataset_name}: {e}")
                        self.logger.debug(f"Traceback: {traceback.format_exc()}")
                        pbar.update(1)
        
        return results
    
    def _run_single_estimation(self, dataset_name: str, dataset_path: Path,
                              estimator_name: str, estimator_config: Dict) -> Dict:
        """
        Run a single estimator on a dataset.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to dataset file
            estimator_name: Name of the estimator
            estimator_config: Estimator configuration
            
        Returns:
            Estimation results
        """
        output_dir = self.output_dir / "estimations" / dataset_name / estimator_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "result.json"
        
        # Build estimation command
        cmd = [
            "./run.sh", "slam",
            estimator_name,
            "--input", str(dataset_path),
            "--output", str(output_file)
        ]
        
        # Add configuration parameters
        config_file = output_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(estimator_config['config'], f)
        cmd.extend(["--config", str(config_file)])
        
        # Run estimation
        timeout = self.config['resources'].get('timeout_minutes', 30) * 60
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout
            )
            
            # The slam command creates a directory with timestamped file
            # Check if output_file exists as file or directory
            actual_output_file = None
            if output_file.exists() and output_file.is_file():
                actual_output_file = output_file
            elif output_file.exists() and output_file.is_dir():
                json_files = list(output_file.glob("*.json"))
                if json_files:
                    actual_output_file = json_files[0]  # Get the first (should be only) JSON file
            elif output_file.with_suffix('').exists() and output_file.with_suffix('').is_dir():
                # Check if directory without .json extension exists
                json_files = list(output_file.with_suffix('').glob("*.json"))
                if json_files:
                    actual_output_file = json_files[0]
            
            if not actual_output_file:
                raise FileNotFoundError(f"No output file found at {output_file}")
            
            # Load results
            with open(actual_output_file, 'r') as f:
                estimation_result = json.load(f)
            
            # Add performance metrics from stdout if available
            if "ATE RMSE" in result.stdout:
                # Parse metrics from output
                import re
                ate_match = re.search(r'ATE RMSE\s+│\s+([\d.]+)', result.stdout)
                rpe_match = re.search(r'RPE RMSE\s+│\s+([\d.]+)', result.stdout)
                time_match = re.search(r'Total Time\s+│\s+([\d.]+)', result.stdout)
                
                if 'metrics' not in estimation_result:
                    estimation_result['metrics'] = {}
                
                if ate_match:
                    estimation_result['metrics']['ate_rmse'] = float(ate_match.group(1))
                if rpe_match:
                    estimation_result['metrics']['rpe_trans_rmse'] = float(rpe_match.group(1))
                if time_match:
                    estimation_result['runtime_ms'] = float(time_match.group(1)) * 1000
                    
                estimation_result['converged'] = 'Converged' in result.stdout
            
            return estimation_result
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout: {estimator_name} on {dataset_name}")
            return {'status': 'timeout'}
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error: {estimator_name} on {dataset_name}: {e.stderr if e.stderr else e}")
            return {'status': 'error', 'error': str(e), 'stderr': e.stderr, 'stdout': e.stdout}
        except Exception as e:
            self.logger.error(f"Unexpected error: {estimator_name} on {dataset_name}: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'error': str(e)}
    
    def _compute_comparisons(self, estimation_results: Dict) -> Dict:
        """
        Compute comparison metrics for all results.
        
        Args:
            estimation_results: All estimation results
            
        Returns:
            Comparison results for each dataset
        """
        self.logger.info("Computing comparison metrics...")
        comparisons = {}
        
        for dataset_name, dataset_results in estimation_results.items():
            if not dataset_results:
                continue
            
            # Create comparison result from the estimator results
            # For now, create a simple comparison (would need proper implementation)
            comparison_result = self._create_comparison_result(dataset_results)
            comparisons[dataset_name] = comparison_result
            
            # Log summary
            if comparison_result.best_estimator:
                self.logger.info(
                    f"{dataset_name}: Best estimator is {comparison_result.best_estimator} "
                    f"(ATE RMSE: {comparison_result.performances[comparison_result.best_estimator].trajectory_metrics.ate_rmse:.3f})"
                )
        
        return comparisons
    
    def _extract_kpis(self, comparison_results: Dict) -> Dict:
        """
        Extract key performance indicators from results.
        
        Args:
            comparison_results: Comparison results for all datasets
            
        Returns:
            Dictionary of KPIs
        """
        self.logger.info("Extracting KPIs...")
        kpis = {}
        
        kpi_names = self.config['evaluation'].get('kpis', [
            'ate_rmse', 'rpe_translation_rmse', 'runtime_ms',
            'peak_memory_mb', 'convergence_rate'
        ])
        
        # Aggregate KPIs across datasets
        for kpi in kpi_names:
            kpis[kpi] = {}
            
            for dataset_name, comparison in comparison_results.items():
                if not comparison or not comparison.performances:
                    continue
                    
                for estimator_name, performance in comparison.performances.items():
                    if estimator_name not in kpis[kpi]:
                        kpis[kpi][estimator_name] = []
                    
                    # Extract KPI value
                    if kpi == 'ate_rmse':
                        value = performance.trajectory_metrics.ate_rmse
                    elif kpi == 'rpe_translation_rmse':
                        value = performance.trajectory_metrics.rpe_trans_rmse
                    elif kpi == 'runtime_ms':
                        value = performance.runtime_ms
                    elif kpi == 'peak_memory_mb':
                        value = performance.peak_memory_mb
                    elif kpi == 'convergence_rate':
                        value = 1.0 if performance.converged else 0.0
                    else:
                        value = None
                    
                    if value is not None:
                        kpis[kpi][estimator_name].append({
                            'dataset': dataset_name,
                            'value': value
                        })
        
        # Compute summary statistics
        kpi_summary = {}
        for kpi_name, estimator_data in kpis.items():
            kpi_summary[kpi_name] = {}
            for estimator, values in estimator_data.items():
                if values:
                    vals = [v['value'] for v in values]
                    kpi_summary[kpi_name][estimator] = {
                        'mean': np.mean(vals),
                        'std': np.std(vals),
                        'min': np.min(vals),
                        'max': np.max(vals),
                        'count': len(vals)
                    }
        
        return {
            'raw': kpis,
            'summary': kpi_summary
        }
    
    def _generate_dashboard(self, comparison_results: Dict, kpis: Dict) -> Path:
        """
        Generate the evaluation dashboard.
        
        Args:
            comparison_results: All comparison results
            kpis: Extracted KPIs
            
        Returns:
            Path to the generated dashboard
        """
        self.logger.info("Generating evaluation dashboard...")
        
        dashboard_dir = self.output_dir / "dashboard"
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate dashboard using the dashboard module
        dashboard_path = create_evaluation_dashboard(
            comparison_results,
            kpis,
            self.config['dashboard'],
            dashboard_dir
        )
        
        self.logger.info(f"Dashboard generated: {dashboard_path}")
        return dashboard_path
    
    def _save_results(self, comparison_results: Dict, kpis: Dict, dashboard_path: Path):
        """
        Save all results to disk.
        
        Args:
            comparison_results: All comparison results
            kpis: Extracted KPIs
            dashboard_path: Path to dashboard
        """
        self.logger.info("Saving results...")
        
        # Save summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'kpis': kpis,
            'dashboard': str(dashboard_path),
            'datasets': list(comparison_results.keys()),
            'estimators': list(self.config['estimators'].keys())
        }
        
        summary_file = self.output_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed results
        for dataset_name, comparison in comparison_results.items():
            if comparison:
                result_file = self.output_dir / "comparisons" / f"{dataset_name}.json"
                result_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert comparison to dict for saving
                comparison_dict = {
                    'best_estimator': comparison.best_estimator,
                    'performances': {}
                }
                
                for est_name, perf in comparison.performances.items():
                    comparison_dict['performances'][est_name] = {
                        'estimator_type': perf.estimator_type.name,
                        'runtime_ms': perf.runtime_ms,
                        'peak_memory_mb': perf.peak_memory_mb,
                        'converged': perf.converged,
                        'num_iterations': perf.num_iterations,
                        'ate_rmse': perf.trajectory_metrics.ate_rmse if perf.trajectory_metrics else 0,
                        'ate_mean': perf.trajectory_metrics.ate_mean if perf.trajectory_metrics else 0,
                        'rpe_trans_rmse': perf.trajectory_metrics.rpe_trans_rmse if perf.trajectory_metrics else 0
                    }
                
                with open(result_file, 'w') as f:
                    json.dump(comparison_dict, f, indent=2)
        
        # Create CSV summary for easy analysis
        self._create_csv_summary(comparison_results, kpis)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def _create_comparison_result(self, dataset_results: Dict) -> ComparisonResult:
        """Create a comparison result from estimator results."""
        performances = {}
        
        for estimator_name, result in dataset_results.items():
            if result.get('status') == 'error' or result.get('status') == 'timeout':
                continue
                
            # Create performance object
            traj_metrics = TrajectoryMetrics()
            if 'metrics' in result:
                traj_metrics.ate_rmse = result['metrics'].get('ate_rmse', 0.0)
                traj_metrics.ate_mean = result['metrics'].get('ate_mean', 0.0)
                traj_metrics.rpe_trans_rmse = result['metrics'].get('rpe_trans_rmse', 0.0)
            
            from src.evaluation.comparison import EstimatorPerformance
            from src.estimation.base_estimator import EstimatorType
            
            perf = EstimatorPerformance(
                estimator_type=getattr(EstimatorType, estimator_name.upper(), EstimatorType.EKF),
                runtime_ms=result.get('runtime_ms', 0.0),
                peak_memory_mb=result.get('memory_mb', 0.0),
                trajectory_metrics=traj_metrics,
                consistency_metrics=None,
                num_iterations=result.get('iterations', 1),
                converged=result.get('converged', True),
                metadata=result.get('metadata', {})
            )
            performances[estimator_name] = perf
        
        comparison = ComparisonResult(performances=performances)
        
        # Find best estimator
        if performances:
            best_ate = float('inf')
            for name, perf in performances.items():
                if perf.trajectory_metrics.ate_rmse < best_ate:
                    best_ate = perf.trajectory_metrics.ate_rmse
                    comparison.best_estimator = name
        
        return comparison
    
    def _create_csv_summary(self, comparison_results: Dict, kpis: Dict):
        """Create CSV summaries of results."""
        # Create performance matrix
        data = []
        for dataset_name, comparison in comparison_results.items():
            if not comparison or not comparison.performances:
                continue
                
            for estimator_name, performance in comparison.performances.items():
                row = {
                    'dataset': dataset_name,
                    'estimator': estimator_name,
                    'ate_rmse': performance.trajectory_metrics.ate_rmse,
                    'ate_mean': performance.trajectory_metrics.ate_mean,
                    'rpe_trans_rmse': performance.trajectory_metrics.rpe_trans_rmse,
                    'runtime_ms': performance.runtime_ms,
                    'memory_mb': performance.peak_memory_mb,
                    'converged': performance.converged
                }
                data.append(row)
        
        if data:
            df = pd.DataFrame(data)
            csv_file = self.output_dir / "performance_matrix.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Performance matrix saved to {csv_file}")


def run_evaluation(config_path: str) -> Dict:
    """
    Main entry point for running the evaluation.
    
    Args:
        config_path: Path to evaluation configuration
        
    Returns:
        Evaluation results
    """
    orchestrator = EvaluationOrchestrator(config_path)
    return orchestrator.run()
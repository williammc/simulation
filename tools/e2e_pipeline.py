"""
End-to-end (e2e) command implementation.
Runs comprehensive e2e pipeline across datasets and estimators.
"""

import os
import yaml
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional

from rich.console import Console

from tools.e2e_orchestrator import E2EOrchestrator

console = Console()


def run_e2e(
    config_file: Path,
    output_dir: Optional[Path] = None,
    parallel_jobs: Optional[int] = None,
    datasets: Optional[str] = None,
    estimators: Optional[str] = None,
    skip_generation: bool = False,
    skip_dashboard: bool = False,
    dry_run: bool = False
) -> int:
    """
    Run comprehensive end-to-end pipeline.
    
    Args:
        config_file: Path to e2e configuration YAML
        output_dir: Override output directory from config
        parallel_jobs: Number of parallel estimation jobs
        datasets: Comma-separated list of datasets to evaluate
        estimators: Comma-separated list of estimators to run
        skip_generation: Skip dataset generation even if missing
        skip_dashboard: Skip dashboard generation
        dry_run: Show what would be done without running
    
    Returns:
        Exit code (0 for success)
    """
    
    # Check if config file exists
    if not config_file.exists():
        console.print(f"[red]Error: Configuration file not found: {config_file}[/red]")
        console.print("\n[yellow]Tip: Use the default config at config/evaluation_config.yaml[/yellow]")
        console.print("[yellow]Or try the educational config: config/evaluation_educational.yaml[/yellow]")
        return 1
    
    # Load and potentially modify config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override settings from command line
    if output_dir:
        config['evaluation']['output_dir'] = str(output_dir)
    if parallel_jobs:
        config['evaluation']['parallel_jobs'] = parallel_jobs
    if skip_generation:
        if 'simulated' in config.get('datasets', {}):
            config['datasets']['simulated']['generate_if_missing'] = False
        if 'tum_vi' in config.get('datasets', {}):
            config['datasets']['tum_vi']['download_if_missing'] = False
    
    # Filter datasets if specified
    if datasets:
        dataset_list = [d.strip() for d in datasets.split(",")]
        # Filter simulated datasets
        if 'simulated' in config.get('datasets', {}):
            original_types = config['datasets']['simulated'].get('types', [])
            config['datasets']['simulated']['types'] = [
                d for d in original_types
                if d.get('name') in dataset_list
            ]
        # Filter TUM-VI sequences
        if 'tum_vi' in config.get('datasets', {}):
            original_seqs = config['datasets']['tum_vi'].get('sequences', [])
            config['datasets']['tum_vi']['sequences'] = [
                s for s in original_seqs
                if s.get('name') in dataset_list
            ]
    
    # Filter estimators if specified
    if estimators:
        estimator_list = [e.strip().lower() for e in estimators.split(",")]
        for est_name in config.get('estimators', {}).keys():
            if est_name not in estimator_list:
                config['estimators'][est_name]['enabled'] = False
    
    # Disable dashboard if requested
    if skip_dashboard and 'dashboard' in config:
        config['dashboard']['sections'] = []
    
    # Show configuration summary
    console.print("\n[bold]E2E Pipeline Configuration:[/bold]")
    console.print(f"  Config File: {config_file}")
    console.print(f"  Output Directory: {config['evaluation']['output_dir']}")
    console.print(f"  Parallel Jobs: {config['evaluation'].get('parallel_jobs', 1)}")
    
    # Count enabled items
    num_sim = len(config.get('datasets', {}).get('simulated', {}).get('types', []))
    num_tum = len(config.get('datasets', {}).get('tum_vi', {}).get('sequences', []))
    num_datasets = num_sim + num_tum
    
    enabled_estimators = [
        name for name, cfg in config.get('estimators', {}).items()
        if cfg.get('enabled', True)
    ]
    
    console.print(f"  Datasets: {num_datasets} ({num_sim} simulated, {num_tum} TUM-VI)")
    console.print(f"  Estimators: {len(enabled_estimators)} ({', '.join(enabled_estimators)})")
    console.print(f"  Total Runs: {num_datasets * len(enabled_estimators)}")
    
    # Check for auto-generation
    if config.get('datasets', {}).get('simulated', {}).get('generate_if_missing'):
        console.print("\n[cyan]Note: Missing datasets will be auto-generated[/cyan]")
        cache_dir = config['datasets']['simulated'].get('cache_dir', 'data/trajectories')
        console.print(f"  Cache directory: {cache_dir}")
    
    if dry_run:
        console.print("\n[yellow]Dry run mode - no actual execution[/yellow]")
        console.print("\n[bold]Would evaluate:[/bold]")
        
        # Show datasets
        console.print("\n[cyan]Simulated Datasets:[/cyan]")
        for dataset in config.get('datasets', {}).get('simulated', {}).get('types', []):
            console.print(f"  - {dataset.get('name', 'unnamed')}")
            if 'config' in dataset:
                console.print(f"    Config: {dataset['config']}")
        
        console.print("\n[cyan]TUM-VI Sequences:[/cyan]")
        for seq in config.get('datasets', {}).get('tum_vi', {}).get('sequences', []):
            console.print(f"  - {seq.get('name', 'unnamed')}")
        
        console.print("\n[cyan]Estimators:[/cyan]")
        for est in enabled_estimators:
            console.print(f"  - {est}")
        
        return 0
    
    # Run evaluation
    try:
        console.print("\n[green]Starting e2e pipeline...[/green]")
        
        # Save modified config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
        
        try:
            # Create orchestrator with modified config
            orchestrator = E2EOrchestrator(temp_config_path)
            
            # Run pipeline
            results = orchestrator.run()
            
            console.print("\n[green]✓ E2E pipeline complete![/green]")
            
            # Show dashboard if created
            if results and 'dashboard' in results:
                console.print(f"\n[bold]Dashboard:[/bold] {results['dashboard']}")
                
                # Optionally open in browser
                if not skip_dashboard:
                    webbrowser.open(f"file://{Path(results['dashboard']).absolute()}")
            
            # Show KPI summary
            if results and 'kpis' in results and 'summary' in results['kpis']:
                console.print("\n[bold]Top KPIs:[/bold]")
                kpi_summary = results['kpis']['summary']
                
                # Find best performer for each KPI
                for kpi_name, estimator_stats in kpi_summary.items():
                    if estimator_stats:
                        best_est = min(estimator_stats.items(), key=lambda x: x[1]['mean'])
                        console.print(f"  {kpi_name}: {best_est[0]} ({best_est[1]['mean']:.3f})")
            
            return 0
            
        finally:
            # Clean up temp config file
            if 'temp_config_path' in locals():
                try:
                    os.unlink(temp_config_path)
                except:
                    pass
        
    except Exception as e:
        console.print(f"\n[red]Error during e2e pipeline: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1
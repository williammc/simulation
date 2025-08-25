# Evaluation Module Documentation

## Overview

The evaluation module (`src/evaluation/`) provides comprehensive metrics and analysis tools for assessing SLAM algorithm performance. It implements standard metrics from the robotics community and custom evaluation criteria for specific use cases.

## Table of Contents
- [Architecture](#architecture)
- [Trajectory Metrics](#trajectory-metrics)
- [Landmark Metrics](#landmark-metrics)
- [Performance Metrics](#performance-metrics)
- [Statistical Analysis](#statistical-analysis)
- [Benchmarking Framework](#benchmarking-framework)
- [Usage Examples](#usage-examples)

## Architecture

```
src/evaluation/
├── __init__.py
├── trajectory_evaluator.py   # Trajectory error metrics
├── landmark_evaluator.py     # Map quality assessment
├── metrics.py               # Core metric implementations
├── statistical_analysis.py  # Statistical tools
├── benchmark.py             # Benchmarking framework
└── report_generator.py      # Evaluation reports
```

## Trajectory Metrics

### Absolute Trajectory Error (ATE)

ATE measures the absolute difference between estimated and ground truth trajectories after alignment.

```python
from src.evaluation import TrajectoryEvaluator

evaluator = TrajectoryEvaluator()

# Compute ATE
ate = evaluator.compute_ate(
    estimated_trajectory,
    ground_truth_trajectory,
    align=True,  # Perform SE3 alignment
    alignment_type='umeyama'  # or 'horn', 'svd'
)

print(f"ATE RMSE: {ate.rmse:.3f} m")
print(f"ATE Mean: {ate.mean:.3f} m")
print(f"ATE Median: {ate.median:.3f} m")
print(f"ATE Std: {ate.std:.3f} m")
print(f"ATE Max: {ate.max:.3f} m")
```

### Relative Pose Error (RPE)

RPE measures drift by comparing relative motion between consecutive poses.

```python
# Compute RPE
rpe = evaluator.compute_rpe(
    estimated_trajectory,
    ground_truth_trajectory,
    delta=1.0,  # 1 meter segments
    delta_unit='distance'  # or 'time', 'frames'
)

# Separate translation and rotation
rpe_trans = rpe.translation  # meters
rpe_rot = rpe.rotation      # radians

print(f"RPE Translation RMSE: {rpe_trans.rmse:.3f} m")
print(f"RPE Rotation RMSE: {np.degrees(rpe_rot.rmse):.2f} deg")
```

### Trajectory Alignment

```python
# Align trajectories before comparison
aligned_trajectory, transform = evaluator.align_trajectories(
    estimated_trajectory,
    ground_truth_trajectory,
    method='umeyama'
)

# Get alignment transformation
print(f"Scale: {transform.scale:.3f}")
print(f"Translation: {transform.translation}")
print(f"Rotation: {transform.rotation}")
```

### Custom Metrics

```python
# Define custom metric
def smoothness_metric(trajectory):
    """Compute trajectory smoothness."""
    velocities = np.diff(trajectory.positions, axis=0)
    accelerations = np.diff(velocities, axis=0)
    jerk = np.diff(accelerations, axis=0)
    return np.mean(np.linalg.norm(jerk, axis=1))

# Apply custom metric
smoothness = evaluator.compute_custom_metric(
    trajectory,
    smoothness_metric
)
```

## Landmark Metrics

### Map Quality Assessment

```python
from src.evaluation import LandmarkEvaluator

landmark_eval = LandmarkEvaluator()

# Evaluate landmark accuracy
landmark_metrics = landmark_eval.evaluate(
    estimated_landmarks,
    ground_truth_landmarks,
    association_threshold=0.5  # meters
)

print(f"Correctly associated: {landmark_metrics.precision:.2%}")
print(f"Landmarks found: {landmark_metrics.recall:.2%}")
print(f"Mean position error: {landmark_metrics.mean_error:.3f} m")
```

### Map Consistency

```python
# Check map consistency
consistency = landmark_eval.check_consistency(
    landmarks,
    observations,
    poses
)

print(f"Reprojection error: {consistency.reprojection_error:.2f} pixels")
print(f"Inconsistent landmarks: {consistency.inconsistent_count}")
```

### Coverage Analysis

```python
# Analyze spatial coverage
coverage = landmark_eval.compute_coverage(
    landmarks,
    bounds=[[-10, 10], [-10, 10], [0, 5]]
)

print(f"Coverage ratio: {coverage.ratio:.2%}")
print(f"Density: {coverage.density:.2f} landmarks/m³")
print(f"Uniformity score: {coverage.uniformity:.3f}")
```

## Performance Metrics

### Computational Performance

```python
from src.evaluation import PerformanceEvaluator

perf_eval = PerformanceEvaluator()

# Measure runtime performance
performance = perf_eval.evaluate_performance(
    estimator,
    test_data,
    metrics=['runtime', 'memory', 'cpu']
)

print(f"Average frame time: {performance.avg_frame_time:.2f} ms")
print(f"Peak memory: {performance.peak_memory:.1f} MB")
print(f"CPU utilization: {performance.cpu_usage:.1f}%")
```

### Real-time Capability

```python
# Check real-time constraints
realtime = perf_eval.check_realtime_capability(
    processing_times,
    sensor_rate=30.0  # Hz
)

print(f"Real-time ratio: {realtime.ratio:.2f}")
print(f"Dropped frames: {realtime.dropped_frames}")
print(f"Max latency: {realtime.max_latency:.2f} ms")
```

### Scalability Analysis

```python
# Test scalability
scalability = perf_eval.analyze_scalability(
    estimator,
    dataset_sizes=[100, 500, 1000, 5000]
)

# Plot complexity
scalability.plot_complexity()
print(f"Complexity: O(n^{scalability.complexity_exponent:.2f})")
```

## Statistical Analysis

### Error Distribution Analysis

```python
from src.evaluation import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Analyze error distribution
distribution = analyzer.analyze_distribution(errors)

print(f"Distribution type: {distribution.best_fit}")
print(f"Normality test p-value: {distribution.normality_pvalue:.4f}")
print(f"Skewness: {distribution.skewness:.3f}")
print(f"Kurtosis: {distribution.kurtosis:.3f}")

# Confidence intervals
ci_95 = distribution.confidence_interval(0.95)
print(f"95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
```

### Hypothesis Testing

```python
# Compare two algorithms
comparison = analyzer.compare_algorithms(
    algorithm1_errors,
    algorithm2_errors,
    test='wilcoxon'  # or 't-test', 'mann-whitney'
)

print(f"P-value: {comparison.pvalue:.4f}")
print(f"Effect size: {comparison.effect_size:.3f}")
print(f"Significant: {comparison.is_significant}")
```

### Monte Carlo Analysis

```python
# Monte Carlo evaluation
monte_carlo = analyzer.monte_carlo_evaluation(
    estimator,
    simulation_params,
    n_runs=100
)

print(f"Success rate: {monte_carlo.success_rate:.2%}")
print(f"Mean error: {monte_carlo.mean_error:.3f} ± {monte_carlo.std_error:.3f}")
print(f"Worst case: {monte_carlo.worst_case:.3f}")
```

## Benchmarking Framework

### Standard Benchmarks

```python
from src.evaluation import Benchmark

# Run standard benchmark suite
benchmark = Benchmark()

results = benchmark.run_standard_suite(
    estimator,
    datasets=['EuRoC', 'TUM', 'KITTI']
)

# Generate report
benchmark.generate_report(results, 'benchmark_report.html')
```

### Custom Benchmark

```python
# Define custom benchmark
class CustomBenchmark(Benchmark):
    def __init__(self):
        super().__init__()
        self.metrics = ['ate', 'rpe', 'runtime', 'memory']
        
    def evaluate_scenario(self, estimator, scenario):
        """Evaluate specific scenario."""
        result = estimator.process(scenario.data)
        
        return {
            'ate': self.compute_ate(result, scenario.ground_truth),
            'rpe': self.compute_rpe(result, scenario.ground_truth),
            'runtime': result.runtime,
            'memory': result.peak_memory
        }

# Run custom benchmark
custom_bench = CustomBenchmark()
results = custom_bench.run(estimator, test_scenarios)
```

### Comparative Benchmarking

```python
# Compare multiple algorithms
algorithms = {
    'EKF': ekf_estimator,
    'GTSAM': gtsam_estimator,
    'SWBA': swba_estimator
}

comparison = benchmark.compare_algorithms(
    algorithms,
    test_dataset,
    metrics=['accuracy', 'speed', 'robustness']
)

# Generate comparison table
comparison.print_table()
comparison.plot_radar_chart()
comparison.save_latex_table('comparison.tex')
```

## Usage Examples

### Complete Evaluation Pipeline

```python
from src.evaluation import EvaluationPipeline

# Create evaluation pipeline
pipeline = EvaluationPipeline()

# Configure metrics
pipeline.add_metric('ate', weight=0.3)
pipeline.add_metric('rpe', weight=0.3)
pipeline.add_metric('runtime', weight=0.2)
pipeline.add_metric('landmark_accuracy', weight=0.2)

# Run evaluation
results = pipeline.evaluate(
    estimation_result,
    ground_truth,
    save_report=True
)

# Get overall score
score = results.weighted_score
print(f"Overall score: {score:.2f}/100")
```

### Robustness Testing

```python
from src.evaluation import RobustnessEvaluator

robustness_eval = RobustnessEvaluator()

# Test with noise perturbations
noise_robustness = robustness_eval.test_noise_robustness(
    estimator,
    base_data,
    noise_levels=[0.01, 0.05, 0.1, 0.2]
)

# Test with outliers
outlier_robustness = robustness_eval.test_outlier_robustness(
    estimator,
    base_data,
    outlier_ratios=[0.05, 0.1, 0.2]
)

# Test with missing data
dropout_robustness = robustness_eval.test_dropout_robustness(
    estimator,
    base_data,
    dropout_rates=[0.1, 0.2, 0.3]
)

# Generate robustness report
robustness_eval.generate_report(
    'robustness_analysis.pdf'
)
```

### Cross-Validation

```python
from src.evaluation import CrossValidator

# K-fold cross-validation
cv = CrossValidator(n_folds=5)

cv_results = cv.evaluate(
    estimator,
    dataset,
    metrics=['ate', 'rpe']
)

print(f"Mean ATE: {cv_results.ate_mean:.3f} ± {cv_results.ate_std:.3f}")
print(f"Mean RPE: {cv_results.rpe_mean:.3f} ± {cv_results.rpe_std:.3f}")
```

## Evaluation Reports

### Report Generation

```python
from src.evaluation import ReportGenerator

report_gen = ReportGenerator()

# Configure report
report_gen.set_title("SLAM Evaluation Report")
report_gen.add_section("Trajectory Analysis", trajectory_results)
report_gen.add_section("Landmark Analysis", landmark_results)
report_gen.add_section("Performance", performance_results)

# Add visualizations
report_gen.add_plot(trajectory_plot, "Trajectory Comparison")
report_gen.add_plot(error_plot, "Error Distribution")

# Generate report
report_gen.generate(
    format='html',  # or 'pdf', 'latex'
    output_path='evaluation_report.html',
    include_raw_data=True
)
```

### LaTeX Export

```python
# Export results for paper
latex_exporter = LatexExporter()

# Create results table
table = latex_exporter.create_table(
    results,
    caption="SLAM algorithm comparison",
    label="tab:results"
)

# Export plots
latex_exporter.export_figure(
    plot,
    'trajectory_comparison.pdf',
    caption="Trajectory comparison",
    label="fig:trajectory"
)

# Generate complete LaTeX section
latex_exporter.generate_section(
    'evaluation_section.tex',
    results,
    plots
)
```

## Best Practices

### Metric Selection

1. **Choose appropriate metrics**
   - ATE for global accuracy
   - RPE for drift assessment
   - Runtime for real-time capability

2. **Consider application requirements**
   - Accuracy vs. speed trade-offs
   - Memory constraints
   - Robustness needs

### Fair Comparison

1. **Ensure fair testing**
   - Same hardware/environment
   - Identical test data
   - Multiple runs for statistics

2. **Account for randomness**
   - Set random seeds
   - Report confidence intervals
   - Use statistical tests

### Reproducibility

1. **Document everything**
   - Configuration files
   - Software versions
   - Hardware specifications

2. **Share evaluation code**
   - Provide scripts
   - Include test data
   - Document procedures

## Troubleshooting

### Common Issues

1. **Trajectory Association**
   - Time synchronization issues
   - Different coordinate frames
   - Solution: Use interpolation and frame transformation

2. **Metric Instability**
   - Outliers affecting statistics
   - Solution: Use robust metrics (median, trimmed mean)

3. **Computational Cost**
   - Large-scale evaluation slow
   - Solution: Use sampling or parallel processing

## References

- ATE/RPE Metrics: "A Benchmark for the Evaluation of RGB-D SLAM Systems" (Sturm et al.)
- Statistical Analysis: "The Elements of Statistical Learning" (Hastie et al.)
- Benchmarking: "ORB-SLAM2: an Open-Source SLAM System" (Mur-Artal & Tardós)
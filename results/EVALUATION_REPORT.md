# EVALUATION REPORT

## Executive Summary
- Best portrait extraction: `portrait_hq` (faces required, portrait filter, blur threshold 120).
- GPU idle ratio: 73.8%
- Quality improvement: person-mode filters remove non-face frames vs baseline.

## Quantitative Results
See `results/comparison.md`.

## Quality Assessment
Portrait issues: 6
YOLO vs Haar faces detected: 10 vs 8

## Performance Analysis
See `results/performance.json` and `results/gpu_stats.csv`.

## Recommendations
- Use `--person-mode --require-faces --pose-filter portrait --blur-threshold 120 --top-n 5` for portraits.
- If GPU utilization stays below 70%, increase batch size or use faster face model.
- Consider caching face detections for re-runs.

## TODO
- [ ] Optimize slowest operation (if >2s per video).
- [ ] Fine-tune composite score weights based on output quality.
- [ ] Add resume capability if batch processing interrupted.
- [ ] Consider caching face detection results for re-runs.
- [ ] Implement smart interval (extract more frames near detected faces).
# ssl_53_simclr_grid_r2_step

Same architecture and task as experiment 53, but now using step based metrics instead of epoch based metrics. The previous experiment showed the larger models beginning to overfit, and so exposing the model to more new data (which there is enough to train for 48 hours on) could produce better top1 probe accuracies. Summary of changes implemented in this experiment in comparison to exp 52:
1. full epoch (ds_limit=0)
2. step based metrics 
    - save an artifact capapable of being used to finetune every 10k steps 
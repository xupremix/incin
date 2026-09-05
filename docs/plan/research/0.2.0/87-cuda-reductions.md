#87 CUDA reductions/order stats (14 ops) — L
Finding: only sum/mean/max/min execute; missing argmax argmin argsort cumsum norm prod_all prod_dim std_all/dim/keepdim topk var_all/dim/keepdim. render_cuda_reduction accepts 4 ops; indexed path rejects fast warp-shuffle layout; launch_reduce_op generalizes axis/keepdim + f32 accum for halves.
Recommendation: extend (init,update,finish) templates (prod; var/std Welford; norm pow-accumulate; argmax/argmin/topk value+index composites with low-index tie rule); cumsum = separate work-efficient scan kernel; write tie-breaking + argsort stability contract into OPERATION_SEMANTICS first; widen CPU topk past f32 simultaneously.
Risk: tie agreement across backends; Welford mandatory for halves; NaN semantics must match CPU partial_cmp.

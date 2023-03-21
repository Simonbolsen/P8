from main import *

with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/main_profiler'),
    record_shapes=True
) as prof:
    if __name__ == '__main__':
        args = argparser.parse_args()
        run_main(args)


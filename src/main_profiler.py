from main import *


if __name__ == '__main__':
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/main_profiler'),
        record_shapes=True
    ) as prof:

        args = argparser.parse_args()
        run_main(args)
        prof.step()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1), file=open("./log/profiler.txt", "a"))

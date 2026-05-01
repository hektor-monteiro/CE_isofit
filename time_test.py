import cProfile, pstats
profiler = cProfile.Profile()
profiler.enable()

try:
    with open("OCFit-9.0.py") as f:
        code = compile(f.read(), "OCFit-9.0.py", 'exec')
        exec(code)
except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"Exception: {e}")

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(50)

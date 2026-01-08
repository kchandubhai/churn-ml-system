def estimate_task(task_name):
    # A "dummy" ML model: longer words = more time
    hours = len(task_name) / 2
    return f"Task '{task_name}' will take about {hours} hours."

print(estimate_task("Build MLOps Pipeline"))
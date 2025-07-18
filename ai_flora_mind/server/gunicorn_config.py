"""Gunicorn configuration for AI Flora Mind API."""

# Gunicorn basic settings, we can adjust them as we need.
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
loglevel = "info"

# Additional production settings
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
keepalive = 5
preload_app = True
reload = False

# Logging configuration
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
accesslog = "-"
errorlog = "-"
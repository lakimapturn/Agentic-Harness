"""Redis connection and RQ queue construction."""

import os

from redis import Redis
from rq import Queue
from rq.serializers import JSONSerializer

QUEUE_NAME = "harness-tasks"

def redis_connection(url: str | None = None):
    return Redis.from_url(url or os.getenv("HARNESS_REDIS_URL", "redis://localhost:6379/0"))


def task_queue(connection=None):
    return Queue(QUEUE_NAME, connection=connection or redis_connection(), serializer=JSONSerializer)



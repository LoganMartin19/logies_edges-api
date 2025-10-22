# api/app/utils/cache.py
import json, time, threading
from typing import Callable, Any, Optional

try:
    import redis
    _r = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
except Exception:
    _r = None

# in-proc fallback (process-local)
_fallback = {}
_lock = threading.Lock()

def _now() -> float:
    return time.time()

def get_json(key: str) -> Optional[Any]:
    if _r:
        val = _r.get(key)
        return json.loads(val) if val else None
    with _lock:
        row = _fallback.get(key)
        if not row: return None
        exp, val = row
        if exp and exp < _now():
            _fallback.pop(key, None)
            return None
        return val

def set_json(key: str, obj: Any, ttl_sec: int):
    if _r:
        _r.setex(key, ttl_sec, json.dumps(obj))
        return
    with _lock:
        _fallback[key] = (_now() + ttl_sec, obj)

def cache_json(key: str, ttl_sec: int, fetch_fn: Callable[[], Any]) -> Any:
    val = get_json(key)
    if val is not None:
        return val
    data = fetch_fn()
    set_json(key, data, ttl_sec)
    return data
"""Comparable collection identity; never infer missing collection bounds."""
import json
import hashlib

PROFILE_KEYS = ("sessionStart", "sessionEnd", "maxDte", "strikeRange", "greekInterval", "greekToleranceMin")
REQUIRED_KEYS = PROFILE_KEYS[:4]


def collection_profile(config):
    cfg = json.loads(config) if isinstance(config, str) else (config or {})
    return {key: cfg.get(key) for key in PROFILE_KEYS}


def profile_identity(methodology, config):
    profile = collection_profile(config)
    if not methodology or any(profile.get(key) is None for key in REQUIRED_KEYS):
        return None
    canonical = json.dumps({"methodology": methodology, "collection": profile}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()

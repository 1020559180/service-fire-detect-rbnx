#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""DJI bridge APK auto-discovery.

Does not depend on the rbnx framework: scan the common DJI RC Pro / RC-N1 hotspot subnets and find the first host where
``http://<ip>:<port>/api/status`` returns 200 (i.e. the bridge APK).

Used instead of the hard-coded ``172.20.10.3``: when the RC Pro is replaced by a phone/tablet or joins the home router
and the IP changes, it can still connect automatically.

Usage:
    from .discover import discover_apk
    ip = discover_apk()                 # -> "172.20.10.3" or None
    ip = discover_apk(port=8080, timeout=0.4)
"""
from __future__ import annotations

import concurrent.futures
from typing import List, Optional

import requests

# common DJI hotspot hosts (ordered by hit probability; try these first for a fast path)
_KNOWN_HOSTS = [
    "172.20.10.3", "172.20.10.2", "172.20.10.1",       # DJI RC Pro hotspot
    "192.168.42.2", "192.168.42.3", "192.168.42.1",    # DJI RC-N1 / some models
    "192.168.1.2", "192.168.1.3", "192.168.0.2", "192.168.0.3",
]

# common DJI subnets that need a full scan
_KNOWN_SUBNETS = ["172.20.10.0/24", "192.168.42.0/24"]

_cache: Optional[str] = None


def _probe(ip: str, port: int, timeout: float) -> Optional[str]:
    try:
        r = requests.get(f"http://{ip}:{port}/api/status", timeout=timeout)
        if r.status_code == 200:
            return ip
    except Exception:  # noqa: BLE001
        pass
    return None


def _subnet_hosts(cidr: str) -> List[str]:
    base = cidr.rsplit("/", 1)[0]
    octets = base.split(".")[:3]
    # the gateway / APK is usually at a low address; probe .3/.2/.1/.4-.10 first, then the rest
    order = [3, 2, 1, 4, 5, 6, 7, 8, 9, 10] + list(range(11, 255))
    return [f"{'.'.join(octets)}.{i}" for i in order]


def discover_apk(port: int = 8080, timeout: float = 0.4, use_cache: bool = True) -> Optional[str]:
    """Auto-discover the bridge APK's IP; return None on failure."""
    global _cache
    if use_cache and _cache:
        return _cache

    # 1) fast path: try the known hosts first
    for ip in _KNOWN_HOSTS:
        if _probe(ip, port, timeout):
            _cache = ip
            return ip

    # 2) full scan of the known subnets (concurrent, take the first hit by completion order)
    for cidr in _KNOWN_SUBNETS:
        ips = _subnet_hosts(cidr)
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            futures = {ex.submit(_probe, ip, port, timeout): ip for ip in ips}
            for fut in concurrent.futures.as_completed(futures):
                ip = fut.result()
                if ip:
                    _cache = ip
                    return ip
    return None


if __name__ == "__main__":
    print(discover_apk())

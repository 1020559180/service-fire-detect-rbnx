#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""DJI 桥接 APK 自动发现。

不依赖 rbnx 框架：扫描常见 DJI RC Pro / RC-N1 热点网段，找到第一个在
``http://<ip>:<port>/api/status`` 返回 200 的主机（即桥接 APK）。

用于替代硬编码的 ``172.20.10.3``：当 RC Pro 换成手机/平板、或接入家庭路由
导致 IP 变化时，仍能自动连上。

用法：
    from .discover import discover_apk
    ip = discover_apk()                 # -> "172.20.10.3" 或 None
    ip = discover_apk(port=8080, timeout=0.4)
"""
from __future__ import annotations

import concurrent.futures
from typing import List, Optional

import requests

# 常见 DJI 热点主机（按命中概率排序，先快速试这些）
_KNOWN_HOSTS = [
    "172.20.10.3", "172.20.10.2", "172.20.10.1",       # DJI RC Pro 热点
    "192.168.42.2", "192.168.42.3", "192.168.42.1",    # DJI RC-N1 / 部分机型
    "192.168.1.2", "192.168.1.3", "192.168.0.2", "192.168.0.3",
]

# 需要全扫的常见 DJI 网段
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
    # 网关/APK 常在低地址，先探 .3/.2/.1/.4-.10，再扫其余
    order = [3, 2, 1, 4, 5, 6, 7, 8, 9, 10] + list(range(11, 255))
    return [f"{'.'.join(octets)}.{i}" for i in order]


def discover_apk(port: int = 8080, timeout: float = 0.4, use_cache: bool = True) -> Optional[str]:
    """自动发现桥接 APK 的 IP；失败返回 None。"""
    global _cache
    if use_cache and _cache:
        return _cache

    # 1) 快速路径：先试已知主机
    for ip in _KNOWN_HOSTS:
        if _probe(ip, port, timeout):
            _cache = ip
            return ip

    # 2) 全扫已知网段（并发，按完成顺序取第一个命中）
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

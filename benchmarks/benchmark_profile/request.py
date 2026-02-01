"""
请求处理模块 - 调用 /profile_step API
"""
import aiohttp
from aiohttp import ClientTimeout
from typing import Any


# API端点配置
PROFILE_STEP_URL: str | None = None
PROFILE_STEP_BATCH_URL: str | None = None
RESET_PREFIX_CACHE_URL: str | None = None
HEADERS = {"Content-Type": "application/json"}


def init_url(host: str = "127.0.0.1", port: int = 8000):
    """
    初始化API端点URL

    Args:
        host: 服务器主机名
        port: 服务器端口
    """
    global PROFILE_STEP_URL, PROFILE_STEP_BATCH_URL, RESET_PREFIX_CACHE_URL
    base_url = f"http://{host}:{port}"
    PROFILE_STEP_URL = f"{base_url}/profile_step"
    PROFILE_STEP_BATCH_URL = f"{base_url}/profile_step_batch"
    RESET_PREFIX_CACHE_URL = f"{base_url}/reset_prefix_cache"


async def send_reset_prefix_cache() -> bool:
    """发送重置 prefix_cache 的请求"""
    timeout = ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(
                RESET_PREFIX_CACHE_URL,
                json=None,
                headers=HEADERS
            ) as response:
                if response.status == 200:
                    print("成功重置 prefix_cache")
                    return True
                else:
                    print(f"重置 prefix_cache 失败: {response.status}")
                    return False
        except Exception as e:
            print(f"重置 prefix_cache 错误: {e}")
            return False


async def send_profile_step(
    batch_size: int,
    compute_tokens: int,
    access_tokens: int,
    num_iterations: int = 20,
    warmup_iterations: int = 5,
    timeout_seconds: int = 600,
) -> dict[str, Any] | None:
    """
    调用 /profile_step API 测量稳态吞吐节拍

    Args:
        batch_size: 请求数量 (N)
        compute_tokens: 计算 token 数 (C)
        access_tokens: KV cache 中已有的 token 数 (A)
        num_iterations: 测量迭代次数
        warmup_iterations: 预热迭代次数
        timeout_seconds: 超时时间（秒）

    Returns:
        成功返回结果字典，失败返回 None
    """
    data = {
        "batch_size": batch_size,
        "compute_tokens": compute_tokens,
        "access_tokens": access_tokens,
        "num_iterations": num_iterations,
        "warmup_iterations": warmup_iterations,
    }

    timeout = ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(
                PROFILE_STEP_URL,
                json=data,
                headers=HEADERS
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    print(f"profile_step 失败: {response.status} - {error_text}")
                    return None
        except aiohttp.ClientError as e:
            print(f"profile_step 请求错误: {e}")
            return None
        except Exception as e:
            print(f"profile_step 未知错误: {e}")
            return None


async def send_profile_step_batch(
    samples: list[list[int]],
    num_iterations: int = 20,
    warmup_iterations: int = 5,
    timeout_seconds: int = 3600,
) -> list[dict[str, Any]] | None:
    """
    调用 /profile_step_batch API 批量测量稳态吞吐节拍

    Args:
        samples: [[B, C, A], ...] 列表
        num_iterations: 测量迭代次数
        warmup_iterations: 预热迭代次数
        timeout_seconds: 超时时间（秒）

    Returns:
        成功返回结果列表，失败返回 None
    """
    data = {
        "samples": samples,
        "num_iterations": num_iterations,
        "warmup_iterations": warmup_iterations,
    }

    timeout = ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(
                PROFILE_STEP_BATCH_URL,
                json=data,
                headers=HEADERS
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    print(f"profile_step_batch 失败: {response.status} - {error_text}")
                    return None
        except aiohttp.ClientError as e:
            print(f"profile_step_batch 请求错误: {e}")
            return None
        except Exception as e:
            print(f"profile_step_batch 未知错误: {e}")
            return None

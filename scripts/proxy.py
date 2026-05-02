"""
代理配置模块 - 在 Jupyter notebook 中导入使用

用法:
```python
from proxy import set_proxy, unset_proxy

# 设置代理
set_proxy()

# 取消代理
unset_proxy()
```
"""

import os


def set_proxy():
    """设置代理环境变量"""
    os.environ['http_proxy'] = 'http://localhost:8118/'
    os.environ['https_proxy'] = 'http://localhost:8118/'
    os.environ['HTTP_PROXY'] = 'http://localhost:8118/'
    os.environ['HTTPS_PROXY'] = 'http://localhost:8118/'
    os.environ['all_proxy'] = 'socks5://localhost:1080/'
    os.environ['ALL_PROXY'] = 'socks5://localhost:1080/'
    os.environ['no_proxy'] = 'localhost,127.0.0.0/8,192.168.1.0/24,::1,*.local'
    os.environ['NO_PROXY'] = 'localhost,127.0.0.0/8,192.168.1.0/24,::1,*.local'
    print("Proxy settings applied!")


def unset_proxy():
    """取消代理环境变量"""
    for key in list(os.environ.keys()):
        if 'proxy' in key.lower():
            del os.environ[key]
    print("Proxy settings removed!")


def show_proxy():
    """显示当前代理设置"""
    print("Current proxy settings:")
    for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
                'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY']:
        value = os.environ.get(key, 'Not set')
        print(f"  {key}={value}")


if __name == "__main__":
    set_proxy()

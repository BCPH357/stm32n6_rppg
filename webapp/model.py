"""
Webapp Model Wrapper
直接從 server_training 目錄 import UltraLightRPPG 模型

優點:
- 避免代碼重複
- 確保模型定義一致
- 簡化維護（只需更新 server_training/model.py）

使用方式:
    from model import UltraLightRPPG
    model = UltraLightRPPG(window_size=8, num_rois=3)
"""

import sys
import os

# 將 server_training 目錄加入 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
server_training_dir = os.path.join(parent_dir, 'server_training')

if server_training_dir not in sys.path:
    sys.path.insert(0, server_training_dir)

# 直接從 server_training 導入模型
# 使用 importlib 避免循環導入
import importlib.util
spec = importlib.util.spec_from_file_location(
    "server_training_model",
    os.path.join(server_training_dir, "model.py")
)
server_training_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_training_model)

# 導出 UltraLightRPPG
UltraLightRPPG = server_training_model.UltraLightRPPG

# Re-export 以保持 API 兼容性
__all__ = ['UltraLightRPPG']

# 為了向後兼容，也可以直接使用 from webapp.model import UltraLightRPPG
if __name__ == "__main__":
    # 測試 import 是否成功
    print("=" * 60)
    print("Testing Model Import from server_training")
    print("=" * 60)
    print(f"Server training dir: {server_training_dir}")
    print(f"Module imported from: {UltraLightRPPG.__module__}")

    # 測試模型實例化
    model = UltraLightRPPG(window_size=8, num_rois=3)
    print(f"Model class: {model.__class__.__name__}")
    print(f"Total parameters: {model.get_num_params():,}")
    print("\n[OK] Model import successful!")

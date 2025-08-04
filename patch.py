import subprocess
import os
import torch
from pathlib import Path

def patch_model():
    try:
        result = subprocess.run(['pip', 'show', 'basicsr'], capture_output=True, text=True, check=True)
        location_line = next(line for line in result.stdout.split('\n') if line.startswith('Location:'))
        basics_path = Path(location_line.split(' ')[1].strip()) / 'basicsr'
        # basics_path = Path('/usr/local/python3.8/lib/python3.8/site-packages/basicsr')
        
        # 修改 base_model.py
        base_model_path = basics_path / 'models' / 'base_model.py'
        if base_model_path.exists():
            with open(base_model_path, 'r') as f:
                content = f.read()
            
            # 添加 import torch_npu
            content = content.replace('import torch', 'import torch\nimport torch_npu')
            
            # 修改设备选择逻辑
            old_code = "self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')"
            new_code = """if opt['num_gpu'] != 0 and hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.init()
            self.device = torch.device('npu')
            print(f"Using NPU: {torch.npu.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")"""
            
            content = content.replace(old_code, new_code)
            
            with open(base_model_path, 'w') as f:
                f.write(content)
        
        # 修改degradations
        base_model_path = basics_path / 'data' / 'degradations.py'
        if base_model_path.exists():
            with open(base_model_path, 'r') as f:
                content = f.read()
            # 添加 import torch_npu
            content = content.replace('functional_tensor', '_functional_tensor')
            with open(base_model_path, 'w') as f:
                f.write(content)
        print("Model patching completed successfully")
    except Exception as e:
        print(f"Error patching model: {str(e)}")
        raise

if __name__ == "__main__":
    patch_model()

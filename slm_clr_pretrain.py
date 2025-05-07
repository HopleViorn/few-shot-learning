import argparse
import os
from trainer import SLM_CLR_Trainer, DEFAULT_CONFIG
import torch
def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot Learning Training Pipeline")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--eval', action='store_true', help='Run evaluation only')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model for evaluation')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 打印设备信息
    print(f"\n=== 设备信息 ===")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
    
    if not args.eval:
        print(f"\nStarting training for {args.epochs} epochs...")
        config = DEFAULT_CONFIG.copy()
        config.update({
            'batch_size': args.batch_size,
            'lr': args.lr
        })
        trainer = SLM_CLR_Trainer(config)
        trainer.train(args.epochs)

if __name__ == "__main__":
    main()
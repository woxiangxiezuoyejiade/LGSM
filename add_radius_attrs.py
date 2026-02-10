

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.loader_downstream import add_radius_attrs_to_file, add_radius_attrs_to_datasets

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='给.pt文件添加radius_edge_index和radius_edge_weight')
    parser.add_argument('--file', type=str, help='.pt文件路径')
    parser.add_argument('--dataset', type=str, nargs='+', help='数据集名称（可指定多个，用空格分隔）')
    parser.add_argument('--all_datasets', action='store_true', help='处理data_dir下的所有数据集')
    parser.add_argument('--split', type=str, default='train', help='split名称（train/valid/test），仅当指定--file时使用')
    parser.add_argument('--data_dir', type=str, default='./data/chem_dataset', help='数据目录')
    parser.add_argument('--cutoff', type=float, default=None, help='cutoff值（默认从config读取）')
    
    args = parser.parse_args()
    
    # 确定文件路径
    if args.file:
        file_path = args.file
        add_radius_attrs_to_file(file_path, args.cutoff)
    elif args.all_datasets:
        # 处理所有数据集
        add_radius_attrs_to_datasets(datasets=None, data_dir=args.data_dir, cutoff=args.cutoff)
    elif args.dataset:
        # 处理指定的数据集（可以是多个）
        add_radius_attrs_to_datasets(datasets=args.dataset, data_dir=args.data_dir, cutoff=args.cutoff)
    else:
        print("必须指定以下选项之一:")
        print("  --file: 处理单个文件")
        print("  --dataset: 处理指定的数据集（可指定多个）")
        print("  --all_datasets: 处理所有数据集")
        parser.print_help()


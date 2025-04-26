import os
import subprocess
from pathlib import Path
from datetime import datetime
import json

def run_python_file(file_path):
    """运行单个Python文件并返回其输出"""
    try:
        result = subprocess.run(['python', file_path], 
                               capture_output=True, 
                               text=True, 
                               timeout=30)  # 设置30秒超时
        return {
            'file': os.path.basename(file_path),
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr if result.returncode != 0 else None
        }
    except subprocess.TimeoutExpired:
        return {
            'file': os.path.basename(file_path),
            'success': False,
            'output': None,
            'error': '执行超时'
        }
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'success': False,
            'output': None,
            'error': str(e)
        }

def save_results_to_file(results, output_file=None):
    """将结果保存到文件中"""
    if output_file is None:
        # 如果没有指定输出文件，使用当前时间创建一个文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"logs/python_programs_results_{timestamp}.json"
    
    # 创建一个可以序列化为JSON的结果字典
    serializable_results = []
    for result in results:
        # 创建一个新的字典，只包含可以序列化的字段
        serializable_result = {
            'file': result['file'],
            'success': result['success'],
            'output': result['output'],
            'error': result['error']
        }
        serializable_results.append(serializable_result)
    
    # 添加汇总信息
    summary = {
        'total': len(results),
        'success': sum(1 for r in results if r['success']),
        'failure': sum(1 for r in results if not r['success']),
        'timestamp': datetime.now().isoformat()
    }
    
    # 创建最终的结果字典
    final_results = {
        'results': serializable_results,
        'summary': summary
    }
    
    # 将结果保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到文件: {output_file}")
    return output_file



def main():
    """主函数：遍历目录并运行所有Python文件"""
    # 获取python_programs目录的绝对路径
    programs_dir = Path(__file__).parent.parent / 'python_programs'
    
    # 确保目录存在
    if not programs_dir.exists() or not programs_dir.is_dir():
        print(f"错误：目录 {programs_dir} 不存在或不是一个目录")
        return
    
    # 获取所有.py文件
    python_files = sorted(programs_dir.glob('*.py'))
    
    if not python_files:
        print(f"在 {programs_dir} 中没有找到Python文件")
        return
    
    print(f"找到 {len(python_files)} 个Python文件，开始执行...\n")
    
    # 运行每个文件并收集结果
    results = []
    for file_path in python_files:
        print(f"正在运行 {file_path.name}...")
        result = run_python_file(file_path)
        results.append(result)
        
        # 打印当前文件的结果
        print(f"文件: {result['file']}")
        print(f"状态: {'成功' if result['success'] else '失败'}")
        if result['success']:
            print(f"输出:\n{result['output']}")
        else:
            print(f"错误:\n{result['error']}")
        print("-" * 80)
    
    # 打印汇总信息
    success_count = sum(1 for r in results if r['success'])
    print(f"\n执行完成: {success_count}/{len(results)} 个文件成功执行")

    # 保存结果到文件
    output_file = save_results_to_file(results)
    print(f"详细结果已保存到: {output_file}")


# def run_command(command):
#     """执行命令并记录输出"""
#     logging.info(f"执行命令: {command}")
#     try:
#         process = subprocess.Popen(
#             command,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             shell=True,
#             universal_newlines=True
#         )
        
#         # 实时输出命令执行结果
#         while True:
#             stdout_line = process.stdout.readline()
#             stderr_line = process.stderr.readline()
            
#             if stdout_line == '' and stderr_line == '' and process.poll() is not None:
#                 break
                
#             if stdout_line:
#                 logging.info(stdout_line.strip())
                
#             if stderr_line:
#                 logging.error(stderr_line.strip())
                
#         return_code = process.poll()
        
#         if return_code != 0:
#             logging.error(f"命令执行失败，返回码: {return_code}")
#             return False
            
#         logging.info(f"命令执行成功，返回码: {return_code}")
#         return True
#     except Exception as e:
#         logging.error(f"执行命令时发生错误: {str(e)}")
#         return False


if __name__ == "__main__":
    main()

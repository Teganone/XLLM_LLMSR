import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

class LoggingUtils:
    """
    日志工具类，提供统一的日志记录功能
    """
    
    # 日志格式
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 日志级别映射
    LOG_LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    # 日志实例缓存
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def setup_logger(cls, 
                    name: str, 
                    level: str = "info", 
                    log_file: Optional[str] = None, 
                    log_format: Optional[str] = None,
                    use_console: bool = True) -> logging.Logger:
        """
        设置并获取日志记录器
        
        参数:
        - name: 日志记录器名称
        - level: 日志级别，可选值为 "debug", "info", "warning", "error", "critical"
        - log_file: 日志文件路径，如果为None则不记录到文件
        - log_format: 日志格式，如果为None则使用默认格式
        - use_console: 是否输出到控制台
        
        返回:
        - 日志记录器实例
        """
        # 如果已经创建过该名称的日志记录器，则直接返回
        if name in cls._loggers:
            return cls._loggers[name]
        
        # 创建日志记录器
        logger = logging.getLogger(name)
        
        # 设置日志级别
        log_level = cls.LOG_LEVELS.get(level.lower(), logging.INFO)
        logger.setLevel(log_level)
        
        # 设置日志格式
        formatter = logging.Formatter(log_format or cls.LOG_FORMAT)
        
        # 添加控制台处理器
        if use_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 添加文件处理器（如果指定了日志文件）
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # 缓存日志记录器
        cls._loggers[name] = logger
        
        return logger
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        获取已创建的日志记录器
        
        参数:
        - name: 日志记录器名称
        
        返回:
        - 日志记录器实例，如果不存在则创建一个新的
        """
        if name not in cls._loggers:
            return cls.setup_logger(name)
        return cls._loggers[name]
    
    @classmethod
    def setup_default_loggers(cls, log_dir: str = "logs") -> Dict[str, logging.Logger]:
        """
        设置默认的日志记录器
        
        参数:
        - log_dir: 日志目录
        
        返回:
        - 日志记录器字典
        """
        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 获取当前日期作为日志文件名的一部分
        date_str = datetime.now().strftime("%Y%m%d")
        
        # 创建应用日志记录器
        app_logger = cls.setup_logger(
            name="app",
            level="info",
            log_file=os.path.join(log_dir, f"app_{date_str}.log")
        )
        
        # 创建错误日志记录器
        error_logger = cls.setup_logger(
            name="error",
            level="error",
            log_file=os.path.join(log_dir, f"error_{date_str}.log")
        )
        
        # 创建模型日志记录器
        model_logger = cls.setup_logger(
            name="model",
            level="info",
            log_file=os.path.join(log_dir, f"model_{date_str}.log")
        )
        
        return {
            "app": app_logger,
            "error": error_logger,
            "model": model_logger
        }
    
    @staticmethod
    def log_function_call(logger: logging.Logger, level: str = "info"):
        """
        装饰器：记录函数调用
        
        参数:
        - logger: 日志记录器
        - level: 日志级别
        
        用法:
        @LoggingUtils.log_function_call(logger, "info")
        def my_function(arg1, arg2):
            pass
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 获取日志级别对应的方法
                log_method = getattr(logger, level.lower(), logger.info)
                
                # 记录函数调用
                log_method(f"调用函数 {func.__name__} 开始，参数: args={args}, kwargs={kwargs}")
                
                try:
                    # 执行函数
                    result = func(*args, **kwargs)
                    
                    # 记录函数返回
                    log_method(f"调用函数 {func.__name__} 结束，返回值: {result}")
                    
                    return result
                except Exception as e:
                    # 记录异常
                    logger.error(f"调用函数 {func.__name__} 异常: {str(e)}")
                    raise
            
            return wrapper
        
        return decorator
    
    @staticmethod
    def log_execution_time(logger: logging.Logger, level: str = "info"):
        """
        装饰器：记录函数执行时间
        
        参数:
        - logger: 日志记录器
        - level: 日志级别
        
        用法:
        @LoggingUtils.log_execution_time(logger, "info")
        def my_function():
            pass
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 获取日志级别对应的方法
                log_method = getattr(logger, level.lower(), logger.info)
                
                # 记录开始时间
                start_time = datetime.now()
                
                try:
                    # 执行函数
                    result = func(*args, **kwargs)
                    
                    # 计算执行时间
                    execution_time = datetime.now() - start_time
                    
                    # 记录执行时间
                    log_method(f"函数 {func.__name__} 执行时间: {execution_time}")
                    
                    return result
                except Exception as e:
                    # 计算执行时间
                    execution_time = datetime.now() - start_time
                    
                    # 记录异常和执行时间
                    logger.error(f"函数 {func.__name__} 异常: {str(e)}, 执行时间: {execution_time}")
                    raise
            
            return wrapper
        
        return decorator
    


if __name__ == '__main__':
    # 创建日志记录器
    logger = LoggingUtils.setup_logger(
        name="my_logger",
        level="info",
        log_file="logs/my_log.log"
    )

    # 记录日志
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")

    # 使用装饰器记录函数调用
    @LoggingUtils.log_function_call(logger)
    def my_function(arg1, arg2):
        return arg1 + arg2

    # 使用装饰器记录函数执行时间
    @LoggingUtils.log_execution_time(logger)
    def time_consuming_function():
        import time
        time.sleep(1)
        return "done"

    # 调用函数
    result = my_function(1, 2)
    time_consuming_function()

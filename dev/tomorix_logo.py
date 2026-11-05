"""
Tomorix Logo and Banner utilities.
"""

def print_tomorix_banner():
    """打印 Tomorix 的 ASCII 艺术横幅"""
    # 使用 ANSI 颜色代码：蓝色和黄色
    BLUE = '\033[94m'      # 亮蓝色
    CYAN = '\033[96m'      # 青色
    YELLOW = '\033[93m'    # 黄色
    BOLD = '\033[1m'       # 粗体
    RESET = '\033[0m'      # 重置
    
    # ASCII 艺术字（使用 Big 字体）
    banner = f"""
{BLUE}{BOLD}  ████████╗ ██████╗ ███╗   ███╗ ██████╗ ██████╗ ██╗██╗  ██╗
  ╚══██╔══╝██╔═══██╗████╗ ████║██╔═══██╗██╔══██╗██║╚██╗██╔╝
     ██║   ██║   ██║██╔████╔██║██║   ██║██████╔╝██║ ╚███╔╝ 
     ██║   ██║   ██║██║╚██╔╝██║██║   ██║██╔══██╗██║ ██╔██╗ 
     ██║   ╚██████╔╝██║ ╚═╝ ██║╚██████╔╝██║  ██║██║██╔╝ ██╗
     ╚═╝    ╚═════╝ ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝{RESET}
                                                              
{CYAN}          ╔══════════════════════════════════════════╗{RESET}
{CYAN}          ║  {YELLOW}{BOLD}Tomorrow's Solutions, Today's Innovation{RESET}{CYAN}  ║{RESET}
{CYAN}          ╚══════════════════════════════════════════╝{RESET}
"""
    print(banner)


def print_tomorix_simple_banner():
    """打印简化版的 Tomorix 横幅（无颜色）"""
    banner = """
  ████████╗ ██████╗ ███╗   ███╗ ██████╗ ██████╗ ██╗██╗  ██╗
  ╚══██╔══╝██╔═══██╗████╗ ████║██╔═══██╗██╔══██╗██║╚██╗██╔╝
     ██║   ██║   ██║██╔████╔██║██║   ██║██████╔╝██║ ╚███╔╝ 
     ██║   ██║   ██║██║╚██╔╝██║██║   ██║██╔══██╗██║ ██╔██╗ 
     ██║   ╚██████╔╝██║ ╚═╝ ██║╚██████╔╝██║  ██║██║██╔╝ ██╗
     ╚═╝    ╚═════╝ ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
                                                              
          ══════════════════════════════════════════
            Tomorrow's Solutions, Today's Innovation
          ══════════════════════════════════════════
"""
    print(banner)


def print_tomorix_compact_banner():
    """打印紧凑版的 Tomorix 横幅（带颜色）"""
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    banner = f"""
{BLUE}{BOLD}╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   ████████╗ ██████╗ ███╗   ███╗ ██████╗ ██████╗ ██╗██╗  ██╗║
║      ██║   ██║   ██║████╗ ████║██║   ██║██████╔╝██║╚███╔╝ ║
║      ██║   ╚██████╔╝██║ ╚═╝ ██║╚██████╔╝██║  ██║██║██╔╝ ██╗║
║                                                            ║
║          {YELLOW}🚀 Tomorrow's Solutions, Today 🚀{BLUE}              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def print_tomorix_mini_banner():
    """打印迷你版的 Tomorix 横幅"""
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    print(f"""
{BLUE}{BOLD}┌─────────────────────────────────────┐
│  TOMORIX  {YELLOW}█▓▒░{BLUE}  Future-Ready Tech  │
└─────────────────────────────────────┘{RESET}
""")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Tomorix Banner 样式预览")
    print("="*60)
    
    print("\n【样式 1：标准版（带颜色）】")
    print_tomorix_banner()
    
    print("\n" + "-"*60)
    print("\n【样式 2：简化版（无颜色）】")
    print_tomorix_simple_banner()
    
    print("\n" + "-"*60)
    print("\n【样式 3：紧凑版（带边框）】")
    print_tomorix_compact_banner()
    
    print("\n" + "-"*60)
    print("\n【样式 4：迷你版】")
    print_tomorix_mini_banner()
    
    print("\n" + "="*60)
    print("使用方法：")
    print("  from tomorix_logo import print_tomorix_banner")
    print("  print_tomorix_banner()")
    print("="*60 + "\n")

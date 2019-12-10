[TOC]

# sumatra pdf 介绍（转载）

### 介绍

> - 支持PDF, ePub, Mobi, XPS, DjVu, CHM, CBZ 和 CBR格式的电子文件
> - 仅支持Windows，原因见官网论坛
> - 绿色版，小巧，于此潮流而不失赤子之心者，唯 Sumatra PDF 而已矣
> - Sumatra PDF 的启动是异常迅速，几乎没有启动过程，程序界面直接呈现
> - 图形化显示最近文档
> - 打开历史文档后，直接跳转到上次查看的位置
> - 这里介绍**比官方文档还全 && 详细的快捷键及命令行**

### 快捷键

#### 导航

> - Mouse Left/ Right 拖动 区别是左键在文字区是选中文字，右键在文字区也是拖动
> - J/ K, Up/ Down 向上/ 下滚动一行 **Unix风格**
> - Space 向下滚动一屏
> - Shift + Space 向上滚动一屏
> - N/P 下一页/ 上一页 **Unix风格**
> - Page Down/Page Up 下一页/ 上一页
> - Alt + 向左键 上一视图 对应于跳转页面
> - Alt + 向右键 下一视图
> - Ctrl + G/ G 跳转页码
> - Home 第一页
> - End 最后一页
> - **B 书籍视图翻页**
> - Ctrl + Shift + Right 打开同目录中的下一个PDF文件
> - Ctrl + Shift + Left 打开同目录中的上一个PDF文件
> - Tab 在各个视图框和查找框之间跳转
> - Alt 打开菜单栏

#### 阅读

> - +, - 放大/缩小
> - Ctrl + Scroll Wheel 放大/ 缩小
> - **Mouse Right + Scrol Whell 放大/ 缩小**
> - Mouse Middle 控制向下，向上移动的速度
> - **Z 在适应页面, 适应宽度, 适应内容之间切换**
> - **C 连续显示页面/ 不连续显示页面**
> - Ctrl + Shift + - 向左旋转
> - Ctrl + Shift + + 向右旋转
> - F12 显示/隐藏书签(目录)
> - F6 切换书签和主窗口之间的焦点
> - Ctrl + L/ F11 幻灯片模式(最小全屏模式) 两个快捷键底色不同
> - Ctrl + Shift + L 全屏模式
> - Shift + F11 全屏模式
> - ESC 退出全屏或幻灯片模式
> - . **幻灯片模式中, 背景变为黑色**
> - **W 幻灯片模式中, 背景变为白色**
> - **I 全屏/幻灯片模式中, 显示页码**

#### 文件操作

> - Ctrl + O 打开文件
> - Ctrl + W 关闭文件
> - Ctrl + S 另存为
> - Ctrl + P 打印
> - **R 重新载入**
> - Ctrl + F, / 查找文本不区分大小写
> - F3/ Shift + F3 查找下一个/上一个
> - Ctrl + Q/ Q 退出 **Unix风格**
> - Ctrl + Left Mouse 块状选择文本或图片并复制到剪贴板

### 命令行

> - 命令行博大精深，[奉上链接](https://github.com/sumatrapdfreader/sumatrapdf/wiki/Command-line-arguments)，其实是比较懒

### 收藏/书签功能

> - Sumatra PDF也有简单的书签功能，这个功能在里面其实叫收藏，和“查看”选项里的“书签”不是一个意思。点击“收藏——收藏XX页”就可以给当页添加书签。还可以添加简单的描述。
> - 需要查看该文档的收藏，只需要点击“显示收藏”即可

### 文本设置

> - 设置 -> 高级选项

```
    FixedPageUI [
        TextColor = #000000
        BackgroundColor = #d6e7cb
        SelectionColor = #7CFC00
        WindowMargin = 0 0 0 0
        PageSpacing = 4 4
        GradientColors = #2828aa #28aa28 #aa2828
    ]
    ShowMenubar = false
    ShowToolbar = false
    CheckForUpdates = false1234567891011
```

> - [官方文档](http://www.sumatrapdfreader.org/settings.html)
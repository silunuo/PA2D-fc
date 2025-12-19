### PA2D 1.0.0-beta1 版本

#### 关于 **1.0.0-beta1**
此项目已经做了2个半月了，我经常重构以及更改接口，以及前段时间做了一些测试，认为这个版本已经可以用于实验性地做一些小的作品了。

#### 如何使用这个库

#####  Step 1: 实验性地使用一下 Window 类
引入头文件后，尝试创建一个窗口
```cpp
#include<pa2d.h>
using namepsace pa2d;
// 创建一个 640*480 的窗口
Window window(640, 480，"My first PA2D Window");
```
事实上，初次创建窗口并不会立即显示，因为初始化并不代表着显示

让我们在main函数中，调用窗口的`show()`方法
```cpp
int main(){
  window.show();
  // 我提供了 waitForClose() 方法方便展示窗口
  // 防止程序立即退出
  window.waitForClose();
}
```

运行后，我们就会获得以下的窗口
<img width="1000" height="806" alt="QQ_1766143887620" src="https://github.com/user-attachments/assets/4affbe8a-a220-4f1b-b454-de7d312eae8b" />



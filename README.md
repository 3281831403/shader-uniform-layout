# shader-uniform-layout

std140/std430 uniform 块布局、填充与打包修复题。仓库开局是纯桩环境：8 个函数全部 `raise NotImplementedError`，15 个测试全部红。

## ① 题面

把着色器参数改成结构化 uniform 块后，颜色全乱、矩阵还错位：vec3 只按 12 字节对齐，后续字段全部错位；mat4 按 4 字节对齐，列向量偏移全错；数组 stride 直接取元素大小，跨元素读到脏数据；std430 与 std140 混用同一套规则；结构体嵌套没有递归计算，内部字段偏移错误；打包时不做填充，字节整体错位；int 与 float 都是 4 字节但混用时没有区分打包方式；越界字段不报错，最终写穿缓冲；批量布局共用偏移表，块与块之间互相串；小端打包把字节序写反，数值完全错误。layout/std140.py 是核心，tests/test_std140.py 15个红。要修的是：按 std140/std430 分别计算标量、vec3、矩阵列与结构体对齐，数组 stride 取 16 的倍数，递归布局嵌套结构体，按偏移填充并用小端打包，校验错位/越界/非 16 倍数，批量布局相互隔离且失败整体回滚。

## ② 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态的输出尾部必须恰好是：

```text
15 failed in <时间>s
```

即恰好 15 个 `failed`，且没有 collection error、import error 或 syntax error。测试只使用 Python 标准库（`math`、`struct`）和 pytest，离线运行，不依赖随机数，也不访问网络。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## ④ 测试约束

`tests/test_std140.py` 不可修改：不得删改测试、增加 skip/xfail、放宽断言，也不得通过测试侧的特殊分支绕过断言。修复应全部落在 `layout/std140.py`，并保持 `LayoutError` 的异常契约以及 docstring 中关于 std140/std430 对齐、数组 stride、嵌套布局和字节序的约定。

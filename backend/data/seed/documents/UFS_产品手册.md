# UFS 3.1/4.0 通用闪存存储产品手册

## 1. 产品概述
UFS (Universal Flash Storage) 是面向移动与车载的高速串行闪存器件,
采用差分信号与指令队列机制, 读写带宽远超 eMMC, 是中高端手机、
平板与车机的首选嵌入式半导体存储产品.

## 2. 规格对比
| 规格 | 接口 | 单通道速率 | 理论带宽 (2 lane) | 典型容量 | 封装 |
|------|------|-----------|-------------------|----------|------|
| UFS 3.1 | HS-Gear4 | 11.6 Gbps/lane | 23.2 Gbps | 128GB-512GB | 153-ball BGA |
| UFS 4.0 | HS-Gear5 | 23.2 Gbps/lane | 46.4 Gbps | 256GB-1TB | 153-ball BGA |

## 3. 关键特性
- 顺序读写: UFS 3.1 读 ~2100 MB/s, 写 ~1200 MB/s
- 顺序读写: UFS 4.0 读 ~4200 MB/s, 写 ~2800 MB/s
- HS-Gear4: UFS 3.1 高速档, 单 lane 11.6 Gbps
- HS-Gear5: UFS 4.0 高速档, 单 lane 23.2 Gbps
- LU 分区: 支持多个逻辑单元, 区分系统与用户数据
- HPB (Host Performance Booster): 主机端缓存映射表, 提升随机读
- 写入增强器 (Write Booster): 缓存写入, 提升突发写性能

## 4. 应用场景
- 智能手机: UFS 3.1 中高端机型, UFS 4.0 旗舰机型
- 平板电脑: UFS 3.1 主流, 支持多任务与大文件
- 车载信息娱乐 (IVI): 车规 UFS 3.1, 通过 AEC-Q100
- AR/VR 设备: UFS 4.0 满足高带宽低延迟需求

## 5. 与 eMMC 对比
- 接口: UFS 全双工串行差分, eMMC 半双工并行 (8-bit)
- 带宽: UFS 4.0 较 eMMC 5.1 提升 6 倍以上
- 队列: UFS 支持多指令队列 (CQ), eMMC 单指令处理
- 容量: UFS 支持更大容量 (≤1TB), eMMC 适合中低端
- 成本: eMMC 成本更低, UFS 性能更优

## 6. 合规与可靠性
- JEDEC JESD220 (UFS 标准) / JESD223 (UFSHCI)
- AEC-Q100 车规认证 (车规型号)
- 工作温度: 0°C ~ 85°C (商业级) / -40°C ~ 105°C (车规级)
- TBW 与 P/E cycle 评估寿命, 支持固件 OTA 升级

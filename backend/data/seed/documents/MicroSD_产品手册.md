# MicroSD 卡产品手册

## 1. 产品概述
MicroSD 卡是基于 NAND Flash 的小体积可移动半导体存储卡, 广泛应用于手机、
平板、行车记录仪、安防监控与物联网设备. 本手册覆盖 MicroSD / SD 卡的容量等级、
速度等级与应用选型.

## 2. 容量等级
| 等级 | 容量范围 | 文件系统 | 典型应用 |
|------|----------|----------|----------|
| SD / SDSC | 1-2GB | FAT16 | 老旧设备 |
| SDHC | 4-32GB | FAT32 | 手机 / 相机 |
| SDXC | 64GB-2TB | exFAT | 4K 视频 / 大文件 |
| SDUC | 2-128TB | exFAT | 工业与大容量存储 |

## 3. 速度等级
- C10 (Class 10): 最低写入 10 MB/s, 满足 1080p 视频
- UHS-I: 接口 50 MB/s (DS) / 104 MB/s (SDR50 / SDR104)
- UHS-II: 接口 156-312 MB/s, 第二排针脚
- UHS-III: 接口 312-624 MB/s
- U1 (UHS Speed Class 1): 最低写入 10 MB/s
- U3 (UHS Speed Class 3): 最低写入 30 MB/s, 支持 4K 视频
- V10 / V30 / V60 / V90 (Video Speed Class): 视频最低写入速率等级

## 4. 应用总线与最高速率
- SD Bus: 默认 25 MHz, 高速 50 MHz
- UHS-I SDR104: 104 MB/s
- UHS-II SDR312: 312 MB/s
- PCIe (microSD Express): 复用 UHS-II 接口, 支持 NVMe, >800 MB/s

## 5. 应用场景
- 智能手机扩展存储: microSDXC U3 V30, 128-512GB
- 行车记录仪与安防监控: U3 V30 / V60, 支持 4K 录制
- 行车记录仪: 高耐用卡, MLC NAND, 持续擦写
- 工业物联网: 工规 MicroSD, 宽温 -25°C ~ 85°C
- 游戏机扩展: 高速随机读, microSDXC U3

## 6. 可靠性与耐久性
- 颗粒: TLC 消费级, MLC / SLC 高耐用级
- TBW: 写入寿命指标, 工规卡支持高 TBW
- 防水防磁防 X 光: 工业级三防
- 磨损均衡与 ECC: 内置 FTL 与纠错算法
- 工作温度: 0°C ~ 70°C (商业级) / -25°C ~ 85°C (工规)

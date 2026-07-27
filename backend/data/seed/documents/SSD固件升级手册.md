# SSD 固件升级与维护手册

## 1. 固件版本管理
- 固件版本号: MAJOR.MINOR.PATCH (如 1.2.3)
- MAJOR: 主版本, 含 FTL 算法或硬件架构变更, 需全量验证
- MINOR: 次版本, 功能增强与性能优化
- PATCH: 补丁版本, 修复缺陷与兼容性
- 版本记录: 维护 changelog, 记录每版本变更与回滚点

## 2. 升级前准备
- 备份用户数据: 升级存在异常风险, 必须先备份
- 检查当前固件版本: 通过 SMART 或厂商工具读取
- 确认供电稳定: 笔记本充满电, 桌面接 UPS, 防止掉电变砖
- 关闭杀毒与磁盘加密 (BitLocker / FileVault), 防止写入冲突
- 下载官方固件包, 校验 SHA256 完整性

## 3. 升级流程
1. 运行厂商升级工具 (如 Samsung Magician / Kioxia SSD Utility)
2. 工具识别 SSD 型号与当前固件, 自动匹配目标版本
3. 加载固件包, 工具二次校验签名与兼容性
4. 触发升级: 工具下发固件, SSD 内部写入 flash 并切换启动区
5. 重启主机, SSD 加载新固件并完成初始化
6. 升级后核对版本号, 跑一次全盘 SMART 自检

## 4. 回滚机制
- A/B 双区: SSD 固件存储区分 A / B 两份, 升级写非活动区, 失败可回滚
- 回滚触发: 升级后无法识别或 SMART 异常, 自动回滚至上一可用版本
- 手动回滚: 工具选择旧版本固件包强制写入, 仅限维护场景
- 回滚限制: 跨 MAJOR 版本一般不支持自动回滚

## 5. SMART 健康监测
- 05 (Reallocated Sector Count): 重映射扇区数, 反映坏块增长
- 09 (Power-On Hours): 累计通电时长
- 0C (Power Cycle Count): 开关机次数
- AA (Available Reserved Space): 剩余预留空间, 低于阈值需更换
- B1 (Wear Range Delta): 磨损均衡偏差, 越小越均衡
- AD (Wear Leveling Count): 平均擦写次数
- E7 (SSD Life Left): 剩余寿命百分比
- E9 (Total NAND Writes): 累计 NAND 写入量

## 6. 日常维护技术
- TRIM: 操作系统通知 SSD 回收已删除数据块, 维持写入性能
- Garbage Collection (GC): SSD 后台回收有效数据, 释放空闲块
- Wear Leveling: 磨损均衡, 均匀分配擦写, 延长颗粒寿命
- Over-Provisioning (OP): 预留空间, 提升写入性能与寿命
- 定期 SMART 巡检, 关注重映射扇区增长趋势

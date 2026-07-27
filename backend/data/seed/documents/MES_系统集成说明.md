# MES 系统集成说明

## 1. 集成架构
车规 eMMC 5.1/eMMC 5.1 通过 REST API 与 MES 系统集成, 实现物料数据实时同步.

## 2. 接口列表
- POST /api/v1/material/in: 入库
- POST /api/v1/material/out: 出库
- GET /api/v1/material/stock: 查询库存
- POST /api/v1/inventory/check: 盘点结果上传

## 3. 鉴权
- OAuth 2.0 client_credentials
- Token 有效期 1 小时, 自动续期
- 接口调用频率限制: 100 QPS

## 4. 数据格式
JSON, UTF-8 编码, 时间戳为 ISO 8601.

## 5. 同步策略
- 实时同步: 入库 / 出库立即推送
- 批量同步: 盘点结果每小时批量上传
- 异常重试: 失败请求自动重试 3 次, 间隔 1/2/5 分钟

## 6. 错误码
| 错误码 | 含义 | 处理建议 |
|--------|------|----------|
| 400 | 参数错误 | 检查请求体 |
| 401 | 鉴权失败 | 重新获取 token |
| 409 | 数据冲突 | 检查物料编码 |
| 500 | 服务器错误 | 联系运维 |

## 7. 测试环境
- 测试地址: https://mes-test.example.com
- 测试账号: 向 MES 管理员申请
- 提供完整 Postman 集合

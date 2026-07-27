"""
Neo4j 图谱种子数据灌入脚本

功能:
1. 连接 Neo4j, 检查图谱是否已有数据
2. 灌入真实业务实体关系 (31节点/56关系/9类型)
3. 打印灌入统计

用法:
    python -m scripts.seed_graph
    或: python scripts/seed_graph.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 确保 backend 在 sys.path 中
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.graphrag.neo4j_store import Neo4jStore  # noqa: E402
from app.graphrag.schemas import (  # noqa: E402
    Entity,
    EntityType,
    Relation,
    RelationType,
)


# ======================== 实体类型简写 ========================
# 取枚举 value 作为 Neo4j 节点 label, upsert 时会做白名单校验
_P = EntityType.Product.value
_D = EntityType.Department.value
_PE = EntityType.Person.value
_PO = EntityType.Policy.value
_S = EntityType.Supplier.value
_C = EntityType.Certification.value
_PA = EntityType.Patent.value


# ======================== 种子实体 (31 个) ========================
# (name, type, properties) —— 与前端 seedGraphData.nodes 完全一致
SEED_ENTITIES: list[tuple[str, str, dict]] = [
    # ---- 产品 (6) ----
    ("车规 eMMC 5.1", _P, {"类别": "嵌入式存储", "车规等级": "AEC-Q100 Grade 3", "容量": "8~128GB"}),
    ("3D TLC NAND Flash", _P, {"类别": "NAND 晶圆", "堆叠": "128 层", "工艺": "TLC"}),
    ("NVMe SSD 企业级", _P, {"类别": "固态硬盘", "接口": "PCIe Gen4 x4", "容量": "1.92~7.68TB"}),
    ("DDR5 DRAM", _P, {"类别": "DRAM 颗粒", "速率": "DDR5-6400", "容量": "16Gb"}),
    ("UFS 4.0", _P, {"类别": "嵌入式存储", "接口": "UFS 4.0", "速率": "23.2Gbps"}),
    ("MicroSD 卡", _P, {"类别": "存储卡", "标准": "SD Express", "容量": "64~512GB"}),
    # ---- 部门 (4) ----
    ("研发中心", _D, {"编码": "RD", "人数": 48}),
    ("质量保证部", _D, {"编码": "QA", "人数": 20}),
    ("采购部", _D, {"编码": "PROC", "人数": 9}),
    ("合规认证部", _D, {"编码": "CMP", "人数": 6}),
    # ---- 人员 (4) ----
    ("李强", _PE, {"角色": "编辑", "部门": "研发中心"}),
    ("王敏", _PE, {"角色": "编辑", "部门": "质量保证部"}),
    ("陈工", _PE, {"角色": "认证工程师", "部门": "合规认证部"}),
    ("赵磊", _PE, {"角色": "采购工程师", "部门": "采购部"}),
    # ---- 制度 (3) ----
    ("可靠性测试规范", _PO, {"标准": "AEC-Q100/JEDEC", "版本": "2026"}),
    ("供应链风险管理办法", _PO, {"版本": "2026", "分级": "I/II/III"}),
    ("出口管制合规政策", _PO, {"版本": "2026", "适用": "半导体存储"}),
    # ---- 供应商 (4) ----
    ("中芯国际", _S, {"等级": "A", "主供": "NAND/DRAM 晶圆代工"}),
    ("华虹半导体", _S, {"等级": "A", "主供": "NAND 晶圆代工(二供)"}),
    ("长电科技", _S, {"等级": "A", "主供": "OSAT 封测"}),
    ("通富微电", _S, {"等级": "A", "主供": "OSAT 封测"}),
    # ---- 认证 (8) ----
    ("ISO 9001", _C, {"机构": "SGS", "到期": "2027-06"}),
    ("ISO 14001", _C, {"机构": "TUV", "到期": "2027-03"}),
    ("BSCI 社会责任", _C, {"机构": "amfori", "到期": "2026-11"}),
    ("IATF 16949", _C, {"机构": "TUV", "适用": "汽车供应链"}),
    ("AEC-Q100", _C, {"适用": "车规集成电路", "等级": "Grade 0~3"}),
    ("CE 认证", _C, {"指令": "LVD/EMC/RED", "到期": "长期"}),
    ("FCC 认证", _C, {"等级": "SDoC", "到期": "长期"}),
    ("RoHS 2.0", _C, {"版本": "2011/65/EU", "到期": "长期"}),
    # ---- 专利 (2) ----
    ("ZL202310458X 3D NAND 堆叠结构", _PA, {"类型": "发明专利", "申请人": "研发中心"}),
    ("ZL202410112X eMMC 磨损均衡算法", _PA, {"类型": "发明专利", "申请人": "研发中心"}),
]


# ======================== 关系类型简写 ========================
_BEL = RelationType.BELONGS_TO.value
_SUP = RelationType.SUPPLIES.value
_CER = RelationType.CERTIFIED_BY.value
_AUD = RelationType.AUDITED_BY.value
_PAR = RelationType.PARTICIPATES_IN.value
_GOV = RelationType.GOVERNED_BY.value
_REF = RelationType.REFERENCES.value
_INV = RelationType.INVENTED_BY.value
_AUT = RelationType.AUTHORED_BY.value


# ======================== 种子关系 (56 条) ========================
# (source, source_type, target, target_type, relation_type) —— 与前端 seedGraphData.links 完全一致
SEED_RELATIONS: list[tuple[str, str, str, str, str]] = [
    # 产品 → 部门 (归属)
    ("车规 eMMC 5.1", _P, "研发中心", _D, _BEL),
    ("3D TLC NAND Flash", _P, "研发中心", _D, _BEL),
    ("NVMe SSD 企业级", _P, "研发中心", _D, _BEL),
    ("DDR5 DRAM", _P, "研发中心", _D, _BEL),
    ("UFS 4.0", _P, "研发中心", _D, _BEL),
    ("MicroSD 卡", _P, "研发中心", _D, _BEL),
    ("3D TLC NAND Flash", _P, "采购部", _D, _BEL),
    # 产品 → 供应商 (供货)
    ("3D TLC NAND Flash", _P, "中芯国际", _S, _SUP),
    ("3D TLC NAND Flash", _P, "华虹半导体", _S, _SUP),
    ("3D TLC NAND Flash", _P, "长电科技", _S, _SUP),
    ("3D TLC NAND Flash", _P, "通富微电", _S, _SUP),
    ("车规 eMMC 5.1", _P, "长电科技", _S, _SUP),
    ("车规 eMMC 5.1", _P, "通富微电", _S, _SUP),
    ("NVMe SSD 企业级", _P, "中芯国际", _S, _SUP),
    # 产品 → 认证
    ("车规 eMMC 5.1", _P, "AEC-Q100", _C, _CER),
    ("车规 eMMC 5.1", _P, "IATF 16949", _C, _CER),
    ("车规 eMMC 5.1", _P, "CE 认证", _C, _CER),
    ("车规 eMMC 5.1", _P, "RoHS 2.0", _C, _CER),
    ("NVMe SSD 企业级", _P, "CE 认证", _C, _CER),
    ("NVMe SSD 企业级", _P, "FCC 认证", _C, _CER),
    ("NVMe SSD 企业级", _P, "RoHS 2.0", _C, _CER),
    ("UFS 4.0", _P, "CE 认证", _C, _CER),
    ("UFS 4.0", _P, "FCC 认证", _C, _CER),
    ("MicroSD 卡", _P, "CE 认证", _C, _CER),
    ("MicroSD 卡", _P, "FCC 认证", _C, _CER),
    ("MicroSD 卡", _P, "RoHS 2.0", _C, _CER),
    # 部门 → 认证 (审核/参与)
    ("质量保证部", _D, "ISO 9001", _C, _AUD),
    ("质量保证部", _D, "ISO 14001", _C, _AUD),
    ("合规认证部", _D, "BSCI 社会责任", _C, _AUD),
    ("合规认证部", _D, "CE 认证", _C, _PAR),
    ("合规认证部", _D, "FCC 认证", _C, _PAR),
    # 供应商 → 认证 (资质)
    ("中芯国际", _S, "ISO 9001", _C, _CER),
    ("中芯国际", _S, "IATF 16949", _C, _CER),
    ("长电科技", _S, "ISO 9001", _C, _CER),
    ("长电科技", _S, "IATF 16949", _C, _CER),
    ("通富微电", _S, "ISO 9001", _C, _CER),
    # 产品/供应商 → 制度 (受约束)
    ("车规 eMMC 5.1", _P, "可靠性测试规范", _PO, _GOV),
    ("3D TLC NAND Flash", _P, "可靠性测试规范", _PO, _GOV),
    ("NVMe SSD 企业级", _P, "可靠性测试规范", _PO, _GOV),
    ("NVMe SSD 企业级", _P, "出口管制合规政策", _PO, _GOV),
    ("UFS 4.0", _P, "出口管制合规政策", _PO, _GOV),
    ("车规 eMMC 5.1", _P, "供应链风险管理办法", _PO, _GOV),
    ("中芯国际", _S, "供应链风险管理办法", _PO, _GOV),
    # 产品 → 专利 (发明)
    ("3D TLC NAND Flash", _P, "ZL202310458X 3D NAND 堆叠结构", _PA, _INV),
    ("车规 eMMC 5.1", _P, "ZL202410112X eMMC 磨损均衡算法", _PA, _INV),
    # 产品 → 专利/产品 (引用)
    ("NVMe SSD 企业级", _P, "ZL202310458X 3D NAND 堆叠结构", _PA, _REF),
    ("NVMe SSD 企业级", _P, "3D TLC NAND Flash", _P, _REF),
    ("车规 eMMC 5.1", _P, "3D TLC NAND Flash", _P, _REF),
    # 专利 → 人员 (发明人)
    ("ZL202310458X 3D NAND 堆叠结构", _PA, "李强", _PE, _INV),
    ("ZL202410112X eMMC 磨损均衡算法", _PA, "李强", _PE, _INV),
    # 人员 → 部门 (归属)
    ("李强", _PE, "研发中心", _D, _BEL),
    ("王敏", _PE, "质量保证部", _D, _BEL),
    ("陈工", _PE, "合规认证部", _D, _BEL),
    ("赵磊", _PE, "采购部", _D, _BEL),
    # 制度/专利 → 部门/人员 (撰写)
    ("可靠性测试规范", _PO, "质量保证部", _D, _AUT),
    ("出口管制合规政策", _PO, "陈工", _PE, _AUT),
]


def _count_by_type(items, type_idx: int) -> dict[str, int]:
    """按指定位置的 type 字段聚合计数"""
    counts: dict[str, int] = {}
    for item in items:
        t = item[type_idx]
        counts[t] = counts.get(t, 0) + 1
    return counts


async def main() -> int:
    """主流程: 连接 → 去重检查 → 灌入实体 → 灌入关系 → 打印统计"""
    store = Neo4jStore()

    # stats() 内部会触发惰性连接, 不可用时返回 available=False
    stats = await store.stats()
    if not stats.get("available"):
        print("[seed_graph] Neo4j 不可用, 降级跳过种子灌入 (不报错退出)")
        return 0

    # 图谱已有数据则跳过, 避免重复灌入
    if stats.get("nodes", 0) > 0:
        print(
            f"[seed_graph] 图谱已有数据 (节点 {stats['nodes']} / 关系 "
            f"{stats.get('relationships', 0)}), 跳过种子灌入"
        )
        await store.close()
        return 0

    # 确保约束与索引就绪 (新增 Certification/Patent 等类型的唯一约束)
    await store.init_schema()

    print("=" * 60)
    print("开始灌入 Neo4j 图谱种子数据...")
    print(f"  目标: 实体 {len(SEED_ENTITIES)} 个 / 关系 {len(SEED_RELATIONS)} 条")
    print("=" * 60)

    # 实体先于关系写入 (关系 MATCH 依赖两端节点存在)
    ent_ok = 0
    for name, etype, props in SEED_ENTITIES:
        entity = Entity(name=name, type=etype, properties=dict(props))
        eid = await store.upsert_entity(entity)
        if eid:
            ent_ok += 1
        else:
            print(f"  [警告] 实体写入失败: {name}")

    rel_ok = 0
    for src, src_t, tgt, tgt_t, rel_t in SEED_RELATIONS:
        rel = Relation(
            source_entity=src,
            source_type=src_t,
            target_entity=tgt,
            target_type=tgt_t,
            relation_type=rel_t,
        )
        rid = await store.upsert_relation(rel)
        if rid:
            rel_ok += 1
        else:
            print(f"  [警告] 关系写入失败: {src} -[{rel_t}]-> {tgt}")

    # 拉取实际入库后的图谱统计
    final_stats = await store.stats()
    await store.close()

    ent_dist = _count_by_type(SEED_ENTITIES, 1)
    rel_dist = _count_by_type(SEED_RELATIONS, 4)

    print("=" * 60)
    print("Neo4j 图谱种子数据灌入完成")
    print(f"  实体: {ent_ok}/{len(SEED_ENTITIES)}")
    print(f"  关系: {rel_ok}/{len(SEED_RELATIONS)}")
    print("  实体类型分布:")
    for t, c in sorted(ent_dist.items(), key=lambda x: -x[1]):
        print(f"    - {t}: {c}")
    print("  关系类型分布:")
    for t, c in sorted(rel_dist.items(), key=lambda x: -x[1]):
        print(f"    - {t}: {c}")
    print(
        f"  图谱实际存量: 节点 {final_stats.get('nodes', 0)} / "
        f"关系 {final_stats.get('relationships', 0)}"
    )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

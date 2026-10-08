# 基础与来源

本库为 BW1100/gfx938 的实验知识库。页面分层和四个检索/索引/校验工具来自
qhy991/ROCm-KernelWiki-Q，固定提交 c7cb7b6e422bdbe618fd69d0495b5d20718c7da7。
原始地址：https://github.com/qhy991/ROCm-KernelWiki-Q 。

复用 scripts/query.py、get_page.py、generate-indices.py、validate.py 与 data/schemas.yaml。
保留其来源记录；新增 source-experiment、本机 controlled vocabulary 和 evidence-scope检查。查询同时提供keyword/structured filter时，新库先过滤后排序，原库不受修改。
没有复制原1078页或把AMD硬件声明改称Hygon声明。原库继续作为参考资料独立维护。
2026-10-08 核对：该固定提交根目录未附 LICENSE，GitHub 上游仓库许可字段为空。
本库保留来源与固定版本，不为继承代码补造许可证或授予未经确认的再分发许可。
公开可读不代表所有内容已按某个开源许可证授权。

BW1100结论来自本账户保留的源锁、数值/计时/实际profile/terminal receipts、已有baseline文档和
Compiler提交。原实验拥有原始事实，本库只维护有适用范围的综合页面和source记录。
不复制数据集、weights、credentials、raw模型会话或全部GPU缓存；实验原产物留在所属目录。

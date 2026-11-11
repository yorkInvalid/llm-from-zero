# 云与 AI 解决方案面试调研

检索日期：2026-11-03。用途：确定通用技术训练范围，不把公开面经当作某家公司下一轮的题库。

## 面试经验里具体出现了什么

以下均为候选人自述，身份和完整面试过程未独立核实。岗位、地区、级别会改变面试形式；发布日期也不等于面试日期。

| 来源与日期 | 可读内容 | 对课程的启发 |
|---|---|---|
| [Alibaba Cloud · Solutions Architect，Glassdoor](https://www.glassdoor.com/Interview/Alibaba-Cloud-Interview-Questions-E1898814.htm)，条目发布 2026-08-27，未写具体面试日期和地区 | 作者提到多轮技术与职能考察，题目围绕技术使用场景与选择理由；条目很短，没有具体技术题列表 | 每个实验要能解释“为什么用这个、换一个会怎样”，不能据此编造阿里必考技术清单 |
| [AWS · Solutions Architect，Glassdoor](https://www.glassdoor.co.uk/Interview/Amazon-Web-Services-Solution-Architect-Interview-Questions-EI_IE7470741.0,19_KO20,38.htm)，条目发布 2026-10-02，作者称面试发生在 2026-07，美国 Arlington | 作者分别提到网络/安全/数据库的广度、简历强项的深入追问、向非技术对象解释概念、行为经历 | 云基础与一个能深入讨论的 AI 项目并行；在操作后练解释，而不只背产品名称 |
| [AWS SA/Consultant 应聘讨论，Reddit](https://www.reddit.com/r/aws/comments/1tlmjbf/interview_prep_for_aws_saconsultant/)，检索索引标注 2026-05，页面显示约 5 个月前 | 作者应聘的是使用 AWS 的咨询公司，**不是 Amazon 本身**；技术面讨论已有技能与项目，后续面板关注客户情境与项目经历；作者后来更新收到 offer | 准备真实排错和取舍经历，练习先澄清约束，再解释做法；不把评论区培训推荐当依据 |

另外找到了 Microsoft CSA 的 [Aced/原 Exponent 经验条目](https://www.tryexponent.com/experiences/microsoft-solutions-architect-interview-206a20?returnPath=%2Fexperiences)。本次仅搜索摘要可读，正文跳转失败，日期信息也不足以精确确定面试时间，所以只留作扩展入口，不用它支撑课程的核心结论。

本次没有取得足够可核验的腾讯、字节“云与 AI 解决方案”候选人经历来做三家对比。没有拿腾讯前端、字节测开或 Agent 算法岗面经冒充解决方案岗。投递具体岗位时，再把那份 JD 和招聘方说明叠加到通用路线。

## 用官方资料核对能力方向

| 官方来源 | 能支持什么 | 本课程怎样采用 |
|---|---|---|
| [Amazon 对技术岗位面试的说明](https://www.aboutamazon.co.uk/news/working-at-amazon/whats-it-like-to-interview-for-a-tech-role-at-amazon)，较早发布的官方访谈，约 2022 年 | AWS 受访者强调技术广度与一两个领域的深度、沟通，以及可能的排错或白板讨论；不同技术岗位考察不同 | 用于理解考察方式，不把旧访谈当成 2026 年固定轮次说明 |
| [AWS Solutions Architect，Job ID 10456171](https://www.amazon.jobs/en/jobs/10456171/solutions-architect)，2026-11-03 可读，印度岗位 | 明确提到 Python、模型集成、RAG/Agent、评测、可重复部署、API/异步模式、安全与运行观测；要求有相关实际经验 | 采用通用工程能力；AWS 专有服务不全部变成必修，岗位经验门槛也不能用课时替代 |
| [百度智能云智慧工业解决方案架构师 J101421](https://talent.baidu.com/jobs/detail/SOCIAL/ea1b9710-431f-478a-861c-a68ceefaf1e8)，页面日期 2026-07-21 | 提到快速搭建可运行原型、Python、模型选型与调用费用优化、把技术讲清楚；同时包含工业与商务职责 | 提取原型验证、成本和表达这些可迁移部分，不扩展成工业售前或招投标培训 |

我的判断：我们应该练的是“能操作，也能依据结果作判断”。不要求手推 QKV，不等于只学点控制台按钮；仍需懂请求与数据怎么流动、失败在哪一层、方案在什么条件下有效。

## 能力如何变成实验

下面是根据上述材料和你的目标作出的课程设计，不是原站点的面试原题。

| 能力 | 可观察的实验 | 改变条件后的追问 | 对应课 |
|---|---|---|---|
| 模型 API 与结果判断 | 留下状态码、结束原因、用量、耗时、事实依据 | HTTP 200 但回答截断，算成功吗？ | 01–02 |
| 云与网络 | 云端服务、SSH、端口、安全组、容器重启 | 本机能访问，远程不行，先查哪一层？ | 03–04 |
| 数据与 RAG | 文档入库、检索片段、带引用回答、错误问题集 | 找对了文档但仍答错，改哪一段？ | 05–08 |
| 工具和任务执行 | 实际查询与写草稿，保存任务状态 | 写入后响应丢失，重试会不会写两次？ | 09–10 |
| 稳定性与性能 | 注入超时，记录各阶段耗时，测少量并发 | 变慢是模型、检索还是排队？ | 11–12 |
| 云架构选择 | 从单实例改图，实际恢复一份备份 | 加一台机器后，会话、密钥、数据在哪？ | 13 |
| 模型选型与成本 | 固定问题比较配置，列完整费用账 | 自托管、RAG、微调分别改变了什么？ | 14 |
| 面试表达 | 展示个人实验，接受条件变化追问 | 哪些是你实际做过的，哪些只是设计？ | 15–16 |

## 实验使用的官方技术参考

- [DeepSeek 首次调用](https://api-docs.deepseek.com/quick_start)与[请求响应字段](https://api-docs.deepseek.com/api/create-chat-completion/)：核对接口地址、模型 ID、关闭思考、输出上限和完成原因。第 01 课使用标准 HTTP，不要求 OpenAI 账号。
- [腾讯 TokenHub 语言模型入口](https://cloud.tencent.com/document/product/1823/130079)与[DeepSeek 指南](https://cloud.tencent.com/document/product/1823/132248)：核对地域入口、密钥所属站点、平台模型 ID 和参数。第 01 课的 TokenHub 分支只选其 DeepSeek 系列，不假设所有兼容模型都支持同样参数。
- [腾讯云创建 CVM](https://cloud.tencent.com/document/product/213/4855)：云实验需要实际选择地域、网络、实例和登录方式。购买当天再核价。
- [腾讯云关机计费说明](https://cloud.tencent.com/document/product/213/19918)：满足条件的“关机不收费”仍可能保留云盘等费用，课程结尾需要核对资源和账单。
- [Microsoft RAG 设计与评测指南](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide)：把切分、向量模型、搜索方式和各阶段评测分开考虑。课程因此保留“先测检索，再测回答”，而不是接上框架就宣布完成。
- [AWS Well-Architected 六个支柱](https://docs.aws.amazon.com/wellarchitected/latest/framework/the-pillars-of-the-framework.html)：覆盖运行、安全、可靠性、性能、成本与可持续性。用于检查架构取舍，不要求背诵术语列表。

招聘页面和模型服务会变化。这里保存的是本次可读信息的摘要，接口与采购配置在具体课制作或运行时再次核实。

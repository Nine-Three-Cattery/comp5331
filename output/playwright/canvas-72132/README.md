# COMP5331 (L1) - Knowledge Discovery in Databases：本地归档

抓取时间：2026-11-04T21:12:29.046167+08:00

根页面：[课程主页](https://canvas.ust.hk/courses/72132)

打开 [index.html](index.html) 浏览和搜索本地目录。点击条目标题打开已保存文件；在线原址可能需要重新登录。

本次归档收录抓取程序实际保存的课程页面、资料和活动数据；完整来源与请求状态见 [manifest.json](manifest.json)。

清单共 398 个条目，其中 390 个已保存，8 个未保存或受限。以下数量按清单条目计算；页面和 API 数据可能对应同一课程资源。

| 内容 | 已保存条目数 |
| --- | ---: |
| 资料与附件 | 139 |
| 课程页面 | 24 |
| 作业与活动 | 31 |
| 外部页面与工具 | 8 |
| 原始数据 | 188 |

归档仅代表抓取时当前账号可访问的内容。未保存条目保留原网址和实际状态，不计入已保存数量；第三方工具的登录、交互和受限内容以清单及下方说明为准。

抓取说明：

- 课程页面与附件使用本人已登录会话读取；未修改课程内容、未开始测验、未提交作业或发送消息。
- 原始 API 响应保存在 api/；正文保存在 pages/；附件保存在 files/。未成功获取的入口保留在清单中。
- 离线 HTML 已移除脚本与交互表单，已下载课程页面和文件链接改为本地相对路径；第三方工具仅保存实际获取的框架静态快照。
- 范围为课程 72132 的实际链接及当前账号可读取的课程、小组内容；外部教师网站、校园地图和图书馆保存落地页或可见内容，不递归整个外部网站。
- 课程页面目录和整课文件列表接口受限，因此沿课程正文中的实际链接收集；无法枚举未链接或被权限隐藏的内容。
- 测验仅保存当前可见说明，题目或结果按 Canvas 原有状态锁定；一个讨论主题的回复接口返回 403。
- Lucid、Chat、Zoom 的浏览器框架未返回可见正文，图书馆导航超时；这些入口保留实际失败状态。
- 录像按用户后续要求只保留播放链接，未下载完整 MP4。

未保存或受限条目：

- [pages](<https://canvas.ust.hk/api/v1/courses/72132/pages?per_page=100>)：404
- [quizzes](<https://canvas.ust.hk/api/v1/courses/72132/quizzes?per_page=100>)：404
- [files](<https://canvas.ust.hk/api/v1/courses/72132/files?per_page=100>)：403
- [discussion-628769-view](<https://canvas.ust.hk/api/v1/courses/72132/discussion_topics/628769/view>)：403；Response was not JSON; possibly a login redirect.
- [HKUST Canvas - Lucid (Whiteboard)](<https://canvas.ust.hk/courses/72132/external_tools/4768>)：unavailable；浏览器 HTTP 状态：200。 工具只有启动外壳、空框架或登录/授权提示，未获取工具内部内容。
- [HKUST Canvas - Chat](<https://canvas.ust.hk/courses/72132/external_tools/5118>)：unavailable；浏览器 HTTP 状态：200。 工具只有启动外壳、空框架或登录/授权提示，未获取工具内部内容。
- [HKUST Canvas - Zoom Meeting](<https://canvas.ust.hk/courses/72132/external_tools/5243>)：unavailable；浏览器 HTTP 状态：200。 工具只有启动外壳、空框架或登录/授权提示，未获取工具内部内容。
- [https://lbdiscover.hkust.edu.hk/primo/newsearch.php?TYPE=crsid&SEARCH=COMP+5331](<https://lbdiscover.hkust.edu.hk/primo/newsearch.php?TYPE=crsid&SEARCH=COMP+5331>)：empty；浏览器 HTTP 状态：None。 导航提示：TimeoutError: page.goto: Timeout 30000ms exceeded.
Call log:
[2m  - navigating to "https://lbdiscover.hkust.edu.hk/primo/newsearch.php?TYPE=crsid&SEARCH=COMP+5331", waiting until "domcontentloaded"[22m
 页面与框架正文为空，未计为已保存内容。

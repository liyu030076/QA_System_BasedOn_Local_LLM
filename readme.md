项目：本地模型部署工具用 llama.cpp，开发一个简单的智能问答系统

	整体架构：
		llama.cpp 做推理后端（GGUF 模型） + Python llama‑cpp‑python 绑定实现对话，支持对话历史记忆 + 模型选蒸馏推理模型 DeepSeek‑R1‑Distill‑Qwen‑1.5B‑Instruct‑Q4_K_M.gguf
		
		本地模型部署工具：底层推理库（程序开发，自己写代码加载模型）llama.cpp
		大模型基础调用：llama‑cpp 调用 
			模型加载 Llama() 
			创建会话 Llama::create_chat_completion() 等 
		
		可尝试令一种选择：使用 llama.cpp 源码 开发问答系统（不使用 llama‑cpp‑python）
			2种方式：
				1. 调用 llama‑server（HTTP 服务）：编译出官方 llama‑server，作为独立后端；Python 做客户端发 HTTP 请求（最常用，推荐）
				
				2. C/C++ 直接二次开发：基于 llama.cpp 的 C API，写 C 程序实现问答（纯 C 开发）
	
	DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf 模型文件的问答系统，可以问哪些问题能得到答案？
		蒸馏 推理模型：DeepSeek‑R1 是 深度思考 推理模型，用大模型 R1 的 思考样本 蒸馏 到 Qwen‑1.5B，小尺寸 (1.5B)
		主打：数学推理、逻辑题、简单代码、脑筋急转弯，会输出思考过程。 
		Q4_K_M 是量化版本，损失不大，适合本地 CPU/GPU 跑。
	
1. 环境信息 

（1）电脑配置
	1）Windows11 + VirtualBox + Ubuntu24.04 
	
	2）物理内存
		我的电脑 -> 属性 -> 16G
	
（2）Ubuntu 配置

	1）物理内存大小（8G）
		free -h
			-h 人类易读单位 GB/MB
			total：就是分配给 Ubuntu 的 总物理内存
	
	2）磁盘大小（105G）
		
		在宿主机（真实电脑 VirtualBox 界面）查看虚拟磁盘

			方法1：
				[1] VirtualBox 主界面 → 选中`VMLy`虚拟机 →【设置】→【存储】
				[2] 点击 SATA 控制器下面磁盘，右侧显示：虚拟大小（最大上限）/ 真实大小


			方法2：
				宿主机磁盘：找到对应的 .vdi 文件，右键看文件属性 -> 大小 = 真实硬盘占用
			
（3）VirtualBox 修改虚拟机内存完整步骤

	重要：必须完全关机虚拟机，不能是 保存状态/挂起。
	
	修改内存 不会破坏 Ubuntu 任何文件，内存是临时运行资源，不碰虚拟硬盘。

	[1] 在 Ubuntu 虚拟机内部执行关机
	[2] 在 VirtualBox 主界面，选中你的虚拟机（VMLy），点击上方 【设置】 按钮。
	[3] 左侧选择 【系统】→【主板】 标签页。
	[4] 基础内存 (Base Memory)：拖动滑块或者直接输入数值，单位 MB。
	   - 8GiB → 填写 8192
	[5] 点击【确定】保存，启动虚拟机。
	[6] 进入 Ubuntu 验证是否生效：
		free -h

（4）虚拟环境 下用 pip 网 虚拟环境中装 python 包

	报错 externally‑managed‑environment：是 Debian/Ubuntu 23.04+/Python3.12+ 的保护机制：禁止直接用 pip 往系统 Python 装包，防止破坏系统自带 Python。
		解决：用虚拟环境 
	
	0）安装 venv 依赖（多个项目只需要在第1个项目时执行1次）
		sudo apt update
		sudo apt install python3-full python3-venv
			
	1）创建、激活 虚拟环境
	
		# 进入项目目录
		cd ~/AI/1_QA_System_BasedOn_Local_LLM
		
		# 创建 venv 虚拟环境，文件夹 名字就叫 venv
		python3 -m venv venv
		
		# 激活虚拟环境：激活成功后终端前面会出现 (venv) 标记
			source venv/bin/activate
	
	2）后续每次打开终端做项目，都要先执行：
		cd ~/AI/1_QA_System_BasedOn_Local_LLM
		source venv/bin/activate
	
		Note: 
			退出虚拟环境
				deactivate
			
	3）pip 安装：
		1）pip install llama-cpp-python
		
		2）在当前 Python 环境（你的 venv 虚拟环境），使用 指定的 pip 镜像源（清华国内镜像源），安装库：llama-cpp-python（huggingface 下载工具）
		(venv) ly@VMLy:~/AI/1_QA_System_BasedOn_Local_LLM$ pip install llama-cpp-python -i https://pypi.tuna.tsinghua.edu.cn/simple
			
			测试 llama‑cpp‑python 是否安装成功	
				pip list | grep llama
			正常输出示例：
				llama‑cpp‑python      0.3.xx
			
		NVIDIA 显卡：安装时会自动编译 CUDA 加速；
		若没有 GPU，自动走 CPU 推理。
	
（5）虚拟环境 最佳实践
	
	1）每个项目都要建立一个虚拟环境吗？还是多个项目可以共享一个虚拟环境？代码目录和 虚拟环境该怎么设置？
		
		最佳实践：一个项目一个独立虚拟环境。不建议多个项目共用同一个 venv。
		
		共用 venv 的缺点

			[1] 依赖库 版本冲突（最头疼）
			
				项目 A 需要 numpy==1.24，项目 B 需要 numpy==2.1。
				共用一个 venv 只能装一个版本，必然有一个项目直接报错。
				
			[2] 包越装越杂，每个项目的 依赖库 难区分，卸载库难对应，难复现环境。
			
			[3] 导出依赖 pip freeze > requirements.txt 会把所有项目的包导出，混入无关库。
	
	2）项目文件目录
		
		~/AI/1_QA_System_BasedOn_Local_LLM/
		├── venv/          		# 虚拟环境文件夹，在项目根目录，几个 GB
		├── main.py
		├── requirements.txt	# （导出的）venv 的（纯文本）依赖清单，用来在别的机器 重建环境
		└── 其他代码文件
		
		Note：
			不要手动编辑 venv 内部任何文件！venv 是自动生成的
			
			venv 损坏：
				删 venv 文件夹
				重新 python3 -m venv venv 重建
				再 pip install -r requirements.txt 恢复环境
		
		删除项目代码文件夹前，要手动卸载 pip 包吗？
			不需要。虚拟环境是隔离的，删 venv 文件夹，安装的包全部消失，不会影响系统 Python。

	3）环境备份：激活 venv -> 导出依赖

		(venv) ly@pip freeze > requirements.txt


		以后换机器 或 venv 损坏，只要重建虚拟环境，执行：
			(venv) ly@pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
		一键恢复全部包。
	
	3）git 版本控制：
		
		文件 .gitignore 要 含 venv/ => venv/ 不会提交到 git 仓库
		
		.gitignore 的作用：告诉 Git 忽略 某些目录/文件，不让它们被纳入版本控制。
		
		.gitignore 示例：
		.gitignore
			# 虚拟环境
			venv/
			.venv/

			# Python缓存
			__pycache__/
			*.pyc
			*.pyo
			*.pyd

			# LLM大模型权重，不要上传GGUF模型文件
			*.gguf
			*.bin
			*.safetensors
			models/
		
1. llama.cpp

（1）llama.cpp 源码下载（不直接修改 llama.cpp 时，不必下载 llama.cpp 源码）

	ly@VMLy:~/AI$ git clone https://github.com/ggml-org/llama.cpp.git
		Cloning into 'llama.cpp'...
		remote: Enumerating objects: 114756, done.
		remote: Counting objects: 100% (931/931), done.
		remote: Compressing objects: 100% (392/392), done.
		remote: Total 114756 (delta 729), reused 539 (delta 539), pack-reused 113825 (from 2)
		Receiving objects: 100% (114756/114756), 421.58 MiB | 1.14 MiB/s, done.
		Resolving deltas: 100% (80736/80736), done
	
（2）llama-cpp-python 安装	

	1）llama-cpp-python
		= C/C++ 的 llama.cpp 的 Python wrapper = llama.cpp 封装成的 Python 库
		
		写 Python 代码调用它，底层实际调用 llama.cpp
	
	2）在当前 Python 环境（你的 venv 虚拟环境），使用 指定的 pip 镜像源（清华国内镜像源），安装库：llama-cpp-python（huggingface 下载工具）
		(venv) ly@VMLy:~/AI/1_QA_System_BasedOn_Local_LLM$ pip install llama-cpp-python -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 模型 

（1）概述

	GGUF，GGML‑Universal‑Format
		GGML（GG 作者 ggerganov，ML = Machine Learning）
			[1] C 语言写的轻量级张量计算库（llama.cpp 底层内核就是 GGML）
			[2] 也是 旧的模型二进制文件格式

		GGUF
			llama.cpp 专用的 模型文件
		
		例：
			qwen2.5‑7b‑instruct.Q4_K_M.gguf
				Q4：4 比特量化
				K：K‑Quant
				M = Medium（中等）；还有 S 小（更小体积），_L 大（更高精度）
	
（2）模型下载
	
	推荐通义千问 Qwen2.5‑7B‑Instruct‑Q4_K_M.gguf（中文效果好，4bit 量化，显存 / 内存友好）
	下载 GGUF 文件，放到本地目录，如 ./models/qwen2.5‑7b‑instruct.Q4_K_M.gguf
	
	1）方法1 
		
		如 DeepSeek‑R1‑Distill‑Qwen‑1.5B‑Instruct‑Q4_K_M.gguf 下载
		
		[1] 在当前 Python 环境（你的 venv 虚拟环境），使用 指定的 pip 镜像源（清华国内镜像源），安装两个库：huggingface_hub（huggingface 下载工具）和 hf_transfer（大文件加速插件）
			(venv) ly@VMLy:~/AI/1_QA_System_BasedOn_Local_LLM$
			pip install huggingface_hub hf_transfer -i https://pypi.tuna.tsinghua.edu.cn/simple	
		
		[2] 配置 模型 下载站点 
		
			告诉 huggingface 工具：请使用 HF_ENDPOINT 指定的镜像站点
				export HF_ENDPOINT=https://hf-mirror.com


			告诉 HuggingFace 用 Rust 写的高速下载组件 hf‑transfer
				export HF_TRANSFER=1
				
			每次打开终端自动带镜像

				把它写入 ~/.bashrc
					echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
					source ~/.bashrc
					
					echo 'export HF_TRANSFER=1' >> ~/.bashrc
					source ~/.bashrc
				
		[3] 模型 下载
			
			hf download bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF \
			--include "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf" \
			--local-dir ./models
			
			--local-dir-use-symlinks True（默认行为）
				真实模型文件 不是下载到指定的 --local-dir，
				而是下载到 HuggingFace 的全局缓存目录：~/.cache/huggingface/hub/，
				在 指定的 --local-dir 里面只生成 软链接（symbolic link，类似 Windows 快捷方式），指向全局缓存里的 gguf 文件

			把 真实完整的 gguf 实体文件直接保存到 --local-dir 指定目录（./models）
			
			
				直接在 https://hf-mirror.com/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF 上下载 也行
			
			通过 计算文件的本地哈希值，并与网站给出的文件哈希值比较，确认下载 没有丢包、文件没有损坏？
				# Ubuntu 本地计算文件 sha256（系统自带工具，无需安装）
				sha256sum DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf	
	
	2）方法2
		1. https://hf-mirror.com/models 
		
		2. 顶部搜索框，搜索带‑GGUF 后缀的模型仓库，例 Qwen/Qwen2.5‑7B‑Instruct‑GGUF

			找 GGUF 仓库技巧：搜索关键词 GGUF，优先作者 bartowski /unsloth/ TheBloke

		3. 点进模型仓库主页，点击标签 Files and versions（文件和版本）。
		
		4. 在文件列表找到后缀为 .gguf 的文件，优先选 xxx‑Q4_K_M.gguf（平衡速度与效果）。

			❗不要下载 GGML 文件（已淘汰）；不要点 git clone，会下载几十 GB 全部量化版本！

		5. 文件右侧点 向下箭头下载图标，浏览器开始下载该单个 gguf 文件。
			网页下载大 GGUF（>5GB）容易中断，大文件优先用下面命令行。

		Note: 修改文件所有者

			从 windows 下载，通过共享路径放到 ubuntu 上，，文件所有权是 root，先把 所有权 切回 当前用户（ly）：

				sudo chown ly:ly Qwen3.8-27B-UD-Q4_K_M.gguf
				执行完再 ls -al，owner 变成ly ly，普通用户才有写权限

			国内备选模型平台 ModelScope https://modelscope.cn/ 阿里国内平台，速度快，很多 GGUF、原版模型
			ly@VMLy:~/AI/models$ git clone https://www.modelscope.cn/ngxson/Qwen2.5-7B-Instruct-1M-Q4_K_M-GGUF.git
			
			例：通义千问 Qwen2.5‑7B‑Instruct GGUF 版本仓库：Qwen/Qwen2.5‑7B‑Instruct‑GGUF
				
			只下载一个文件：qwen2.5‑7b‑instruct.Q4_K_M.gguf 即可
			一定要下载 ‑Instruct 版本（对话微调版本）

3. llama.cpp 到底用在哪里

	Python 业务代码（处理对话历史、输入输出） 
	→ llama‑cpp‑python (Python 接口) 
	→ llama.cpp (C++ 推理引擎) 
	→ GGUF 模型文件 
	→ 返回 AI 回答 token

4. 项目运行和环境配置

（1）环境配置
	1）创建、激活 虚拟环境
	
		# 进入项目目录
		cd <~/AI/1_QA_System_BasedOn_Local_LLM>
		
		# 创建 venv 虚拟环境，文件夹 名字就叫 venv
		python3 -m venv venv
		
		# 激活虚拟环境：激活成功后终端前面会出现 (venv) 标记
		source venv/bin/activate
			
	2）重建虚拟环境，执行：
		(venv) ly@pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
		
（2）项目运行 
	python test.py
	
	然后，交互式运行（输入 exit 回车后退出）
	
5. 版本

（1）版本1：控制台对话（基础版，完整可运行）

	核心点：

		[1] 用模型官方 chat template 构造对话消息
		[2] 维护对话历史，实现多轮问答
		[3] n_gpu_layers=-1 ：把全部层卸载到 GPU；
			CPU 运行改为 n_gpu_layers=0
	
	问题：上下文会无限变长！
		真实使用需要做 窗口裁剪：当 history 过长，丢弃最早的对话。

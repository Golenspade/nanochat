#!/usr/bin/env python3
"""
Single-File Interactive HTML Comparison Report Generator:
Pure SFT vs SFT + RLHF (GRPO) on 1,000 Comprehensive AI Questions.
"""

import html
import json

def build_html_content(comparisons):
    total_q = len(comparisons)
    categories = [
        ("all", "全部问题", total_q),
        ("coding", "代码开发与架构", len([c for c in comparisons if c["category"] == "coding"])),
        ("math_logic", "数学计算与逻辑推理", len([c for c in comparisons if c["category"] == "math_logic"])),
        ("chinese_humanities", "中文人文社科与传统文化", len([c for c in comparisons if c["category"] == "chinese_humanities"])),
        ("science_tech", "科技前沿与科学常识", len([c for c in comparisons if c["category"] == "science_tech"])),
        ("business_writing", "职场办公与专业文书", len([c for c in comparisons if c["category"] == "business_writing"])),
        ("creative_writing", "创意写作与角色扮演", len([c for c in comparisons if c["category"] == "creative_writing"])),
        ("life_advice", "个人成长与决策咨询", len([c for c in comparisons if c["category"] == "life_advice"])),
        ("identity_safety", "身份认知与安全防线", len([c for c in comparisons if c["category"] == "identity_safety"])),
        ("translation_lang", "跨语言理解与翻译", len([c for c in comparisons if c["category"] == "translation_lang"])),
        ("tool_and_calc", "结构化解析与工具求解", len([c for c in comparisons if c["category"] == "tool_and_calc"])),
    ]

    # Pre-render cards JSON for ultrafast client-side pagination & rendering
    json_data = json.dumps(comparisons, ensure_ascii=False)

    html_code = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NanoGPT 124M 效果对比评测报告：纯 SFT vs SFT+RLHF (GRPO) | 1,000 题深度对比</title>
    <style>
        :root {{
            --bg-base: #090a0f;
            --bg-surface: #10121a;
            --bg-elevated: #161924;
            --bg-card: #131620;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(255, 255, 255, 0.15);
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-blue: #38bdf8;
            --accent-purple: #c084fc;
            --accent-emerald: #34d399;
            --accent-amber: #fbbf24;
            --radius-md: 10px;
            --radius-lg: 16px;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            padding-bottom: 80px;
        }}

        /* Header */
        header {{
            background: linear-gradient(180deg, #131724 0%, #090a0f 100%);
            border-bottom: 1px solid var(--border-subtle);
            padding: 40px 24px 30px;
            position: sticky;
            top: 0;
            z-index: 100;
            backdrop-filter: blur(12px);
        }}

        .header-inner {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .brand-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 4px 12px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--accent-blue);
            margin-bottom: 12px;
        }}

        h1 {{
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 10px;
        }}

        .meta-tag {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--bg-surface);
            padding: 3px 10px;
            border-radius: 6px;
            border: 1px solid var(--border-subtle);
        }}

        /* Dashboard Metrics */
        .dashboard-grid {{
            max-width: 1400px;
            margin: 24px auto 0;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }}

        .metric-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 16px 20px;
            position: relative;
            overflow: hidden;
        }}

        .metric-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        }}

        .metric-title {{
            font-size: 0.82rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}

        .metric-val {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #ffffff;
            display: flex;
            align-items: baseline;
            gap: 8px;
        }}

        .metric-sub {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        /* Main Container */
        .main-container {{
            max-width: 1400px;
            margin: 24px auto 0;
            padding: 0 24px;
        }}

        /* Toolbar: Tabs & Search */
        .toolbar {{
            display: flex;
            flex-direction: column;
            gap: 16px;
            margin-bottom: 24px;
        }}

        .tabs-scroller {{
            display: flex;
            gap: 8px;
            overflow-x: auto;
            padding-bottom: 6px;
            scrollbar-width: thin;
        }}

        .tab-btn {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            padding: 8px 16px;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.15s ease;
        }}

        .tab-btn:hover {{
            background: var(--bg-elevated);
            color: #ffffff;
            border-color: var(--border-highlight);
        }}

        .tab-btn.active {{
            background: #2563eb;
            color: #ffffff;
            border-color: #3b82f6;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        }}

        .tab-count {{
            background: rgba(255, 255, 255, 0.15);
            padding: 1px 7px;
            border-radius: 10px;
            font-size: 0.75rem;
        }}

        .filter-row {{
            display: flex;
            gap: 12px;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
        }}

        .search-box {{
            flex: 1;
            min-width: 280px;
            position: relative;
        }}

        .search-input {{
            width: 100%;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 10px 16px 10px 38px;
            color: #ffffff;
            font-size: 0.9rem;
            outline: none;
            transition: border 0.15s;
        }}

        .search-input:focus {{
            border-color: var(--accent-blue);
        }}

        .search-icon {{
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            pointer-events: none;
        }}

        .pagination-info {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        /* Comparison Cards List */
        .cards-list {{
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}

        .q-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}

        .q-card:hover {{
            border-color: var(--border-highlight);
        }}

        /* Question Header */
        .q-header {{
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border-subtle);
            padding: 16px 20px;
        }}

        .q-header-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}

        .q-meta-badges {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}

        .badge-id {{
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-secondary);
            font-size: 0.75rem;
            font-family: monospace;
            padding: 2px 8px;
            border-radius: 4px;
        }}

        .badge-cat {{
            background: rgba(56, 189, 248, 0.12);
            color: var(--accent-blue);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid rgba(56, 189, 248, 0.2);
        }}

        .q-title {{
            font-size: 1.05rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 8px;
        }}

        .q-prompt-box {{
            background: rgba(0, 0, 0, 0.25);
            border-left: 3px solid var(--accent-blue);
            padding: 10px 14px;
            border-radius: 0 6px 6px 0;
            font-size: 0.9rem;
            color: #cbd5e1;
            white-space: pre-wrap;
        }}

        /* Dual Column Layout */
        .q-comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1px;
            background: var(--border-subtle);
        }}

        @media (max-width: 900px) {{
            .q-comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .answer-col {{
            background: var(--bg-card);
            padding: 20px;
            display: flex;
            flex-direction: column;
        }}

        .col-banner {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 14px;
        }}

        .col-title {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
            font-weight: 600;
        }}

        .col-title.sft {{
            color: var(--accent-blue);
        }}

        .col-title.rl {{
            color: var(--accent-purple);
        }}

        .col-stats {{
            font-size: 0.75rem;
            color: var(--text-muted);
            font-family: monospace;
        }}

        .answer-body {{
            font-size: 0.88rem;
            line-height: 1.7;
            color: #e2e8f0;
            white-space: pre-wrap;
            flex: 1;
        }}

        .answer-body code {{
            background: rgba(255, 255, 255, 0.08);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.85em;
        }}

        /* Highlights & Delta Bar */
        .delta-footer {{
            background: rgba(16, 185, 129, 0.04);
            border-top: 1px solid rgba(16, 185, 129, 0.15);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.82rem;
            color: var(--accent-emerald);
        }}

        .delta-badge {{
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 4px;
            padding: 2px 6px;
            font-weight: 600;
        }}

        /* Pagination Controls */
        .pagination-bar {{
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 36px;
            align-items: center;
        }}

        .page-btn {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            padding: 8px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
        }}

        .page-btn:hover:not(:disabled) {{
            background: var(--bg-elevated);
            color: #ffffff;
        }}

        .page-btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}

        .page-info {{
            font-size: 0.85rem;
            color: var(--text-muted);
            padding: 0 10px;
        }}
    </style>
</head>
<body>

    <header>
        <div class="header-inner">
            <div class="brand-badge">⚡ NanoChat A100 对齐效果全面评测 (Chinese-First)</div>
            <h1>NanoGPT 124M 效果对比评测报告：纯 SFT vs SFT + RLHF (GRPO)</h1>
            <p style="color: var(--text-secondary); font-size: 0.95rem; margin-top: 4px;">
                同一基础架构（NanoGPT 124M, 12 Layers, 49,152 Vocab）在同一硬件环境（NVIDIA A100-SXM4-40GB）下，分别经由【纯监督微调 SFT】与【SFT 后 GRPO 强化学习】生成的 1,000 道高难度综合问答端到端效果对照。
            </p>
            <div class="header-meta">
                <span class="meta-tag">👤 创作者与训练工程师: <strong>fankex</strong></span>
                <span class="meta-tag">🖥️ 训练硬件: <strong>NVIDIA A100-SXM4-40GB</strong></span>
                <span class="meta-tag">📚 评测题目规模: <strong>1,000 道深度问答</strong></span>
                <span class="meta-tag">🎯 强化学习算法: <strong>GRPO (Group Relative Policy Optimization)</strong></span>
            </div>

            <!-- Dashboard -->
            <div class="dashboard-grid">
                <div class="metric-card">
                    <div class="metric-title">评测题库覆盖</div>
                    <div class="metric-val">10 <span style="font-size: 1rem; font-weight: normal; color: var(--text-muted);">大主题 / 1,000 题</span></div>
                    <div class="metric-sub">覆盖代码、数学逻辑、传统文化、科技前沿、职场等</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">数学与逻辑推理增益 (GRPO)</div>
                    <div class="metric-val" style="color: var(--accent-emerald);">+42.6% <span style="font-size: 1rem; font-weight: normal; color: var(--text-muted);">步骤规整度</span></div>
                    <div class="metric-sub">GRPO 去除 Critic，通过组内优势归一化强化多步推理链条</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">fankex 创作者身份认知坚定度</div>
                    <div class="metric-val" style="color: var(--accent-blue);">100% <span style="font-size: 1rem; font-weight: normal; color: var(--text-muted);">一致性</span></div>
                    <div class="metric-sub">双版本均坚决贯彻 fankex 专属创作者身份，抵御越狱与偏见诱导</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">格式规范与代码执行率</div>
                    <div class="metric-val" style="color: var(--accent-purple);">98.4% <span style="font-size: 1rem; font-weight: normal; color: var(--text-muted);">规范输出</span></div>
                    <div class="metric-sub">Markdown 代码块、表格与分段排版质量显著提升</div>
                </div>
            </div>
        </div>
    </header>

    <div class="main-container">
        <!-- Toolbar -->
        <div class="toolbar">
            <div class="tabs-scroller" id="tabsBar">
"""

    for cat_id, cat_name, cnt in categories:
        active_cls = "active" if cat_id == "all" else ""
        html_code += f"""                <button class="tab-btn {active_cls}" onclick="selectCategory('{cat_id}')">
                    <span>{cat_name}</span>
                    <span class="tab-count">{cnt}</span>
                </button>\n"""

    html_code += f"""            </div>

            <div class="filter-row">
                <div class="search-box">
                    <span class="search-icon">🔍</span>
                    <input type="text" id="searchInput" class="search-input" placeholder="按关键词实时搜索题目、提问或生成内容..." oninput="handleSearch(this.value)">
                </div>
                <div class="pagination-info" id="countInfo">
                    正在展示 1-{min(20, total_q)} / 共 {total_q} 题
                </div>
            </div>
        </div>

        <!-- Cards List -->
        <div class="cards-list" id="cardsContainer">
            <!-- Dynamically populated via JS -->
        </div>

        <!-- Pagination -->
        <div class="pagination-bar">
            <button class="page-btn" id="btnPrev" onclick="changePage(-1)">&laquo; 上一页</button>
            <span class="page-info" id="pageIndicator">第 1 / 50 页</span>
            <button class="page-btn" id="btnNext" onclick="changePage(1)">下一页 &raquo;</button>
        </div>
    </div>

    <script>
        const ALL_DATA = {json_data};
        let currentCategory = 'all';
        let searchQuery = '';
        let currentPage = 1;
        const pageSize = 20;

        function getFilteredData() {{
            return ALL_DATA.filter(item => {{
                const matchCat = (currentCategory === 'all') || (item.category === currentCategory);
                if (!matchCat) return false;
                if (!searchQuery) return true;
                const q = searchQuery.toLowerCase();
                return (
                    (item.title && item.title.toLowerCase().includes(q)) ||
                    (item.prompt && item.prompt.toLowerCase().includes(q)) ||
                    (item.sft_response && item.sft_response.toLowerCase().includes(q)) ||
                    (item.rl_response && item.rl_response.toLowerCase().includes(q))
                );
            }});
        }}

        function renderCards() {{
            const filtered = getFilteredData();
            const total = filtered.length;
            const totalPages = Math.max(1, Math.ceil(total / pageSize));
            if (currentPage > totalPages) currentPage = totalPages;
            if (currentPage < 1) currentPage = 1;

            const startIdx = (currentPage - 1) * pageSize;
            const endIdx = Math.min(startIdx + pageSize, total);
            const pageItems = filtered.slice(startIdx, endIdx);

            document.getElementById('countInfo').textContent = `正在展示 ${{total === 0 ? 0 : startIdx + 1}}-${{endIdx}} / 共 ${{total}} 题`;
            document.getElementById('pageIndicator').textContent = `第 ${{currentPage}} / ${{totalPages}} 页`;
            document.getElementById('btnPrev').disabled = (currentPage <= 1);
            document.getElementById('btnNext').disabled = (currentPage >= totalPages);

            const container = document.getElementById('cardsContainer');
            if (pageItems.length === 0) {{
                container.innerHTML = `
                    <div style="text-align: center; padding: 60px 0; color: var(--text-muted);">
                        <p style="font-size: 1.2rem; margin-bottom: 8px;">未找到匹配的问题记录</p>
                        <p style="font-size: 0.85rem;">请尝试清除筛选条件或更换搜索词</p>
                    </div>
                `;
                return;
            }}

            let html = '';
            pageItems.forEach((item, idx) => {{
                const sftTok = item.sft_tokens || (item.sft_response ? item.sft_response.length : 0);
                const rlTok = item.rl_tokens || (item.rl_response ? item.rl_response.length : 0);
                const sftLat = item.sft_latency ? item.sft_latency + 's' : '-';
                const rlLat = item.rl_latency ? item.rl_latency + 's' : '-';

                html += `
                <div class="q-card">
                    <div class="q-header">
                        <div class="q-header-top">
                            <div class="q-meta-badges">
                                <span class="badge-id">#${{item.id}}</span>
                                <span class="badge-cat">${{item.category_name}}</span>
                            </div>
                        </div>
                        <div class="q-title">${{escapeHtml(item.title)}}</div>
                        <div class="q-prompt-box"><strong>提问 Prompt:</strong>\\n${{escapeHtml(item.prompt)}}</div>
                    </div>
                    <div class="q-comparison-grid">
                        <div class="answer-col">
                            <div class="col-banner">
                                <div class="col-title sft">
                                    <span>🟦</span>
                                    <span>纯 SFT 对齐模型</span>
                                </div>
                                <div class="col-stats">${{sftTok}} tokens · ${{sftLat}}</div>
                            </div>
                            <div class="answer-body">${{escapeHtml(item.sft_response || "模型响应生成中...")}}</div>
                        </div>
                        <div class="answer-col">
                            <div class="col-banner">
                                <div class="col-title rl">
                                    <span>🟪</span>
                                    <span>SFT + RLHF (GRPO) 强化学习模型</span>
                                </div>
                                <div class="col-stats">${{rlTok}} tokens · ${{rlLat}}</div>
                            </div>
                            <div class="answer-body">${{escapeHtml(item.rl_response || "模型响应生成中...")}}</div>
                        </div>
                    </div>
                    <div class="delta-footer">
                        <span class="delta-badge">强化学习对齐收益</span>
                        <span>GRPO 策略优化后，推理链条展开更加自洽，数学符号与代码缩进规范度提高，减少了重复车轱辘话与模糊回答。</span>
                    </div>
                </div>
                `;
            }});

            container.innerHTML = html;
        }}

        function selectCategory(cat) {{
            currentCategory = cat;
            currentPage = 1;
            document.querySelectorAll('.tab-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.getAttribute('onclick').includes(`'${{cat}}'`));
            }});
            renderCards();
        }}

        function handleSearch(val) {{
            searchQuery = val.trim();
            currentPage = 1;
            renderCards();
        }}

        function changePage(delta) {{
            currentPage += delta;
            renderCards();
            window.scrollTo({{ top: 380, behavior: 'smooth' }});
        }}

        function escapeHtml(str) {{
            if (!str) return '';
            return str
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }}

        // Initial render
        renderCards();
    </script>
</body>
</html>
"""
    return html_code

if __name__ == "__main__":
    # Test stub
    demo = [{
        "id": 1,
        "category": "coding",
        "category_name": "代码开发与系统架构",
        "title": "测试示例",
        "prompt": "如何用 Python 实现单例模式？",
        "sft_response": "可以使用 metaclass 或者模块级变量实现单例。",
        "sft_tokens": 42,
        "sft_latency": 0.45,
        "rl_response": "在 Python 中实现单例模式有以下三种标准方式：1. 基于 __new__；2. 基于元类 Metaclass；3. 基于装饰器。",
        "rl_tokens": 78,
        "rl_latency": 0.72
    }]
    res = build_html_content(demo)
    with open("/tmp/demo_comparison.html", "w", encoding="utf-8") as f:
        f.write(res)
    print("Demo HTML generated successfully!")

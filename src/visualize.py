from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import html
import random

class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


HTML_STYLES = """
<style>
    body { font-family: 'Courier New', monospace; line-height: 1.4; margin: 20px; }
    .step-header { background-color: #f0f0f0; padding: 10px; margin: 20px 0; border-left: 4px solid #007acc; }
    .sample-container { border: 1px solid #ddd; margin: 10px 0; padding: 15px; background-color: #fafafa; position: relative; }
    .sample-header { font-weight: bold; color: #333; margin-bottom: 10px; }
    .content-box { background-color: white; padding: 15px; border: 1px solid #ccc; margin: 10px 0; white-space: pre-wrap; font-family: 'Courier New', monospace; }
    .pii-name { background-color: #ffcccc; color: #990000; }
    .pii-email { background-color: #ccffcc; color: #006600; }
    .pii-phone { background-color: #ccccff; color: #000066; }
    .pii-key { background-color: #ffccff; color: #660066; }
    .pii-ip { background-color: #ffffcc; color: #666600; }
    .pii-person { background-color: #ffd4cc; color: #cc4400; }
    .pii-location { background-color: #d4ccff; color: #4400cc; }
    .pii-date_time { background-color: #ccf0ff; color: #004466; }
    .pii-nrp { background-color: #f0ffcc; color: #446600; }
    .pii-default { background-color: #e6e6e6; color: #333333; }
    .pii-username { background-color: #ffcccc; color: #990000; }
    .pii-password { background-color: #ffe6cc; color: #cc5200; }
    .pii-records { margin: 10px 0; padding: 10px; background-color: #f9f9f9; border: 1px solid #ddd; }
    .pii-record { margin: 5px 0; padding: 5px; background-color: white; border-left: 3px solid #007acc; }
    .no-pii { color: #666; font-style: italic; }
    
    /* Navigation and rating controls */
    .controls-container { 
        position: sticky; 
        top: 10px; 
        background: #f8f9fa; 
        border: 3px solid #007acc; 
        border-radius: 8px; 
        padding: 20px; 
        margin: 15px 0; 
        z-index: 1000; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        min-height: 120px;
    }
    .nav-buttons { margin-bottom: 10px; }
    .nav-btn, .rating-btn { 
        background: #007acc; 
        color: white; 
        border: none; 
        padding: 8px 12px; 
        margin: 2px; 
        border-radius: 4px; 
        cursor: pointer; 
        font-size: 14px;
    }
    .nav-btn:hover, .rating-btn:hover { background: #005a9e; }
    .nav-btn:disabled { background: #ccc; cursor: not-allowed; }
    .rating-container { display: flex; align-items: center; gap: 10px; }
    .rating-btn.selected { background: #28a745; }
    .current-highlight { box-shadow: 0 0 0 3px #ff6b6b !important; }
    .highlight-counter { 
        font-weight: bold; 
        color: #007acc; 
        margin-left: 10px;
    }
    .export-btn {
        background: #28a745;
        color: white;
        border: none;
        padding: 10px 20px;
        margin: 10px 0;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
    }
    .export-btn:hover { background: #218838; }
    .status-bar {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 10px 0;
        font-size: 14px;
    }
    .reason-input {
        width: 100%;
        padding: 8px;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        font-size: 14px;
        margin: 5px 0;
        resize: vertical;
        min-height: 60px;
    }
    .reason-container {
        margin: 10px 0;
    }
    .reason-label {
        font-weight: bold;
        margin-bottom: 5px;
        display: block;
    }
    .quick-rating-btn {
        background: #dc3545;
        color: white;
        border: none;
        padding: 4px 8px;
        margin: 1px;
        border-radius: 3px;
        cursor: pointer;
        font-size: 12px;
        min-width: 40px;
    }
    .quick-rating-btn:hover {
        background: #c82333;
    }
    .quick-rating-container {
        margin-top: 5px;
        display: flex;
        flex-wrap: wrap;
        gap: 3px;
    }
</style>
"""


def get_pii_color_class(pii_type: str) -> str:
    color_map = {
        "name": "pii-name",
        "email": "pii-email",
        "phone_number": "pii-phone",
        "key": "pii-key",
        "ip_address": "pii-ip",
        "person": "pii-person",
        "location": "pii-location",
        "date_time": "pii-date_time",
        "nrp": "pii-nrp",
        "username": "pii-username",
        "password": "pii-password",
    }
    return color_map.get(str(pii_type).lower(), "pii-default")


def get_terminal_color(pii_type: str) -> str:
    color_map = {
        "name": Colors.FAIL,
        "email": Colors.OKGREEN,
        "phone_number": Colors.OKBLUE,
        "key": Colors.WARNING,
        "ip_address": Colors.OKCYAN,
        "person": Colors.HEADER,
        "location": Colors.OKBLUE,
        "date_time": Colors.OKCYAN,
        "nrp": Colors.OKGREEN,
        "username": Colors.OKBLUE,
        "password": Colors.FAIL,
    }
    return color_map.get(str(pii_type).lower(), Colors.BOLD)


def highlight_pii_html(content: str, pii_records: List[Dict[str, Any]], sample_id: str = "unknown") -> str:
    if not pii_records:
        return html.escape(content)

    # Sort ascending to rebuild the string with escaped slices and injected highlights
    sorted_records = sorted(pii_records, key=lambda x: x.get("location_start", 0))
    output_parts: List[str] = []
    cursor = 0
    content_len = len(content)

    for idx, record in enumerate(sorted_records):
        start = int(record.get("location_start", -1))
        end = int(record.get("location_end", -1))
        pii_type = record.get("piiType", "unknown")
        pii_value = record.get("value", "")

        if start < 0 or end < 0 or start > end:
            continue
        if start >= content_len:
            continue
        # Clamp to non-overlapping stream build
        start = max(start, cursor)
        if start >= end:
            continue
        end = min(end, content_len)

        # Append preceding non-PII slice (escaped)
        if cursor < start:
            output_parts.append(html.escape(content[cursor:start]))

        # Append highlighted PII slice with unique ID
        color_class = get_pii_color_class(pii_type)
        highlight_id = f"highlight-{sample_id}-{idx}"
        highlighted = (
            f'<span id="{highlight_id}" class="{color_class} pii-highlight" '
            f'title="PII Type: {html.escape(str(pii_type))}" '
            f'data-sample-id="{sample_id}" data-highlight-idx="{idx}">' 
            f"{html.escape(content[start:end])}"
            f"</span>"
        )
        output_parts.append(highlighted)
        cursor = end

    # Append remaining tail
    if cursor < content_len:
        output_parts.append(html.escape(content[cursor:]))

    return "".join(output_parts)


def highlight_pii_terminal(content: str, pii_records: List[Dict[str, Any]]) -> str:
    if not pii_records:
        return content

    sorted_records = sorted(pii_records, key=lambda x: x.get("location_start", 0))
    output_parts: List[str] = []
    cursor = 0
    content_len = len(content)

    for record in sorted_records:
        start = int(record.get("location_start", -1))
        end = int(record.get("location_end", -1))
        pii_type = record.get("piiType", "unknown")

        if start < 0 or end < 0 or start > end:
            continue
        if start >= content_len:
            continue
        # Clamp to non-overlapping stream build
        start = max(start, cursor)
        if start >= end:
            continue
        end = min(end, content_len)

        if cursor < start:
            output_parts.append(content[cursor:start])

        color = get_terminal_color(pii_type)
        output_parts.append(f"{color}{content[start:end]}{Colors.ENDC}")
        cursor = end

    if cursor < content_len:
        output_parts.append(content[cursor:])

    return "".join(output_parts)


def format_pii_records_html(pii_records: List[Dict[str, Any]]) -> str:
    if not pii_records:
        return '<div class="no-pii">No PII detected</div>'

    html_content = '<div class="pii-records"><h4>Detected PII Records:</h4>'
    for record in pii_records:
        pii_type = record.get("piiType", "unknown")
        value = html.escape(record.get("value", ""))
        confidence = record.get("confidenceScore", record.get("score", 0))
        detected_by = record.get("detectedBy", "unknown")
        position = f"{record.get('location_start', 0)}-{record.get('location_end', 0)}"
        reason = record.get("reason", "")
        html_content += f"""
        <div class=\"pii-record\">
            <strong>Type:</strong> {html.escape(str(pii_type))} | 
            <strong>Value:</strong> \"{value}\" | 
            <strong>Position:</strong> {position} | 
            <strong>score:</strong> {float(confidence):.2f} | 
            <strong>Detector:</strong> {html.escape(str(detected_by))}
            <br/><strong>Reason:</strong> {html.escape(str(reason))}
        </div>
        """
    html_content += "</div>"
    return html_content


def format_pii_records_terminal(pii_records: List[Dict[str, Any]]) -> str:
    if not pii_records:
        return f"{Colors.OKCYAN}No PII detected{Colors.ENDC}"

    result = f"\n{Colors.BOLD}Detected PII Records:{Colors.ENDC}\n"
    for i, record in enumerate(pii_records, 1):
        pii_type = record.get("piiType", "unknown")
        value = record.get("value", "")
        confidence = record.get("confidenceScore", record.get("score", 0))
        detected_by = record.get("detectedBy", "unknown")
        position = f"{record.get('location_start', 0)}-{record.get('location_end', 0)}"
        reason = record.get("reason", "")

        color = get_terminal_color(pii_type)
        result += f"  {i}. {Colors.BOLD}Type:{Colors.ENDC} {color}{pii_type}{Colors.ENDC} | "
        result += f"{Colors.BOLD}Value:{Colors.ENDC} \"{value}\" | "
        result += f"{Colors.BOLD}Position:{Colors.ENDC} {position} | "
        result += f"{Colors.BOLD}score:{Colors.ENDC} {float(confidence):.2f} | "
        result += f"{Colors.BOLD}Detector:{Colors.ENDC} {detected_by}\n"
        if reason:
            result += f"     {Colors.BOLD}Reason:{Colors.ENDC} {reason}\n"
    return result


def generate_enhanced_html_report(all_data: Dict[str, List[Dict[str, Any]]], output_file: str) -> None:
    """Generate an enhanced HTML report with navigation and rating functionality."""
    
    # JavaScript for navigation and rating functionality
    javascript_code = """
<script>
let currentHighlightIndex = 0;
let allHighlights = [];
let ratings = {};

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded');
    
    // Wait a bit for all elements to be rendered
    setTimeout(function() {
        allHighlights = Array.from(document.querySelectorAll('.pii-highlight'));
        console.log('Found', allHighlights.length, 'highlights');
        
        // Debug: log all buttons
        const buttons = document.querySelectorAll('.nav-btn, .rating-btn, .export-btn');
        console.log('Found buttons:', buttons.length);
        buttons.forEach((btn, i) => {
            console.log(`Button ${i}:`, btn.textContent, 'visible:', btn.offsetWidth > 0);
        });
        
        if (allHighlights.length > 0) {
            showCurrentHighlight();
        }
        
        updateNavigationButtons();
        updateCounter();
        updateStatusBar();
    }, 100);
});

function showCurrentHighlight() {
    // Remove current highlight class from all elements
    allHighlights.forEach(el => el.classList.remove('current-highlight'));
    
    if (allHighlights.length > 0 && currentHighlightIndex >= 0 && currentHighlightIndex < allHighlights.length) {
        const current = allHighlights[currentHighlightIndex];
        current.classList.add('current-highlight');
        current.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Update rating buttons based on current highlight
        updateRatingButtons();
        updateStatusBar();
    }
}

function goToPrevious() {
    console.log('Previous button clicked, current index:', currentHighlightIndex);
    if (currentHighlightIndex > 0) {
        currentHighlightIndex--;
        showCurrentHighlight();
        updateNavigationButtons();
        updateCounter();
    }
}

function goToNext() {
    console.log('Next button clicked, current index:', currentHighlightIndex);
    if (currentHighlightIndex < allHighlights.length - 1) {
        currentHighlightIndex++;
        showCurrentHighlight();
        updateNavigationButtons();
        updateCounter();
    }
}

function updateNavigationButtons() {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    
    if (prevBtn) prevBtn.disabled = currentHighlightIndex === 0;
    if (nextBtn) nextBtn.disabled = currentHighlightIndex >= allHighlights.length - 1;
}

function updateCounter() {
    const counter = document.getElementById('highlight-counter');
    if (counter && allHighlights.length > 0) {
        counter.textContent = `${currentHighlightIndex + 1} / ${allHighlights.length}`;
    }
}

function rateHighlight(rating) {
    if (allHighlights.length === 0) return;
    
    const current = allHighlights[currentHighlightIndex];
    const sampleId = current.getAttribute('data-sample-id');
    const highlightIdx = current.getAttribute('data-highlight-idx');
    const key = `${sampleId}-${highlightIdx}`;
    const fileId = current.closest('.sample-container').querySelector('.sample-header').textContent.match(/File ID: ([^)]+)/)?.[1] || 'unknown';
    
    // 获取原因文本
    const reasonInput = document.getElementById('reason-input');
    const reason = reasonInput ? reasonInput.value.trim() : '';
    
    // 如果已存在相同的sampleId和fileId，覆盖原来的评分
    const existingKey = Object.keys(ratings).find(k => {
        const r = ratings[k];
        return r.sampleId === sampleId && r.fileId === fileId;
    });
    
    if (existingKey && existingKey !== key) {
        delete ratings[existingKey];
    }
    
    ratings[key] = {
        sampleId: sampleId,
        fileId: fileId,
        rating: rating,
        reason: reason || null,
        highlightIndex: highlightIdx,
        timestamp: new Date().toISOString()
    };
    
    console.log('Rated highlight:', key, 'with rating:', rating, 'reason:', reason || 'none');
    updateRatingButtons();
    updateStatusBar();
    
    // 打分后自动跳转到下一个高亮
    setTimeout(() => {
        if (currentHighlightIndex < allHighlights.length - 1) {
            goToNext();
        }
    }, 300);
}

function quickRate(reason) {
    if (allHighlights.length === 0) return;
    
    // 设置文本框内容
    const reasonInput = document.getElementById('reason-input');
    if (reasonInput) {
        reasonInput.value = reason;
    }
    
    // 自动评分为1
    rateHighlight(1);
}

function updateRatingButtons() {
    if (allHighlights.length === 0) return;
    
    const current = allHighlights[currentHighlightIndex];
    const sampleId = current.getAttribute('data-sample-id');
    const highlightIdx = current.getAttribute('data-highlight-idx');
    const key = `${sampleId}-${highlightIdx}`;
    
    // Clear all rating button selections
    document.querySelectorAll('.rating-btn').forEach(btn => btn.classList.remove('selected'));
    
    // Update reason input
    const reasonInput = document.getElementById('reason-input');
    
    // Highlight current rating if exists and update reason text
    if (ratings[key]) {
        const ratingBtn = document.getElementById(`rating-${ratings[key].rating}`);
        if (ratingBtn) ratingBtn.classList.add('selected');
        
        if (reasonInput) {
            reasonInput.value = ratings[key].reason || '';
        }
    } else {
        // Clear reason input if no rating exists
        if (reasonInput) {
            reasonInput.value = '';
        }
    }
}

function updateStatusBar() {
    const statusBar = document.getElementById('status-bar');
    if (!statusBar) return;
    
    const totalRatings = Object.keys(ratings).length;
    const current = allHighlights[currentHighlightIndex];
    
    if (current) {
        const sampleId = current.getAttribute('data-sample-id');
        const highlightIdx = current.getAttribute('data-highlight-idx');
        const key = `${sampleId}-${highlightIdx}`;
        const isRated = ratings[key] ? true : false;
        const ratingText = isRated ? `已评分: ${ratings[key].rating}分` : '未评分';
        const statusColor = isRated ? '#28a745' : '#dc3545';
        
        statusBar.innerHTML = `
            <span style="color: ${statusColor}; font-weight: bold;">
                ${ratingText}
            </span>
            <span style="margin-left: 15px;">
                总评分数: ${totalRatings}/${allHighlights.length}
            </span>
        `;
    }
}

function exportRatings() {
    if (Object.keys(ratings).length === 0) {
        alert('没有评分数据可导出！');
        return;
    }
    
    const exportData = Object.values(ratings).map(r => ({
        sampleId: r.sampleId,
        fileId: r.fileId,
        rating: r.rating,
        reason: r.reason,
        timestamp: r.timestamp
    }));
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = 'pii_ratings.json';
    link.click();
    
    console.log('Exported ratings:', exportData);
    alert(`已导出 ${exportData.length} 条评分记录到 pii_ratings.json`);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        goToPrevious();
    } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        goToNext();
    } else if (e.key >= '1' && e.key <= '3') {
        e.preventDefault();
        rateHighlight(parseInt(e.key));
    }
});
</script>
"""
    
    html_content = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>PII Visualization Report (Enhanced)</title>
    {HTML_STYLES}
</head>
<body>
    <h1>PII Visualization Report (Enhanced)</h1>
    <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <!-- Controls -->
    <div class="controls-container">
        <div class="nav-buttons">
            <button id="prev-btn" class="nav-btn" onclick="goToPrevious()">← 上一个高亮</button>
            <button id="next-btn" class="nav-btn" onclick="goToNext()">下一个高亮 →</button>
            <span id="highlight-counter" class="highlight-counter">0 / 0</span>
        </div>
        
        <div class="rating-container">
            <span>为当前高亮打分：</span>
            <button id="rating-1" class="rating-btn" onclick="rateHighlight(1)">1分</button>
            <button id="rating-2" class="rating-btn" onclick="rateHighlight(2)">2分</button>
            <button id="rating-3" class="rating-btn" onclick="rateHighlight(3)">3分</button>
            
            <div class="quick-rating-container">
                <button class="quick-rating-btn" onclick="quickRate('incomplete pii data')" title="评分1 - PII数据不完整">1.1</button>
                <button class="quick-rating-btn" onclick="quickRate('pii is test data')" title="评分1 - PII是测试数据">1.2</button>
                <button class="quick-rating-btn" onclick="quickRate('pii contains too much noise')" title="评分1 - PII包含太多噪音">1.3</button>
                <span style="margin-left: 10px; font-size: 12px; color: #666;">
                    1.1=数据不完整 | 1.2=测试数据 | 1.3=噪音太多
                </span>
            </div>
        </div>
        
        <div class="reason-container">
            <label class="reason-label" for="reason-input">评分原因（可选）：</label>
            <textarea id="reason-input" class="reason-input" placeholder="请输入评分原因..."></textarea>
        </div>
        
        <button class="export-btn" onclick="exportRatings()">导出评分结果 (JSON)</button>
        
        <div id="status-bar" class="status-bar">
            加载中...
        </div>
        
        <div style="margin-top: 10px; font-size: 12px; color: #666;">
            快捷键：← → 导航，1-3 打分
        </div>
    </div>
"""

    sample_counter = 0
    for step_name, samples in all_data.items():
        html_content += f"<div class=\"step-header\"><h2>PII Type: {html.escape(str(step_name))}</h2></div>"
        if not samples:
            html_content += '<div class="no-pii">No samples in this group</div>'
            continue

        for i, sample in enumerate(samples, 1):
            sample_counter += 1
            file_id = sample.get("fileId", "unknown")
            path = sample.get("path", "unknown")
            content = sample.get("text", "")
            pii_records = sample.get("piiRecords", [])

            # Use sample counter as unique sample ID
            sample_id = f"sample-{sample_counter}"
            highlighted_content = highlight_pii_html(content, pii_records, sample_id)
            pii_info = format_pii_records_html(pii_records)

            html_content += f"""
            <div class=\"sample-container\" data-sample-id="{sample_id}">
                <div class=\"sample-header\">Sample {i} (File ID: {html.escape(str(file_id))})</div>
                <div><strong>File Path:</strong> {html.escape(str(path))}</div>
                <div><strong>Content Length:</strong> {len(content)} chars</div>
                <div><strong>PII Records:</strong> {len(pii_records)}</div>

                <h4>File Content:</h4>
                <div class=\"content-box\">{highlighted_content}</div>

                {pii_info}
            </div>
            """

    html_content += f"""
    {javascript_code}
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"{Colors.OKGREEN}Enhanced HTML report generated: {output_file}{Colors.ENDC}")


def generate_html_report(all_data: Dict[str, List[Dict[str, Any]]], output_file: str) -> None:
    html_content = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>PII Visualization Report</title>
    {HTML_STYLES}
</head>
<body>
    <h1>PII Visualization Report</h1>
    <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

    for step_name, samples in all_data.items():
        html_content += f"<div class=\"step-header\"><h2>PII Type: {html.escape(str(step_name))}</h2></div>"
        if not samples:
            html_content += '<div class="no-pii">No samples in this group</div>'
            continue

        for i, sample in enumerate(samples, 1):
            file_id = sample.get("fileId", "unknown")
            path = sample.get("path", "unknown")
            content = sample.get("text", "")
            pii_records = sample.get("piiRecords", [])

            highlighted_content = highlight_pii_html(content, pii_records, f"sample-{i}")
            pii_info = format_pii_records_html(pii_records)

            html_content += f"""
            <div class=\"sample-container\">
                <div class=\"sample-header\">Sample {i} (File ID: {html.escape(str(file_id))})</div>
                <div><strong>File Path:</strong> {html.escape(str(path))}</div>
                <div><strong>Content Length:</strong> {len(content)} chars</div>
                <div><strong>PII Records:</strong> {len(pii_records)}</div>

                <h4>File Content:</h4>
                <div class=\"content-box\">{highlighted_content}</div>

                {pii_info}
            </div>
            """

    html_content += """
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"{Colors.OKGREEN}HTML report generated: {output_file}{Colors.ENDC}")


def print_terminal_report(all_data: Dict[str, List[Dict[str, Any]]]) -> None:
    print(f"\n{Colors.BOLD}======= PII Visualization Report ======={Colors.ENDC}")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for step_name, samples in all_data.items():
        print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.BOLD}PII Type: {step_name}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}")

        if not samples:
            print(f"{Colors.WARNING}No samples in this group{Colors.ENDC}\n")
            continue

        for i, sample in enumerate(samples, 1):
            file_id = sample.get("fileId", "unknown")
            path = sample.get("path", "unknown")
            content = sample.get("text", "")
            pii_records = sample.get("piiRecords", [])

            print(f"\n{Colors.OKBLUE}--- Sample {i} (File ID: {file_id}) ---{Colors.ENDC}")
            print(f"{Colors.BOLD}File Path:{Colors.ENDC} {path}")
            print(f"{Colors.BOLD}Content Length:{Colors.ENDC} {len(content)} chars")
            print(f"{Colors.BOLD}PII Records:{Colors.ENDC} {len(pii_records)}")

            print(f"\n{Colors.BOLD}File Content:{Colors.ENDC}")
            print("-" * 80)
            highlighted_content = highlight_pii_terminal(content, pii_records)
            if len(highlighted_content) > 1000:
                print(highlighted_content[:1000] + f"\n{Colors.WARNING}... (truncated) ...{Colors.ENDC}")
            else:
                print(highlighted_content)
            print("-" * 80)

            pii_info = format_pii_records_terminal(pii_records)
            print(pii_info)
            print()


def _normalize_pii_records(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize various possible pii record shapes into a standard list.

    Supported inputs on each sample dict:
    - Already normalized: `piiRecords: List[Dict]`
    - Single record fields: `location_start`, `location_end`, `piiType`, `value`, optional `detectedBy`, `score/confidenceScore`
    - A list under `locations` or `records` with items having the above fields
    """

    if isinstance(sample.get("piiRecords"), list):
        return sample["piiRecords"]

    def _one_record(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if d is None:
            return None
        start = d.get("location_start")
        end = d.get("location_end")
        pii_type = d.get("piiType")
        value = d.get("value")
        if start is None or end is None or pii_type is None or value is None:
            return None
        return {
            "location_start": int(start),
            "location_end": int(end),
            "piiType": str(pii_type),
            "value": str(value),
            "detectedBy": d.get("detectedBy", sample.get("detectedBy", "unknown")),
            "confidenceScore": float(
                d.get("confidenceScore", d.get("score", sample.get("score", 0)))
            ),
            "reason": d.get("reason", sample.get("reason", "")),
        }

    # Case: single-record fields at the top level
    direct = _one_record(sample)
    if direct:
        return [direct]

    # Case: list of records under common keys
    for key in ("locations", "records"):
        if isinstance(sample.get(key), list):
            normalized = []
            for item in sample[key]:
                rec = _one_record(item)
                if rec:
                    normalized.append(rec)
            if normalized:
                return normalized

    return []


def _coerce_sample_for_report(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce an arbitrary sample dict into the shape expected by reporters.

    Expected keys for downstream functions:
    - fileId: display identifier (uses `fileId` | `id` | `_id` | fallback)
    - path: file path if available
    - content: the code/content string to render and highlight
    - piiRecords: list of normalized pii entries from `_normalize_pii_records`
    """

    content = sample.get("text", "")
    file_id = (
        sample.get("fileId")
        or sample.get("id")
        or sample.get("_id")
        or sample.get("uid")
        or "unknown"
    )
    path = sample.get("path") or sample.get("file_path") or sample.get("filepath") or "unknown"
    pii_records = _normalize_pii_records(sample)

    return {
        "fileId": file_id,
        "path": path,
        "text": content,
        "piiRecords": pii_records,
    }


def visualize_dataset(
    dataset_by_piiType_or_list: Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]],
    output_html: Optional[str] = None,
    terminal_only: bool = False,
    html_only: bool = True,
    enhanced: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Visualize dataset grouped by PII type.

    Parameters
    - dataset_by_piiType_or_list: either
      - mapping of piiType -> list of sample dicts; or
      - a flat list of sample dicts (each containing `piiType`).
      Each sample should contain `content` (full code/file text). Position fields `location_start`/`location_end`
      or structures like `piiRecords`/`locations` will be used to highlight.
    - output_html: optional path to write an HTML report. If None and not terminal_only, a timestamped file is created.
    - terminal_only: if True, only print to terminal and skip HTML generation (ignored when html_only=True).
    - enhanced: if True, generate enhanced HTML with navigation and rating functionality.

    Returns the normalized data payload that was rendered, grouped by PII type.
    """

    # Accept both dict-grouped input or flat list input
    grouped_input: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(dataset_by_piiType_or_list, list):
        for item in dataset_by_piiType_or_list:
            pii_type = (item or {}).get("piiType", "unknown")
            grouped_input.setdefault(str(pii_type), []).append(item)
    elif isinstance(dataset_by_piiType_or_list, dict):
        grouped_input = dataset_by_piiType_or_list
    else:
        grouped_input = {}

    # Normalize into the reporter input shape: Dict[group_name, List[sample]]
    all_data: Dict[str, List[Dict[str, Any]]] = {}
    for pii_type, samples in (grouped_input or {}).items():
        if not isinstance(samples, list):
            continue
        group = [_coerce_sample_for_report(s) for s in samples]
        all_data[str(pii_type)] = group

    # Decide outputs based on flags
    if html_only:
        # HTML only
        html_path = output_html or f"pii_type_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        if enhanced:
            generate_enhanced_html_report(all_data, html_path)
        else:
            generate_html_report(all_data, html_path)
    else:
        if terminal_only:
            # Terminal only
            print_terminal_report(all_data)
        else:
            # Both terminal and HTML
            print_terminal_report(all_data)
            html_path = output_html or f"pii_type_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            if enhanced:
                generate_enhanced_html_report(all_data, html_path)
            else:
                generate_html_report(all_data, html_path)

    return all_data


def visualize_dataset_enhanced(
    dataset_by_piiType_or_list: Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]],
    output_html: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function to generate enhanced HTML visualization with navigation and rating features.
    
    This function is equivalent to calling visualize_dataset with enhanced=True and html_only=True.
    """
    return visualize_dataset(
        dataset_by_piiType_or_list,
        output_html=output_html,
        html_only=True,
        enhanced=True
    )


if __name__ == "__main__":
    import json
    import os

    jsonl_path = "/home/hyang45/gitV1/pii_final/save/dataset/refine/refine_bigcode_the-stack-v2-dedup_Java_600000.jsonl"
    content_path = "/home/hyang45/gitV1/pii_final/save/dataset/merge/merge_bigcode_the-stack-v2-dedup_Java_600000.jsonl"

    with open(content_path, "r", encoding="utf-8") as f:
        base_dataset = [json.loads(line) for line in f]

    with open(jsonl_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    # for 0.95%+10% , 800 needs 70 samples 
    # split the dataset by piiType
    dataset_by_piiType = {}
    for record in dataset:
        if record["score"] < 80:
            continue
        pii_type = record["piiType"]
        if pii_type not in dataset_by_piiType:
            dataset_by_piiType[pii_type] = []
        dataset_by_piiType[pii_type].append(record)
    
    used_ids = set()
    dataset_train_all = []
    dataset_val_all = []
    dataset_test_all = []
    # order ["ip_address", "key", "name", "email", "username", "password"]
    for pii_type in ["ip_address", "key", "name", "email", "username", "password"]:
        count = 0
        dataset_select = []
        for record in dataset_by_piiType[pii_type]:
            if record["id"] in used_ids:
                continue
            used_ids.add(record["id"])
            dataset_select.append(record)
            count += 1
            if count >= 1000:
                break
        # random shuffle and split by 0.8 0.1 0.1.
        random.seed(42)
        random.shuffle(dataset_select)
        dataset_train = dataset_select[:int(len(dataset_select) * 0.8)]
        dataset_val = dataset_select[int(len(dataset_select) * 0.8):int(len(dataset_select) * 0.9)]
        dataset_test = dataset_select[int(len(dataset_select) * 0.9):]

        # random select 70 samples from train for visualize
        dataset_train_visualize = []
        dataset_val_visualize = []
        dataset_test_visualize = []
        for record in dataset_train[:70]:
            id = record["id"]
            content = base_dataset[id]["text"]
            record["text"] = content
            dataset_train_visualize.append(record)
        
        for record in dataset_val[:9]:
            id = record["id"]
            content = base_dataset[id]["text"]
            record["text"] = content
            dataset_val_visualize.append(record)

        for record in dataset_test:
            id = record["id"]
            content = base_dataset[id]["text"]
            record["text"] = content
            dataset_test_visualize.append(record)
        
        # print(dataset_train_visualize[0])
        # {'id': 152923, 'value': '36.2.6.12', 'location_start': 18537, 'location_end': 18546, 'piiType': 'ip_address', 'detectedBy': 'regex', 'isHumanReviewed': False, 'score': 85, 'reason': 'The value is a valid IPv4 address and appears in a database connection string, indicating it is likely a public IP address used for accessing a database.', 'content':".."}
       
        html_name = f"train_visualize_{pii_type}.html"
        visualize_dataset(dataset_train_visualize, output_html=html_name)

        html_name = f"val_visualize_{pii_type}.html"
        visualize_dataset(dataset_val_visualize, output_html=html_name)
        
        html_name = f"test_visualize_{pii_type}.html"
        visualize_dataset(dataset_test_visualize, output_html=html_name)

        # save each id to csv file as a row
        # with open(f"used_ids_train.csv", "a") as f:
        #     f.write(f"type: {pii_type} \n")
        #     for record in dataset_train_visualize:
        #         f.write(f"{record['id']} ")
        #     f.write("\n")
        # with open(f"used_ids_val.csv", "a") as f:
        #     f.write(f"type: {pii_type} \n")
        #     for record in dataset_val_visualize:
        #         f.write(f"{record['id']} ")
        #     f.write("\n")
        # with open(f"used_ids_test.csv", "a") as f:
        #     f.write(f"type: {pii_type} \n")
        #     for record in dataset_test_visualize:
        #         f.write(f"{record['id']} ")
        #     f.write("\n")
        with open(f"used_ids_{pii_type}.csv", "a") as f:
            f.write(f"type: {pii_type} train test val \n")
            for record in dataset_train_visualize:
                f.write(f"{record['id']} ")
            f.write("\n")
            for record in dataset_test_visualize:
                f.write(f"{record['id']} ")
            f.write("\n")
            for record in dataset_val_visualize:
                f.write(f"{record['id']} ")
            f.write("\n")

        

        
        
        
        


        
        


    

    
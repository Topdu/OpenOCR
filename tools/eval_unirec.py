import csv
import json
import os
import re
import sys
import time
from bs4 import BeautifulSoup

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tabulate import tabulate  # 需要确保安装tabulate库: pip install tabulate
from rapidfuzz.distance import Levenshtein
import multiprocessing as mp
import multiprocessing
from multiprocessing.pool import ThreadPool, Pool
from functools import partial
import random
import logging
import functools

logger_initialized = {}
import csv
from PIL import Image

from table_recognition_metric import TEDS

teds = TEDS()
teds_struct = TEDS(structure_only=True)


with open('tools/unicode-latex.json', 'r') as f:
    unicode_latex_all = json.load(f)



def replace_callback(match):
    s = match.group(0)
    # print(s)
    # pattern = re.compile(r'\\\((.*?)\\\)')
    # matches = pattern.findall(s)
    # len(matches) >= 1
    if ('\(' in s and '\)' in s and 'textit' not in s) or 'rac{' in s:
        # print(s)
        if '\item' in s:
            s = s.replace('\item', '')
        s = s.replace('~', ' \\sim ')
        s = s.replace('Ġ', '~')
        s = s.replace('Ċ', ' ')
        s = s.replace('textbf', 'mathbf')
        s = re.sub(r"\[\\textcolor\[RGB\](.*?)\]|\[\\citebox(.*?)\]", ' ', s)
        s = re.sub(r"\\textsuperscript\{(.*?)\}", ' ', s)
        # 如果有多个\(\), 删除所有的 \(\)
        count = s.count("\\(")
        if count > 1:
            new_s = s.replace('\\(', '').replace('\\)', '')
        else:
            new_s = s

        
        # 去除头尾的\(\)
        # new_s = new_s.strip('\(').strip('\)')
        if count > 1 and len(new_s) > 0:
            if new_s[0] == '}':
                return '}' + f'\({new_s[1:]}\)'
            elif new_s[-1] == '{':
                return f'\({new_s[:-1]}\)' + '{'
            elif new_s[0] == ')':
                return ')' + f'\({new_s[1:]}\)'
            elif new_s[-1] == '(':
                return f'\({new_s[:-1]}\)' + '('
            else:
                return f'\({new_s}\)'
        else:
            return new_s
    else:
        return s  # 不满足条件，原样返回



# 映射字典（略简化，仅需支持）
unicode_latex = {
    # 下标
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-", "₌": "=", "₍": "(", "₎": ")", "ₐ": "a", "ₑ": "e", "ₕ": "h", "ᵢ": "i", "ⱼ": "j",
    "ₖ": "k", "ₗ": "l", "ₘ": "m", "ₙ": "n", "ₒ": "o", "ₚ": "p", "ᵣ": "r", "ₛ": "s", "ₜ": "t", "ᵤ": "u",
    "ᵥ": "v", "ₓ": "x",
    # 上标
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")", "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e",
    "ᶠ": "f", "ᵍ": "g", "ʰ": "h", "ⁱ": "i", "ʲ": "j", "ᵏ": "k", "ˡ": "l", "ᵐ": "m", "ⁿ": "n", "ᵒ": "o",
    "ᵖ": "p", "ʳ": "r", "ˢ": "s", "ᵗ": "t", "ᵘ": "u", "ᵛ": "v", "ʷ": "w", "ˣ": "x", "ʸ": "y", "ᶻ": "z",
    "ᴬ": "A", "ᴮ": "B", "ᴰ": "D", "ᴱ": "E", "ᴳ": "G", "ᴴ": "H", "ᴵ": "I", "ᴶ": "J", "ᴷ": "K", "ᴸ": "L",
    "ᴹ": "M", "ᴺ": "N", "ᴼ": "O", "ᴾ": "P", "ᴿ": "R", "ᵀ": "T", "ᵁ": "U", "ⱽ": "V", "ᵂ": "W"
}

def exist_subscript_supscript(label):
    # 检查是否存在下标或上标
    for char in label:
        if char in unicode_latex:
            return True
    return False

def process_math(label):
    # 将形如 $内容$ 的部分替换为 \(内容\)
    label = re.sub(r'\$\$(.*?)\$\$', r'\\[\1\\]', label, flags=re.DOTALL)
    return re.sub(r'\$(.*?)\$', r'\\(\1\\)', label, flags=re.DOTALL)
    # label = re.sub(r'\$\$(.*?)\$\$', r'\\[\1\\]', label, flags=re.DOTALL)
# label = re.sub(r'\$(.*?)\$', r'\\(\1\\)', label, flags=re.DOTALL)


# 定义上标/下标字符集
sup_chars = ''.join(re.escape(k) for k in unicode_latex if k in ''.join("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ"))
sub_chars = ''.join(re.escape(k) for k in unicode_latex if k in ''.join("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))

def convert_unicode_to_latex(label: str) -> str:
    result = []
    i = 0
    while i < len(label):
        ch = label[i]
        # 处理连续下标
        if ch in unicode_latex and ch in "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ":
            j = i
            while j < len(label) and label[j] in unicode_latex and label[j] in sub_chars:
                j += 1
            text = ''.join(unicode_latex[c] for c in label[i:j])
            result.append(f"\\(Ċ_{{{text}}}Ċ\\)")
            i = j
        # 处理连续上标
        elif ch in unicode_latex and ch in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ":
            j = i
            while j < len(label) and label[j] in unicode_latex and label[j] in sup_chars:
                j += 1
            text = ''.join(unicode_latex[c] for c in label[i:j])
            result.append(f"\\(Ċ^{{{text}}}Ċ\\)")
            i = j
        else:
            result.append(ch)
            i += 1

    return ''.join(result)

def replace_textcircled(text):
    # Unicode 带圈数字：① (U+2460) = 9312
    circled_nums = {str(i): chr(9311 + i) for i in range(1, 21)}

    # Unicode 带圈小写字母：ⓐ (U+24D0) = 9424
    circled_lower = {chr(97 + i): chr(9424 + i) for i in range(26)}  # a-z

    # Unicode 带圈大写字母：Ⓐ (U+24B6) = 9398
    circled_upper = {chr(65 + i): chr(9398 + i) for i in range(26)}  # A-Z

    # 合并所有映射
    circled_all = {}
    circled_all.update(circled_nums)
    circled_all.update(circled_lower)
    circled_all.update(circled_upper)

    # 匹配 \textcircled{数字或字母}
    def repl(m):
        key = m.group(1)
        return circled_all.get(key, m.group(0))

    return re.sub(r'\\textcircled{([A-Za-z0-9]{1,2})}', repl, text)

def truncate_repeated_tail(s, threshold=20, keep=1):
    """
    如果字符串尾部重复出现某个元素超过threshold次，则只保留keep个该元素
    
    参数:
        s: 输入字符串
        threshold: 重复次数阈值，默认20
        keep: 保留的重复次数，默认5
    
    返回:
        处理后的字符串
    # # 测试
    # test_input = r'<table><tr><td></td><td>\textbf{1}</td><td>\textbf{2}</td><td>\textbf{3}</td><td>\textbf{4}</td><td>\textbf{5}</td><td>\textbf{6}</td><td>\textbf{7}</td><td>\textbf{8}</td><td>\textbf{9}</td><td>\textbf{10}</td><td>\textbf{11}</td><td>\textbf{12}</td></tr><tr><td rowspan="12">\rotatebox[origin=l]{90}{\parbox[c]{16cm}{\  \ cots cots cots cots cots cots cots cots cots cots cots cots cots cots cots cots cots cots cots cots cots cots'
    # result = truncate_repeated_tail(test_input)
    # print("结果:")
    # print(result)
    # print("\n原字符串长度:", len(test_input))
    # print("处理后长度:", len(result))
    """
    if not s:
        return s
    
    # 尝试不同长度的重复模式（从1到合理的最大长度）
    max_pattern_len = min(100, len(s) // threshold)
    
    for pattern_len in range(1, max_pattern_len + 1):
        if len(s) < pattern_len:
            break
            
        # 提取可能的重复模式
        pattern = s[-pattern_len:]
        
        # 从字符串末尾向前计数该模式的重复次数
        count = 0
        pos = len(s)
        
        while pos >= pattern_len:
            if s[pos - pattern_len:pos] == pattern:
                count += 1
                pos -= pattern_len
            else:
                break
        
        # 如果重复次数超过阈值，进行截断
        if count > threshold:
            # 保留前面的非重复部分 + keep个重复模式
            non_repeat_part = s[:pos]
            kept_repeats = pattern * keep
            # print("截断前 ori:", s)
            # print("截断后 after:", non_repeat_part + kept_repeats)
            return non_repeat_part + kept_repeats
    
    # 没有找到超过阈值的重复模式，返回原字符串
    return s



def normalized_formula(text):
    # Normalize math formulas before matching
    filter_list = ['\\mathbf', '\\mathrm', '\\mathnormal', '\\mathit', '\\mathbb', '\\mathcal', '\\mathscr', '\\mathfrak', '\\mathsf', '\\mathtt', 
                   '\\textbf', '\\text', '\\boldmath', '\\boldsymbol', '\\operatorname', '\\bm',
                   '\\symbfit', '\\mathbfcal', '\\symbf', '\\scriptscriptstyle', '\\notag',
                   '\\setlength', '\\coloneqq', '\\space', '\\thickspace', '\\thinspace', '\\medspace', '\\nobreakspace', '\\negmedspace',
                   '\\quad', '\\qquad', '\\enspace', '\\substackw', ' ', '$$', '\\left', '\\right', '\\displaystyle', '\\text']
                #    '\\left', '\\right', '{', '}', ' ']
    
    # delimiter_filter
    text = text.strip().strip('$').strip('\n')
    # pattern = re.compile(r"\\\[(.+?)(?<!\\)\\\]")
    # match = pattern.search(text)

    # if match:
    #     text = match.group(1).strip()
    
    tag_pattern = re.compile(r"\\tag\{.*?\}")
    text = tag_pattern.sub('', text)
    hspace_pattern = re.compile(r"\\hspace\{.*?\}")
    text = hspace_pattern.sub('', text)
    begin_pattern = re.compile(r"\\begin\{.*?\}")
    text = begin_pattern.sub('', text)
    end_pattern = re.compile(r"\\end\{.*?\}")
    text = end_pattern.sub('', text)
    col_sep = re.compile(r"\\arraycolsep.*?\}")
    text = col_sep.sub('', text)
    text = text.strip('.')
    
    for filter_text in filter_list:
        text = text.replace(filter_text, '')
        
    # text = normalize_text(delimiter_filter(text))
    # text = delimiter_filter(text)
    # text = text.lower()
    return text





def norm_latex(label):
    label = replace_textcircled(label)
    label = normalized_formula(label)

    label = process_math(label)
    label = label.replace('\ ', 'Ġ')
    label = label.replace(' ', 'Ġ')
    label = label.replace('\;', 'Ġ')
    # label = label.replace('\ ', 'Ġ')
    label = label.replace('\.', 'Ġ')
    label = label.replace('\,', 'Ġ')
    label = label.replace('\!', 'Ġ')
    label = convert_unicode_to_latex(label)
    for k, v in unicode_latex_all.items():
        label = label.replace(k, '\(Ċ'+v+'Ċ\)')
    pattern = r"[A-Za-z0-9!\"#Ċ$△%&'()*+,-./:;<=>?@[\\\]^_`{|}~{}\\]+"
    # 使用 re.sub 替换匹配到的字符串
    result = re.sub(pattern, replace_callback, label)
    
    result = result.replace('\(~', ' \(')
    result = result.replace('~\)', '\) ')
    # 如果一些特殊符号是单独出现的，就还保留unicode字符
    # for k, v in unicode_latex_all.items():
    #     result = result.replace('\('+v+'\)', k)
    result = result.replace('\(\ldots\ldots\)', '……')
    result = result.replace('\)\(', '')
    result = result.replace('\) \(', '')

    label_sp = result.split(' ')
    if len(label_sp) > 1:
        label_math_new = ''
        for i, sp in enumerate(label_sp):
            # 判断sp是不是latex数学符号\[a-z]
            if len(sp) > 1 and sp[-1:] == '\\':
                label_math_new += sp + ' '
            elif re.match(r'.*\\[A-Za-z]+$', sp):
                if i < len(label_sp)-1 and label_sp[i+1] and label_sp[i+1][0].isalpha():
                    label_math_new += sp + ' '
                else:
                    label_math_new += sp
            else:
                label_math_new += sp
        result = label_math_new

    result = result.replace('Ġ', ' ')
        
    result = result.replace('Ċ', '')
    result = re.sub(r"\\] \(\d+\)\n\n", "", result)
    result = re.sub(r'<\|sn\|>', '', result)
    result = re.sub(r'<\|unk\|>', '', result)
    result = re.sub(r'\uffff', '', result)
    result = re.sub(r'_{4,}', '___', result)
    result = result.replace('\n\n\[', '')
    result = result.replace('\]\n\n', '')
    result = result.replace('\[\n', '')
    result = result.replace('\n\]', '')
    result = result.replace('\]', '')
    result = result.replace('\[', '')
    result = result.replace('\( ', '')
    result = result.replace(' \)', '')
    result = result.replace('\(', '')
    result = result.replace('\)', '')

    # result = remove_redundant_braces(result)
    
    return result

def normalize_latex_formulas(text: str) -> str:
    """
    查找所有 \( ... \) 和 \[ ... \] 的 LaTeX 公式，
    并用 normalized_formula 处理后回填。
    """

    def replacer(match):
        # 匹配组1是括号类型 "(" 或 "["
        # 匹配组2是公式内容
        left, content = match.group(1), match.group(2)
        normalized = norm_latex(f"\\({content}\\)")
        if left == '(':
            return f"\\({normalized}\\)"
        else:
            return f"\\[{normalized}\\]"

    # 匹配 \( ... \) 或 \[ ... \]
    pattern = re.compile(r'\\([\(\[])(.*?)(?:\\[\)\]])', re.DOTALL)
    return pattern.sub(replacer, text)

@functools.lru_cache()
def get_logger(name="DolphinV2", log_file=None, log_level=logging.DEBUG):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    if log_file is not None and rank == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, "a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    logger_initialized[name] = True
    logger.propagate = False
    return logger


logger = get_logger()

rules = [
    (r'([\u4e00-\u9fa5]),', r'\1，'),
    (r'([\u4e00-\u9fa5]):', r'\1：'),
    (r'([\u4e00-\u9fa5]);', r'\1；'),
    (r'([\u4e00-\u9fa5])!', r'\1！'),
    (r'～', r'~'),
    (r'\t', r''),
    (r'([\u4e00-\u9fa5])\.', r'\1。'),
    (r'([\u4e00-\u9fa5])\?', r'\1？'),
    # (r'"([\u4e00-\u9fa5]+)"', r'“\1”'),
    # (r"'([\u4e00-\u9fa5]+)'", r'‘\1’'),
    (r"“", r'"'),
    (r"”", r'"'),
    (r"‘", r"'"),
    (r"’", r"'"),
    (r"（", r"("),
    (r"）", r")"),
    (r"【", r"["),
    (r"】", r"]"),
    # (r'\(([\u4e00-\u9fa5]+)\)', r'（\1）'),
    # (r'\[([\u4e00-\u9fa5]+)\]', r'【\1】'),
    (r'-<\|sn\|>', ''),
    (r'<\|sn\|>', ''),
    (r'<\|unk\|>', ''),
    (r'<s>', ''),
    (r'</s>', ''),
    (r'\uffff', ''),
    (r'_{4,}', '___'),
    (r'\.{4,}', '...'),
    # ..........
]

# text = "ˇe ´a `u ¨u ¯a"
mapping = {
    "´a": "á", "`a": "à", "ˇa": "ǎ", "¯a": "ā", '¨a': 'ä', '˚a': 'å', '˚A': 'Å',
    "´e": "é", "`e": "è", "ˇe": "ě", "¯e": "ē",
    "´i": "í", "`i": "ì", "ˇi": "ǐ", "¯i": "ī",
    "´o": "ó", "`o": "ò", "ˇo": "ǒ", "¯o": "ō", '¨o': 'ö',
    "´u": "ú", "`u": "ù", "ˇu": "ǔ", "¯u": "ū",
    "¨u": "ü", "˜n": "ñ"
}

def remove_space_before_sn(text):
    # 匹配 “汉字 + 空格 + <|sn|>” 这种模式
    # \u4e00-\u9fff 是中文字符的 Unicode 范围
    return re.sub(r'([\u4e00-\u9fff])\s*<\|sn\|>', r'\1', text)

def normalize_text(text, rm_space=False):
    text = truncate_repeated_tail(text)
    text = replace_textcircled(text)
    text = process_math(text)
    text = normalize_latex_formulas(text)
    text = remove_space_before_sn(text)
    text = re.sub(r'(\d) +', r'\1', text)
    text = re.sub(r' +(\d)', r'\1', text)
    text = text.strip().replace('\n', '').replace('\r', '').replace('\\t', '').replace('\\n', '').replace('/t', '').replace('/n', '')
    # text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
    for k, v in mapping.items():
        text = text.replace(k, v)

    result = re.sub(r"\\] \(\d+\)\n\n", "", text)
    result = result.replace('\n\n\[', '')
    result = result.replace('\]\n\n', '')
    result = result.replace('\[\n', '')
    result = result.replace('\n\]', '')
    result = result.replace('\]', '')
    result = result.replace('\[', '')
    result = result.replace('\( ', '')
    result = result.replace(' \)', '')
    result = result.replace('\(', '')
    text = result.replace('\)', '')
    if re.match(r'^[\^_]\{.*\}$', text):
        text = text[2:-1]
    # 将所有空格替换为一个空格
    text = ' '.join(text.split())
    # 将中文字符后的英文符号转为中文符号，例如：“你好,世界!” -> “你好，世界！”
    for pattern, repl in rules:
        text = re.sub(pattern, repl, text)
    
    if rm_space:
        text = text.replace(' ', '')
    return text


def normalize_text_and_edit_distance(label_text, pred_text, rm_space=False):
    # print(f'label_text_before_normalize: {label_text}, pred_text: {pred_text}')
    label_text = normalize_text(label_text, rm_space=rm_space)
    pred_text = normalize_text(pred_text, rm_space=rm_space)

    # print(f'label_text_after_normalize: {label_text}, pred_text: {pred_text}')
    return Levenshtein.normalized_distance(label_text, pred_text)

def clean_html_table_styles(html_content):
    """移除 HTML 表格及内部标签中的 style、border 等属性"""
    soup = BeautifulSoup(html_content, "html.parser")

    # 移除所有 style 属性、border、width、align 等表格相关样式
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr in ["style", "border", "width", "align", "cellpadding", "cellspacing"]:
                del tag.attrs[attr]

    return str(soup)

def norm_table(table_content):
    table_content = process_math(table_content)
    table_content = normalize_latex_formulas(table_content)
    

    table_content = table_content.replace('<tdcolspan=', '<td colspan=')
    table_content = table_content.replace('<tdrowspan=', '<td rowspan=')
    table_content = table_content.replace('"colspan=', '" colspan=')
    table_content = re.sub(r'<\|sn\|>', '', table_content)
    table_content = re.sub(r'<\|unk\|>', '', table_content)
    table_content = re.sub(r'\uffff', '', table_content)
    table_content = re.sub(r'_{4,}', '___', table_content)
    table_content = re.sub(r'\.{4,}', '...', table_content)

    table_content = re.sub(r'</td\s+colspan="[^"]*"\s*>', '</td>', table_content, flags=re.IGNORECASE)
    table_content = re.sub(r'</td\s+rowspan="[^"]*"\s*>', '</td>', table_content, flags=re.IGNORECASE)
    table_content = re.sub(r'</th\s+rowspan="[^"]*"\s*>', '</th>', table_content, flags=re.IGNORECASE)
    table_content = re.sub(r'</th\s+colspan="[^"]*"\s*>', '</th>', table_content, flags=re.IGNORECASE)
    
    
    table_content = clean_html_table_styles(table_content)

    table_content = convert_unicode_to_latex(table_content)
    for k, v in unicode_latex_all.items():
        table_content = table_content.replace(k, '\('+v+'\)')
    table_content = table_content.replace('Ċ', '')
    # 转换 sup
    table_content = re.sub(r'<sup>(.*?)<\/sup>', r'\\(^{\1}\\)', table_content)
    # 转换 sub
    table_content = re.sub(r'<sub>(.*?)<\/sub>', r'\\(_{\1}\\)', table_content)

    table_content =  re.sub(r'\$(.*?)\$', r'\\(\1\\)', table_content, flags=re.DOTALL)
    # table_content =  re.sub(r'>', r'', table_content, flags=re.DOTALL)
    table_content = re.sub(r'>\s*\n+\s*', '>', table_content, flags=re.DOTALL)

    table_content = table_content.replace('\n', '').strip()
    table_content = table_content.replace('，', ',')
    table_content = table_content.replace('（', '(')
    table_content = table_content.replace('）', ')')
    table_content = table_content.replace('：', ':')
    table_content = table_content.replace('～', '~')
    table_content = table_content.replace('；', ';')
    table_content = table_content.replace('！', '!')
    table_content = table_content.replace('。', '.')
    table_content = table_content.replace('？', '?')
    table_content = table_content.replace('—', '-')
    table_content = table_content.replace('’', "'")
    table_content = table_content.replace('′', "'")
    table_content = table_content.replace('–', '-')
    table_content = table_content.replace('−', '-')
    # table_content = normalize_text(table_content)

    table_content = table_content.replace('<thead>', '')
    table_content = table_content.replace('</thead>', '')
    table_content = table_content.replace('<th', '<td')
    table_content = table_content.replace('</th>', '</td>')
    table_content = table_content.replace('<tbody>', '')
    table_content = table_content.replace('</tbody>', '')
    table_content = table_content.replace('<br>', '')
    table_content = table_content.replace('<br/>', '')
    table_content = table_content.replace('<b>', '')
    table_content = table_content.replace('</b>', '')
    table_content = table_content.replace('<strong>', '')
    table_content = table_content.replace('</strong>', '')
    table_content = table_content.replace('<i>', '')
    table_content = table_content.replace('</i>', '')
    table_content = replace_textcircled(table_content)
    return table_content



ignore_text_type = ['abandon', 'text_mask', 'table_mask', 'need_mask']


def compute_hierarchical_metrics(label_json_data, exclude_text_type=['table']):
    hierarchical_metrics = {}

    for item in label_json_data:
        if item['category_type'] in ignore_text_type:
            continue

        if item.get('text_type', 'unknown') != 'text':
            continue
        if 'data_source' not in item:
            continue
        if 'language' not in item:
            continue
        if 'text_level' not in item:
            continue
        
        data_source = item.get('data_source', 'unknown')
        language = item.get('language', 'unknown')
        text_level = item.get('text_level', 'unknown')
        edit_distance = item['pred_info']['edit_distance']

        if data_source not in hierarchical_metrics:
            hierarchical_metrics[data_source] = {}
        if language not in hierarchical_metrics[data_source]:
            hierarchical_metrics[data_source][language] = {}
        if text_level not in hierarchical_metrics[data_source][language]:
            hierarchical_metrics[data_source][language][text_level] = {'total': 0, 'edit_distance': 0.0}

        hierarchical_metrics[data_source][language][text_level]['total'] += 1
        hierarchical_metrics[data_source][language][text_level]['edit_distance'] += edit_distance

    # 计算平均精度
    for ds, lang_dict in hierarchical_metrics.items():
        for lang, level_dict in lang_dict.items():
            for lvl, metrics in level_dict.items():
                total = metrics['total']
                avg_edit_distance = metrics['edit_distance'] / total if total > 0 else 1.0
                metrics['avg_edit_distance'] = avg_edit_distance
                # metrics['accuracy'] = 1 - avg_edit_distance

    return hierarchical_metrics


def _compute_metric_single(args):
    """单个样本的指标计算（用于多线程并行）"""
    idx, label_info, pred_info, exclude_text_type, is_norm, rm_space = args
    result = {}
    
    if label_info['text_type'] in exclude_text_type:
        return idx, result, False  # skip
    
    if label_info['text_type'] == 'table':
        gt_table = "<html><body>" + norm_table(label_info['html']).strip() + "</body></html>"
        pred_table = "<html><body>" + norm_table(pred_info['text']).strip() + "</body></html>"
        result['TEDS'] = teds(pred_table, gt_table)
        result['TEDS-S'] = teds_struct(pred_table, gt_table)
    elif label_info['text_type'] == 'equation':
        result['edit_distance'] = Levenshtein.normalized_distance(norm_latex(label_info['latex']), norm_latex(pred_info['text']))
    else:
        if 'text' not in label_info or 'text' not in pred_info:
            print(f'未找到预测结果的图片: {label_info["image_path"]}, label_info: {label_info}, pred_info: {pred_info}')
        if is_norm:
            result['edit_distance'] = normalize_text_and_edit_distance(label_info['text'], pred_info['text'], rm_space)
        else:
            result['edit_distance'] = Levenshtein.normalized_distance(label_info['text'], pred_info['text'])
    
    return idx, result, True


def unirec_measure(pred_json_data, label_json_data, exclude_text_type=['table'], is_norm=True, rm_space=False):
    total_num = len(label_json_data)
    label_json_data_new = []
    for label_info in label_json_data:
        if ('text' not in label_info and 'latex' not in label_info and 'html' not in label_info )or label_info['category_type'] in ignore_text_type:
            continue
        if 'text' in label_info and label_info['text'].replace('.', '').replace(' ', '').lower() == 'nodate':
            continue
        # if label_info['data_source'] == 'exam_paper' and label_info['category_type'] == 'header':
        if label_info['category_type'] == 'header' or label_info['data_source'] == 'footer':
            continue
        label_json_data_new.append(label_info)

    label_json_data = label_json_data_new
    
    # 构建多线程任务列表
    tasks = []
    task_indices = []  # 记录有预测结果的样本索引
    for idx, label_info in enumerate(label_json_data):
        image_file_name = label_info['image_path']
        if image_file_name in pred_json_data:
            pred_info = pred_json_data[image_file_name]
            label_info['pred_info'] = pred_info
            tasks.append((idx, label_info, pred_info, exclude_text_type, is_norm, rm_space))
            task_indices.append(idx)
        else:
            if label_info.get('text_type') == 'table':
                label_info['pred_info'] = {'TEDS': 0.0, 'TEDS-S': 0.0}
            else:
                label_info['pred_info'] = {'edit_distance': 1.0}
    
    # 使用多线程并行计算指标
    num_workers = min(32, len(tasks))
    if num_workers > 0:
        logger.info(f"使用 {num_workers} 个线程并行计算指标，共 {len(tasks)} 个样本...")
        with Pool(num_workers) as pool:
            results = pool.map(_compute_metric_single, tasks)
        
        # 回填结果
        for idx, metric_result, valid in results:
            if valid and metric_result:
                label_json_data[idx]['pred_info'].update(metric_result)
        logger.info(f"指标计算完成")
    
    # 按照text_type类别统计指标
    category_type_metrics = {}
    for item in label_json_data:
        if item['category_type'] in ignore_text_type:
            continue
        category_type = item.get('text_type', 'unknown')
        # 其中table类别，需要根据table_language进行统计，equation类别需要按照equation_line_num进行统计
        if category_type == 'table':
            table_language = item.get('table_language', 'unknown')
            # 统计每一种类别的table_language数量和TEDS/TEDS-S
            if category_type in category_type_metrics:
                category_type_metrics[category_type]['total'] += 1
                category_type_metrics[category_type]['TEDS'] += item['pred_info']['TEDS']
                category_type_metrics[category_type]['TEDS-S'] += item['pred_info']['TEDS-S']
                if table_language not in category_type_metrics[category_type]:
                    category_type_metrics[category_type][table_language] = {'total': 1, 'TEDS': item['pred_info']['TEDS'], 'TEDS-S': item['pred_info']['TEDS-S']}
                else:
                    category_type_metrics[category_type][table_language]['total'] += 1
                    category_type_metrics[category_type][table_language]['TEDS'] += item['pred_info']['TEDS']
                    category_type_metrics[category_type][table_language]['TEDS-S'] += item['pred_info']['TEDS-S']
            else:
                category_type_metrics[category_type] = {'total': 1, 'TEDS': item['pred_info']['TEDS'], 'TEDS-S': item['pred_info']['TEDS-S'], table_language: {'total': 1, 'TEDS': item['pred_info']['TEDS'], 'TEDS-S': item['pred_info']['TEDS-S']}}
        elif category_type == 'equation':
            equation_line_num = item.get('equation_line_num', 1)
            # 统计每一种类别的equation_line_num数量和edit_distance
            if category_type in category_type_metrics:
                category_type_metrics[category_type]['total'] += 1
                if equation_line_num not in category_type_metrics[category_type]:
                    category_type_metrics[category_type][equation_line_num] = {'total': 1, 'edit_distance': item['pred_info']['edit_distance']}
                else:
                    category_type_metrics[category_type][equation_line_num]['total'] += 1
                    category_type_metrics[category_type][equation_line_num]['edit_distance'] += item['pred_info']['edit_distance']
            else:
                category_type_metrics[category_type] = {'total': 1, equation_line_num: {'total': 1, 'edit_distance': item['pred_info']['edit_distance']}}
        else:
            # 统计每一种类别的数量和edit_distance
            if category_type in category_type_metrics:
                category_type_metrics[category_type]['total'] += 1
                category_type_metrics[category_type]['edit_distance'] += item['pred_info']['edit_distance']
            else:
                category_type_metrics[category_type] = {'total': 1, 'edit_distance': item['pred_info']['edit_distance']}

    # 按照language类别统计指标（跳过table类型，table使用TEDS/TEDS-S单独统计）
    language_type_metrics = {}
    for item in label_json_data:
        if item['category_type'] in ignore_text_type:
            continue
        text_type = item.get('text_type', 'unknown')
        if text_type in exclude_text_type:
            continue
        if text_type == 'table':
            continue
        language_type = item.get('language', 'unknown')
        if language_type != 'unknown':
            if language_type in language_type_metrics:
                language_type_metrics[language_type]['total'] += 1
                language_type_metrics[language_type]['edit_distance'] += item['pred_info']['edit_distance']
            else:
                language_type_metrics[language_type] = {'total': 1, 'edit_distance': item['pred_info']['edit_distance']}
    

    # 按照text_level类别统计指标, 其中paragraph类别，需要根据line_num进行统计，multi_paragraph类别需要按照paragraph_num进行统计
    # 跳过table类型，table使用TEDS/TEDS-S单独统计
    text_level_type_metrics = {}
    for item in label_json_data:
        if item['category_type'] in ignore_text_type:
            continue
        text_type = item.get('text_type', 'unknown')
        if text_type in exclude_text_type:
            continue
        if text_type == 'table':
            continue
        text_level_type = item.get('text_level', 'unknown')
        if text_level_type != 'unknown':
            if text_level_type == 'paragraph':
                line_num = item.get('line_num', 1)
                if text_level_type in text_level_type_metrics:
                    text_level_type_metrics[text_level_type]['total'] += 1
                    if line_num not in text_level_type_metrics[text_level_type]:
                        text_level_type_metrics[text_level_type][line_num] = {'total': 1, 'edit_distance': item['pred_info']['edit_distance']}
                    else:
                        text_level_type_metrics[text_level_type][line_num]['total'] += 1
                        text_level_type_metrics[text_level_type][line_num]['edit_distance'] += item['pred_info']['edit_distance']
                else:
                    text_level_type_metrics[text_level_type] = {'total': 1, line_num: {'total': 1, 'edit_distance': item['pred_info']['edit_distance']}}
            elif text_level_type == 'multi_paragraph':
                paragraph_num = item.get('paragraph_num', 1)
                if text_level_type in text_level_type_metrics:
                    text_level_type_metrics[text_level_type]['total'] += 1
                    if paragraph_num not in text_level_type_metrics[text_level_type]:
                        text_level_type_metrics[text_level_type][paragraph_num] = {'total': 1, 'edit_distance': item['pred_info']['edit_distance']}
                    else:
                        text_level_type_metrics[text_level_type][paragraph_num]['total'] += 1
                        text_level_type_metrics[text_level_type][paragraph_num]['edit_distance'] += item['pred_info']['edit_distance']
                else:
                    text_level_type_metrics[text_level_type] = {'total': 1, paragraph_num: {'total': 1, 'edit_distance': item['pred_info']['edit_distance']}}
            else:
                if text_level_type in text_level_type_metrics:
                    text_level_type_metrics[text_level_type]['total'] += 1
                    text_level_type_metrics[text_level_type]['edit_distance'] += item['pred_info']['edit_distance']
                else:
                    text_level_type_metrics[text_level_type] = {'total': 1, 'edit_distance': item['pred_info']['edit_distance']}
    
    # 按照data_source类别统计指标（跳过table类型，table使用TEDS/TEDS-S单独统计）
    data_source_type_metrics = {}
    for item in label_json_data:
        if item['category_type'] in ignore_text_type:
            continue
        data_source_type = item.get('data_source', 'unknown')
        if item.get('text_type', 'unknown') in exclude_text_type:
            continue
        if item.get('text_type', 'unknown') == 'table':
            continue
        if data_source_type != 'unknown':
            if data_source_type in data_source_type_metrics:
                data_source_type_metrics[data_source_type]['total'] += 1
                data_source_type_metrics[data_source_type]['edit_distance'] += item['pred_info']['edit_distance']
            else:
                data_source_type_metrics[data_source_type] = {'total': 1, 'edit_distance': item['pred_info']['edit_distance']}
    
    # 将四种情况的指标进行合并
    all_metrics = {
        'language': language_type_metrics,
        'text_level': text_level_type_metrics,
        'category': category_type_metrics,
        'data_source': data_source_type_metrics,
    }
    hierarchical_metrics = compute_hierarchical_metrics(label_json_data=label_json_data, exclude_text_type=exclude_text_type)
    return all_metrics, hierarchical_metrics



def calculate_average_metrics(metrics):
    """计算各类别的平均编辑距离（表格类别使用TEDS/TEDS-S）"""
    avg_metrics = {}
    
    for metric_type, type_data in metrics.items():
        avg_metrics[metric_type] = {}
        
        # 处理有子分类的情况（如table、equation等）
        for category, data in type_data.items():
            # 计算该类别的总平均
            total = data['total']
            
            # 判断是否是表格类别（使用TEDS/TEDS-S指标）
            is_table = 'TEDS' in data
            
            if is_table:
                total_teds = 0
                total_teds_s = 0
            else:
                total_edit_distance = 0
            
            # 检查是否有子分类
            has_subcategories = False
            subcategories = {}
            
            for key, value in data.items():
                if isinstance(value, dict):  # 子分类项
                    has_subcategories = True
                    sub_total = value['total']
                    if is_table:
                        sub_teds = value['TEDS']
                        sub_teds_s = value['TEDS-S']
                        sub_avg_teds = sub_teds / sub_total if sub_total > 0 else 0
                        sub_avg_teds_s = sub_teds_s / sub_total if sub_total > 0 else 0
                        subcategories[key] = {
                            'total': sub_total,
                            'TEDS': sub_teds,
                            'avg_TEDS': sub_avg_teds,
                            'TEDS-S': sub_teds_s,
                            'avg_TEDS-S': sub_avg_teds_s,
                        }
                        total_teds += sub_teds
                        total_teds_s += sub_teds_s
                    else:
                        sub_edit = value['edit_distance']
                        sub_avg = sub_edit / sub_total if sub_total > 0 else 0
                        subcategories[key] = {
                            'total': sub_total,
                            'edit_distance': sub_edit,
                            'avg_edit_distance': sub_avg
                        }
                        total_edit_distance += sub_edit
            
            # 如果没有子分类，直接计算平均值
            if not has_subcategories:
                if is_table:
                    total_teds = data['TEDS']
                    total_teds_s = data['TEDS-S']
                    avg_teds = total_teds / total if total > 0 else 0
                    avg_teds_s = total_teds_s / total if total > 0 else 0
                    avg_metrics[metric_type][category] = {
                        'total': total,
                        'TEDS': total_teds,
                        'avg_TEDS': avg_teds,
                        'TEDS-S': total_teds_s,
                        'avg_TEDS-S': avg_teds_s,
                    }
                else:
                    total_edit_distance = data['edit_distance']
                    avg = total_edit_distance / total if total > 0 else 0
                    avg_metrics[metric_type][category] = {
                        'total': total,
                        'edit_distance': total_edit_distance,
                        'avg_edit_distance': avg
                    }
            else:
                # 有子分类的情况
                if is_table:
                    avg_teds = total_teds / total if total > 0 else 0
                    avg_teds_s = total_teds_s / total if total > 0 else 0
                    avg_metrics[metric_type][category] = {
                        'total': total,
                        'TEDS': total_teds,
                        'avg_TEDS': avg_teds,
                        'TEDS-S': total_teds_s,
                        'avg_TEDS-S': avg_teds_s,
                        'subcategories': subcategories
                    }
                else:
                    avg = total_edit_distance / total if total > 0 else 0
                    avg_metrics[metric_type][category] = {
                        'total': total,
                        'edit_distance': total_edit_distance,
                        'avg_edit_distance': avg,
                        'subcategories': subcategories
                    }
    
    return avg_metrics

all_metrics_key = {
    'category': ['text', 'equation', 'equation_mixed', 'table'],
    'text_level': ['char', 'word', 'line', 'paragraph', 'multi_paragraph'],
    'language': ['zh', 'en', 'mixed'],
    'data_source': ["book", "PPT2PDF", "research_report", "colorful_textbook", "exam_paper", "magazine", "academic_literature", "note", "newspaper"],
    }

def format_metrics_as_tables(avg_metrics):
    """将平均指标格式化为表格"""
    tables = {}
    csv_rows = [['Category'], ['Total Samples'], ['Avg Edit Distance'], ['Avg TEDS'], ['Avg TEDS-S']]
    sub_csv_rows = []
    # for metric_type, categories in avg_metrics.items():
    for metric_type in all_metrics_key.keys():
        categories = avg_metrics[metric_type]
        tables[metric_type] = []
        
        # 主表格标题
        tables[metric_type].append(f"\n===== {metric_type.capitalize()} Metrics =====")
        
        # 主类别表格
        main_table_data = []
        main_headers = ["Category"] + all_metrics_key[metric_type]
        data_sample_list = ["Total Samples"]
        avg_edit_distance_list = ["Avg Edit Distance"]
        avg_teds_list = ["Avg TEDS"]
        avg_teds_s_list = ["Avg TEDS-S"]
        # for category, data in categories.items():
        for key_c in all_metrics_key[metric_type]:
            if key_c in categories:
                data_sample_list.append(categories[key_c]['total'])
                is_table = 'avg_TEDS' in categories[key_c]
                if is_table:
                    avg_edit_distance_list.append("N/A")
                    avg_teds_list.append(f"{categories[key_c]['avg_TEDS']:.4f}")
                    avg_teds_s_list.append(f"{categories[key_c]['avg_TEDS-S']:.4f}")
                else:
                    avg_edit_distance_list.append(f"{categories[key_c]['avg_edit_distance']:.4f}")
                    avg_teds_list.append("N/A")
                    avg_teds_s_list.append("N/A")
            else:
                data_sample_list.append(0)
                avg_edit_distance_list.append("N/A")
                avg_teds_list.append("N/A")
                avg_teds_s_list.append("N/A")
        main_table_data.append(data_sample_list)
        main_table_data.append(avg_edit_distance_list)
        main_table_data.append(avg_teds_list)
        main_table_data.append(avg_teds_s_list)

        # 主表格csv行
        csv_rows[0].extend(main_headers[1:])
        csv_rows[1].extend(data_sample_list[1:])
        csv_rows[2].extend(avg_edit_distance_list[1:])
        csv_rows[3].extend(avg_teds_list[1:])
        csv_rows[4].extend(avg_teds_s_list[1:])

        # 添加主表格
        tables[metric_type].append(tabulate(main_table_data, headers=main_headers, tablefmt="grid"))
        
        # 处理子分类表格
        for category, data in categories.items():
            if 'subcategories' in data:
                tables[metric_type].append(f"\n--- {category} Subcategories ---")
                sub_csv_rows.append([f"{category} Subcategories"])
                sub_table_data = []
                is_table = 'avg_TEDS' in data
                if category == 'table':
                    sub_headers = ["Subcategory"] + list(data['subcategories'].keys())
                else:
                    subcategories_head = list(data['subcategories'].keys())
                    # 排序，并返回索引
                    sort_index = sorted(range(len(subcategories_head)), key=lambda k: subcategories_head[k])
                    sub_headers = ["Subcategory"] + [subcategories_head[i] for i in sort_index]
                sub_data_sample_list = ["Total Samples"]
                if is_table:
                    sub_avg_teds_list = ["Avg TEDS"]
                    sub_avg_teds_s_list = ["Avg TEDS-S"]
                    for subcat, subdata in data['subcategories'].items():
                        sub_data_sample_list.append(subdata['total'])
                        sub_avg_teds_list.append(f"{subdata['avg_TEDS']:.4f}")
                        sub_avg_teds_s_list.append(f"{subdata['avg_TEDS-S']:.4f}")
                    if category == 'table':
                        sub_table_data.append(sub_data_sample_list)
                        sub_table_data.append(sub_avg_teds_list)
                        sub_table_data.append(sub_avg_teds_s_list)
                    sub_csv_rows.append(sub_headers)
                    sub_csv_rows.append(sub_data_sample_list)
                    sub_csv_rows.append(sub_avg_teds_list)
                    sub_csv_rows.append(sub_avg_teds_s_list)
                else:
                    sub_avg_edit_distance_list = ["Avg Edit Distance"]
                    for subcat, subdata in data['subcategories'].items():
                        sub_data_sample_list.append(subdata['total'])
                        sub_avg_edit_distance_list.append(f"{subdata['avg_edit_distance']:.4f}")
                    if category != 'table':
                        # 排序子分类数据
                        sub_data_sample_list = ["Total Samples"] + [sub_data_sample_list[i+1] for i in sort_index]
                        sub_avg_edit_distance_list = ["Avg Edit Distance"] + [sub_avg_edit_distance_list[i+1] for i in sort_index]
                        sub_table_data.append(sub_data_sample_list)
                        sub_table_data.append(sub_avg_edit_distance_list)
                    sub_csv_rows.append(sub_headers)
                    sub_csv_rows.append(sub_data_sample_list)
                    sub_csv_rows.append(sub_avg_edit_distance_list)
                
                tables[metric_type].append(tabulate(sub_table_data, headers=sub_headers, tablefmt="grid"))
    
    return tables, csv_rows+sub_csv_rows

def save_metrics(avg_metrics, hierarchical_metrics, tables, csv_rows, save_path, ignore_eq_table=False, is_norm=True, rm_space=False):
    """保存指标结果为JSON和文本表格"""
    # 保存原始指标数据（JSON格式）
    with open(os.path.join(save_path, f'metrics_details{("_norm" if is_norm else "")}{("_rm_space" if rm_space else "")}{("_ignore_eq_table" if ignore_eq_table else "")}.json'), 'w', encoding='utf-8') as f:
        json.dump(avg_metrics, f, ensure_ascii=False, indent=2)

    # 保存表格（txt格式）
    with open(os.path.join(save_path, f'metrics_tables{("_norm" if is_norm else "")}{("_rm_space" if rm_space else "")}{("_ignore_eq_table" if ignore_eq_table else "")}.txt'), 'w', encoding='utf-8') as f:
        for metric_type, table in tables.items():
            f.write(f"\n\n{metric_type.upper()} METRICS\n")
            for line in table:
                f.write(line + '\n')
    
    # 保存表格（csv格式）
    with open(os.path.join(save_path, f'metrics_tables{("_norm" if is_norm else "")}{("_rm_space" if rm_space else "")}{("_ignore_eq_table" if ignore_eq_table else "")}.csv'), 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    final_row = []
    samples_num = []
    head_rows = [[],[],[]]
    for data_source in all_metrics_key['data_source']:
        for langa in all_metrics_key['language']:
            for text_level in all_metrics_key['text_level']:
                head_rows[0].append(data_source)
                head_rows[1].append(langa)
                head_rows[2].append(text_level)
                try:
                    samples_num.append(hierarchical_metrics[data_source][langa][text_level]['total'])
                    final_row.append(hierarchical_metrics[data_source][langa][text_level]['avg_edit_distance'])
                except KeyError:
                    samples_num.append(0)
                    final_row.append('N/A')
    # 保存表格（csv格式）
    with open(os.path.join(save_path, f'hierarchical_metrics_tables{("_norm" if is_norm else "")}{("_rm_space" if rm_space else "")}{("_ignore_eq_table" if ignore_eq_table else "")}.csv'), 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        step_ma = len(final_row)//len(all_metrics_key['data_source'])
        for i in range(0, len(final_row), step_ma):
            writer.writerow(head_rows[0][i:i+step_ma])
            writer.writerow(head_rows[1][i:i+step_ma])
            writer.writerow(head_rows[2][i:i+step_ma])
            writer.writerow(samples_num[i:i+step_ma])
            writer.writerow(final_row[i:i+step_ma])


def process_batch(build_model, gpu_id, model_args, process_element_func, batch_files, img_path, save_res_path, batch_size=1):
    """在单个GPU上处理一批文件"""
    results = {}
    t_sum = 0.0
    # 用于收集 acceptance 统计数据
    total_acceptance_length = 0.0
    total_full_accept_rate = 0.0
    acceptance_count = 0

    def _execute_task(task_args):
        return process_element_func(**task_args)
    try:
        # gpu_id, model_args
        if model_args is not None:
            model = build_model(gpu_id, model_args)
        else:
            model = build_model(gpu_id)
        for idx, file_info in enumerate(batch_files[::batch_size]):
            if batch_size >1:
                file_path_list = []
                element_type_list = []
                for i in range(batch_size):
                    cur_index = (idx*batch_size+i) % len(batch_files)
                    cur_file_info = batch_files[cur_index]
                    file_path_list.append(os.path.join(img_path, cur_file_info['image_path']))
                    if cur_file_info['text_type'] == 'table':
                        element_type_list.append('table')
                    elif cur_file_info['text_type'] == 'equation':
                        element_type_list.append('formula')
                    else:
                        element_type_list.append('text')
                file_path = file_path_list
                element_type = element_type_list
            else:
                file_path = os.path.join(img_path, file_info['image_path'])
                if file_info['text_type'] == 'table':
                    element_type = 'table'
                elif file_info['text_type'] == 'equation':
                    element_type = 'formula'
                else:
                    element_type = 'text'
            try:
                if isinstance(file_path, list):
                    tasks = []
                    for i in range(batch_size):
                        tasks.append({
                            "file_path": file_path[i],
                            "chat_model": model,
                            "element_type": element_type[i],
                        })
                    t_start = time.time()
                    results_bs = []
                    with ThreadPool(batch_size) as pool:
                        for result in pool.imap_unordered(_execute_task, tasks):
                            results_bs.append(result)
                    t_cost = time.time() - t_start
                    t_sum += t_cost
                    t_cost_avg = t_cost / batch_size
                    # print(f'process batch {idx}/{len(file_list)//batch_size} done')
                    for i in range(batch_size):
                        # 兼容 process_element 返回 dict 或 str
                        if isinstance(results_bs[i], dict):
                            rec_text_bs = results_bs[i]['text']
                            result_entry = {'text': rec_text_bs, 'elapse': t_cost_avg}
                            if 'avg_acceptance_length' in results_bs[i]:
                                result_entry['avg_acceptance_length'] = results_bs[i]['avg_acceptance_length']
                                result_entry['full_accept_rate'] = results_bs[i]['full_accept_rate']
                                total_acceptance_length += results_bs[i]['avg_acceptance_length']
                                total_full_accept_rate += results_bs[i]['full_accept_rate']
                                acceptance_count += 1
                            results[os.path.basename(file_path[i])] = result_entry
                        else:
                            results[os.path.basename(file_path[i])] = {'text': results_bs[i], 'elapse': t_cost_avg}
                else:
                    t_start = time.time()
                    preds_result = process_element_func(file_path, model, element_type, file_info['data_source'])
                    t_cost = time.time() - t_start

                    # 兼容 process_element 返回 dict 或 str
                    if isinstance(preds_result, dict):
                        rec_text = preds_result['text']
                        result_entry = {
                            'text': rec_text,
                            'elapse': t_cost,
                        }
                        if 'avg_acceptance_length' in preds_result:
                            result_entry['avg_acceptance_length'] = preds_result['avg_acceptance_length']
                            result_entry['full_accept_rate'] = preds_result['full_accept_rate']
                            total_acceptance_length += preds_result['avg_acceptance_length']
                            total_full_accept_rate += preds_result['full_accept_rate']
                            acceptance_count += 1
                    else:
                        rec_text = preds_result
                        result_entry = {
                            'text': rec_text,
                            'elapse': t_cost,
                        }
                    
                    results[file_info['image_path']] = result_entry
                    logger.info(f"进度：{idx+1}/{len(batch_files)}，处理文件 {file_path} 成功，耗时 {t_cost:.4f} 秒，识别文本为: {rec_text}")
                t_sum += t_cost
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
                results[file_info['image_path']] = {
                    'text': '',
                    'score': 0.0,
                    'elapse': 0.0,
                    'error': str(e)
                }
                # if 'Connection error.' == str(e):
                #     break
            if idx % 1000 == 0:
                # logger.info(f"进度：{idx+1}/{len(batch_files)}，已处理 {idx+1} 个文件，耗时 {t_sum:.4f} 秒")
                result_file = os.path.join(save_res_path, f'rec_results_{gpu_id}.json')
                with open(result_file, 'w', encoding='utf-8') as fout:
                    json.dump(results, fout, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"处理批次 {gpu_id} 时出错: {str(e)}")

    # 计算本进程的 acceptance 统计
    if acceptance_count > 0:
        proc_avg_acceptance_length = total_acceptance_length / acceptance_count
        proc_avg_full_accept_rate = total_full_accept_rate / acceptance_count
        logger.info(f"GPU {gpu_id} acceptance 统计: avg_acceptance_length={proc_avg_acceptance_length:.4f}, "
                     f"full_accept_rate={proc_avg_full_accept_rate:.4f}, samples={acceptance_count}")

    # 发送结果到主进程
    result_file = os.path.join(save_res_path, f'rec_results_{gpu_id}.json')
    with open(result_file, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    # 保存本进程的 acceptance 统计到单独文件，供主进程汇总
    if acceptance_count > 0:
        stats_file = os.path.join(save_res_path, f'acceptance_stats_{gpu_id}.json')
        with open(stats_file, 'w', encoding='utf-8') as fout:
            json.dump({
                'total_acceptance_length': total_acceptance_length,
                'total_full_accept_rate': total_full_accept_rate,
                'count': acceptance_count,
            }, fout, ensure_ascii=False, indent=2)
    
    logger.info(f"{gpu_id} 共处理 {len(batch_files)} 个文件，耗时 {t_sum:.4f} 秒")



def split_list(lst, n):
    """将列表平均分成n个部分"""
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def main_eval_unirec(save_res_path='./rec_dolphinv2_results_unirec_bench_fixprompt/', num_gpus=8, model_args=None, build_model=None, process_element_func=None, is_unirec=True, img_path=None, resume = False, batch_size=1, unirec_bench_path=None):   
    
    logger.info(f"使用 {num_gpus} 个GPU进行并行处理")
    
    if is_unirec and img_path is None:
        unirec_bench_path = '/apdcephfs_gy4/share_303464260/duyongkun/aliyundata/1daf5d619a4a4b609457804f01c93ecd/tmp/unirec_data/OmniDocBench_crop' if unirec_bench_path is None else unirec_bench_path
        # 加载标签数据
        logger.info(f"加载 {unirec_bench_path} 中的标签数据.....")
        with open(os.path.join(unirec_bench_path, 'OmniDocBench_crop_new_fixlang_num.json'), 'r', encoding='utf-8') as f:
            label_json_data = json.load(f)
        logger.info(f"加载 {os.path.join(unirec_bench_path, 'OmniDocBench_crop_new_fixlang_num.json')} 中的标签数据完成，共 {len(label_json_data)} 个样本")
        img_path = os.path.join(unirec_bench_path, 'images_crop')
    else:
        label_json_data = []
        img_list = os.listdir(img_path)
        for img_name in img_list:
            label_json_data.append({
                'image_path': img_name,
                'text_type': 'text',
                'data_source': 'other',
            })
        logger.info(f"加载 {img_path} 中的数据完成，共 {len(label_json_data)} 个样本")
        random.shuffle(label_json_data)
        
    
    if not os.path.exists(save_res_path):
        os.makedirs(save_res_path)
    
    # 检查结果是否已经存在
    flag_exist = False
    label_json_data_filter = []
    if os.path.exists(os.path.join(save_res_path, 'rec_results.json')):
        logger.info(f"结果文件 {os.path.join(save_res_path, 'rec_results.json')} 已存在，跳过处理")
        with open(os.path.join(save_res_path, 'rec_results.json'), 'r', encoding='utf-8') as f:
            pred_json_data = json.load(f)
        logger.info(f"加载 {os.path.join(save_res_path, 'rec_results.json')} 中的数据完成，共 {len(pred_json_data)} 个样本")
        flag_exist = True
        notin_num = 0
        for label_info in label_json_data:
            # label_infor['text'] = label_infor['text'].replace('\n', '')
            img_name = label_info['image_path']
            # safe_name = quote(img_name)
            if resume:
                if img_name not in pred_json_data:
                    label_json_data_filter.append(label_info)
                    notin_num +=1
                else:
                    if img_name in pred_json_data and 'error' in pred_json_data[img_name]:
                        label_json_data_filter.append(label_info)
                # notin_num +=1
        logger.info(f"未识别 {notin_num} 个样本")
    else:
        pred_json_data = {}
        
    if flag_exist and len(label_json_data_filter) == 0:
        logger.info(f"已经全部识别完成，结果在 {os.path.join(save_res_path, 'rec_results.json')}")
    else:
        if flag_exist:
            logger.info(f"过滤后，共 {len(label_json_data_filter)} 个样本")
            label_json_data_precess = label_json_data_filter
            
        else:
            label_json_data_precess = label_json_data
        
        # 并行处理所有批次
        processes = []
        for gpu_id in range(num_gpus):
            p = multiprocessing.Process(
                target=process_batch,
                args=(build_model, gpu_id, model_args, process_element_func, label_json_data_precess[gpu_id::num_gpus], img_path, save_res_path, batch_size)
            )
            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()

        logger.info('success!')

    if not flag_exist or len(label_json_data_filter) > 0:
        while True:
            # 检查所有GPU是否完成
            all_done = True
            for gpu_id in range(num_gpus):
                result_file = os.path.join(save_res_path, f'rec_results_{gpu_id}.json')
                if not os.path.exists(result_file):
                    all_done = False
                    break
                else:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        pred_json_data.update(json.load(f))
            if all_done:
                break
            time.sleep(10)  # 等待10秒后检查 again
        result_file = os.path.join(save_res_path, f'rec_results.json')
        with open(result_file, 'w', encoding='utf-8') as fout:
            json.dump(pred_json_data, fout, ensure_ascii=False, indent=2)

        # 汇总所有进程的 acceptance 统计
        global_total_acceptance_length = 0.0
        global_total_full_accept_rate = 0.0
        global_acceptance_count = 0
        for gpu_id in range(num_gpus):
            stats_file = os.path.join(save_res_path, f'acceptance_stats_{gpu_id}.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                global_total_acceptance_length += stats['total_acceptance_length']
                global_total_full_accept_rate += stats['total_full_accept_rate']
                global_acceptance_count += stats['count']
        if global_acceptance_count > 0:
            global_avg_acceptance_length = global_total_acceptance_length / global_acceptance_count
            global_avg_full_accept_rate = global_total_full_accept_rate / global_acceptance_count
            logger.info(f"===== 全局 Acceptance 统计 =====")
            logger.info(f"Total samples: {global_acceptance_count}")
            logger.info(f"Global avg_acceptance_length: {global_avg_acceptance_length:.4f}")
            logger.info(f"Global avg_full_accept_rate: {global_avg_full_accept_rate:.4f}")
            # 保存全局统计到文件
            global_stats_file = os.path.join(save_res_path, 'acceptance_stats_global.json')
            with open(global_stats_file, 'w', encoding='utf-8') as fout:
                json.dump({
                    'global_avg_acceptance_length': global_avg_acceptance_length,
                    'global_avg_full_accept_rate': global_avg_full_accept_rate,
                    'total_samples': global_acceptance_count,
                }, fout, ensure_ascii=False, indent=2)
            logger.info(f"Global acceptance stats saved to {global_stats_file}")

    if is_unirec:
        # 计算指标
        exclude_text_type = []

        metrics, hierarchical_metrics = unirec_measure(pred_json_data, label_json_data, exclude_text_type=exclude_text_type, is_norm=False)
        # 计算平均指标
        avg_metrics = calculate_average_metrics(metrics)
        # 格式化为表格
        metric_tables, csv_rows = format_metrics_as_tables(avg_metrics)
        # 打印表格
        for metric_type, table_parts in metric_tables.items():
            print(f"\n===== Without Normalization {metric_type.capitalize()} Metrics Summary =====")
            for part in table_parts:
                print(part)
        # 保存指标结果
        save_metrics(avg_metrics, hierarchical_metrics, metric_tables, csv_rows, save_res_path, ignore_eq_table=False, is_norm=False)
        logger.info(f"Metrics saved to {save_res_path}")

        metrics, hierarchical_metrics = unirec_measure(pred_json_data, label_json_data, exclude_text_type=exclude_text_type, is_norm=True)
        # 计算平均指标
        avg_metrics = calculate_average_metrics(metrics)
        # 格式化为表格
        metric_tables, csv_rows = format_metrics_as_tables(avg_metrics)
        # 打印表格
        for metric_type, table_parts in metric_tables.items():
            print(f"\n===== With Normalization {metric_type.capitalize()} Metrics Summary =====")
            for part in table_parts:
                print(part)
        # 保存指标结果
        save_metrics(avg_metrics, hierarchical_metrics, metric_tables, csv_rows, save_res_path, ignore_eq_table=False, is_norm=True)
        logger.info(f"Metrics saved to {save_res_path}")

        



        exclude_text_type = ['equation', 'equation_mixed', 'table']

        metrics, hierarchical_metrics = unirec_measure(pred_json_data, label_json_data, exclude_text_type=exclude_text_type, is_norm=False)
        # 计算平均指标
        avg_metrics = calculate_average_metrics(metrics)
        # 格式化为表格
        metric_tables, csv_rows = format_metrics_as_tables(avg_metrics)
        # 打印表格
        for metric_type, table_parts in metric_tables.items():
            print(f"\n===== Exclude Eq and Table and without Normalization {metric_type.capitalize()} Metrics Summary =====")
            for part in table_parts:
                print(part)
        # 保存指标结果
        save_metrics(avg_metrics, hierarchical_metrics, metric_tables, csv_rows, save_res_path, ignore_eq_table=True, is_norm=False)
        logger.info(f"Metrics saved to {save_res_path}")

        metrics, hierarchical_metrics = unirec_measure(pred_json_data, label_json_data, exclude_text_type=exclude_text_type)
        # 计算平均指标
        avg_metrics = calculate_average_metrics(metrics)
        # 格式化为表格
        metric_tables, csv_rows = format_metrics_as_tables(avg_metrics)
        # 打印表格
        for metric_type, table_parts in metric_tables.items():
            print(f"\n===== Exclude Eq and Table and Normalization {metric_type.capitalize()} Metrics Summary =====")
            for part in table_parts:
                print(part)
        # 保存指标结果
        save_metrics(avg_metrics, hierarchical_metrics, metric_tables, csv_rows, save_res_path, ignore_eq_table=True, is_norm=True)
        logger.info(f"Metrics saved to {save_res_path}")

        metrics, hierarchical_metrics = unirec_measure(pred_json_data, label_json_data, exclude_text_type=exclude_text_type, rm_space=True)
        # 计算平均指标
        avg_metrics = calculate_average_metrics(metrics)
        # 格式化为表格
        metric_tables, csv_rows = format_metrics_as_tables(avg_metrics)
        # 打印表格
        for metric_type, table_parts in metric_tables.items():
            print(f"\n===== Exclude Eq and Table and Normalization {metric_type.capitalize()} Metrics Summary =====")
            for part in table_parts:
                print(part)
        # 保存指标结果
        save_metrics(avg_metrics, hierarchical_metrics, metric_tables, csv_rows, save_res_path, ignore_eq_table=True, is_norm=True, rm_space=True)
        logger.info(f"Metrics saved to {save_res_path}")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main_eval_unirec()

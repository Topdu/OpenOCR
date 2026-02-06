import json
import os
import re
from typing import List, Dict, Any


def truncate_repeated_tail(s, threshold=20, keep=1):
    """
    如果字符串尾部重复出现某个元素超过threshold次，则只保留keep个该元素

    参数:
        s: 输入字符串
        threshold: 重复次数阈值，默认20
        keep: 保留的重复次数，默认5

    返回:
        处理后的字符串
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


def extract_table_from_html(html_string):
    """Extract and clean table tags from HTML string"""
    try:
        table_pattern = re.compile(r'<table.*?>.*?</table>', re.DOTALL)
        tables = table_pattern.findall(html_string)
        tables = [
            re.sub(r'<table[^>]*>', '<table>', table) for table in tables
        ]
        # tables = [re.sub(r'>\n', '>', table) for table in tables]
        return '\n'.join(tables)
    except Exception as e:
        print(f'extract_table_from_html error: {str(e)}')
        return f'<table><tr><td>Error extracting table: {str(e)}</td></tr></table>'


rules = [
    (r'-<\|sn\|>', ''),
    (r'<\|sn\|>', ''),
    (r'<\|unk\|>', ''),
    (r'\uffff', ''),
    (r'_{4,}', '___'),
    (r'\.{4,}', '...'),
    # (r'(\d) +', r'\1'),
    # (r' +(\d)', r'\1'),
]

# pattern = r"\\(big|Big|bigg|Bigg)\{([^{}])\}"
pattern = r'\\(big|Big|bigg|Bigg|bigl|bigr|Bigl|Bigr|biggr|biggl|Biggl|Biggr)\{(\\?[{}\[\]\(\)\|])\}'


def fix_latex_brackets(text: str) -> str:
    return re.sub(pattern, r'\\\1\2', text)


class MarkdownConverter:
    """Convert structured recognition results to Markdown format"""

    def __init__(self):
        # Define heading levels for different section types
        self.heading_levels = {
            'sec_0': '#',
            'sec_1': '##',
            'sec_2': '###',
            'sec_3': '###',
            'sec_4': '###',
            'sec_5': '###',
        }

        # Define which labels need special handling
        self.special_labels = {
            'sec_0', 'sec_1', 'sec_2', 'sec_3', 'sec_4', 'sec_5', 'list',
            'equ', 'tab', 'fig'
        }

        # Define replacements for special formulas
        self.replace_dict = {
            '\\bm': '\mathbf ',
            '\eqno': '\quad ',
            '\quad': '\quad ',
            '\leq': '\leq ',
            '\pm': '\pm ',
            '\\varmathbb': '\mathbb ',
            '\in fty': '\infty',
            '\mu': '\mu ',
            '\cdot': '\cdot ',
            '\langle': '\langle ',
            '\pm': '\pm '
        }
        # self.bigpattern = pattern = r"\\(big|Big|bigg|Bigg)\{(\\?[()\[\]{}]|\\langle|\\rangle)|\|\}"

    def try_remove_newline(self, text: str) -> str:
        try:
            # Preprocess text to handle line breaks
            text = text.strip()
            text = text.replace('-\n', '')

            # Handle Chinese text line breaks
            def is_chinese(char):
                return '\u4e00' <= char <= '\u9fff'

            lines = text.split('\n')
            processed_lines = []

            # Process all lines except the last one
            for i in range(len(lines) - 1):
                current_line = lines[i].strip()
                next_line = lines[i + 1].strip()

                # Always add the current line, but determine if we need a newline
                if current_line:  # If current line is not empty
                    if next_line:  # If next line is not empty
                        # For Chinese text handling
                        if is_chinese(current_line[-1]) and is_chinese(
                                next_line[0]):
                            processed_lines.append(current_line)
                        else:
                            processed_lines.append(current_line + ' ')
                    else:
                        # Next line is empty, add current line with newline
                        processed_lines.append(current_line + '\n')
                else:
                    # Current line is empty, add an empty line
                    processed_lines.append('\n')

            # Add the last line
            if lines and lines[-1].strip():
                processed_lines.append(lines[-1].strip())

            text = ''.join(processed_lines)
            return text

        except Exception as e:
            print(f'try_remove_newline error: {str(e)}')
            return text  # Return original text on error

    def _handle_text(self, text: str) -> str:
        """
        Process regular text content, preserving paragraph structure
        """
        try:
            if not text:
                return ''
            if text in ['图中没有可识别的文本。', '图中无文本。', '图中没有文本。']:
                return ''
            for rule in rules:
                text = re.sub(rule[0], rule[1], text)
            # Process formulas in text before handling other text processing
            text = self._process_formulas_in_text(text)
            text = text.replace('$\bullet$', '•')
            # rm html table tag
            if '<table>' in text:
                print(text)
                text = re.sub(r'</?(table|tr|th|td|thead|tbody|tfoot)[^>]*>',
                              '',
                              text,
                              flags=re.IGNORECASE)
                text = re.sub(r'\n\s*\n+', '\n', text)
                print(text)
            # text = self.try_remove_newline(text)
            return text
        except Exception as e:
            print(f'_handle_text error: {str(e)}')
            return text  # Return original text on error

    def _process_formulas_in_text(self, text: str) -> str:
        """
        Process mathematical formulas in text by iteratively finding and replacing formulas.
        - Identify inline and block formulas
        - Replace newlines within formulas with \\
        """
        try:
            text = text.replace(r'\upmu',
                                r'\mu').replace('\(', '$').replace('\)', '$')
            for key, value in self.replace_dict.items():
                text = text.replace(key, value)
            return text

        except Exception as e:
            print(f'_process_formulas_in_text error: {str(e)}')
            return text  # Return original text on error

    def _remove_newline_in_heading(self, text: str) -> str:
        """
        Remove newline in heading
        """
        try:
            # Handle Chinese text line breaks
            def is_chinese(char):
                return '\u4e00' <= char <= '\u9fff'

            # Check if the text contains Chinese characters
            if any(is_chinese(char) for char in text):
                return text.replace('\n', '')
            else:
                return text.replace('\n', ' ')

        except Exception as e:
            print(f'_remove_newline_in_heading error: {str(e)}')
            return text

    def _handle_heading(self, text: str, label: str) -> str:
        """
        Convert section headings to appropriate markdown format
        """
        try:
            level = self.heading_levels.get(label, '#')
            text = text.strip()
            text = self._remove_newline_in_heading(text)
            text = self._handle_text(text)
            return f'{level} {text}\n\n'

        except Exception as e:
            print(f'_handle_heading error: {str(e)}')
            return f'# Error processing heading: {text}\n\n'

    def _handle_list_item(self, text: str) -> str:
        """
        Convert list items to markdown list format
        """
        try:
            return f'- {text.strip()}\n'
        except Exception as e:
            print(f'_handle_list_item error: {str(e)}')
            return f'- Error processing list item: {text}\n'

    def _handle_figure(self, text: str, section_count: int) -> str:
        """
        Handle figure content
        """
        try:
            # Check if it's a file path starting with "figures/"
            if text.startswith('figures/'):
                # Convert to relative path from markdown directory to figures directory
                relative_path = f'../{text}'
                return f'![Figure {section_count}]({relative_path})\n\n'

            # Check if it's already a markdown format image link
            if text.startswith('!['):
                # Already in markdown format, return directly
                return f'{text}\n\n'

            # If it's still base64 format, maintain original logic
            if text.startswith('data:image/'):
                return f'![Figure {section_count}]({text})\n\n'
            elif ';' in text and ',' in text:
                return f'![Figure {section_count}]({text})\n\n'
            else:
                # Assume it's raw base64, convert to data URI
                img_format = 'png'
                data_uri = f'data:image/{img_format};base64,{text}'
                return f'![Figure {section_count}]({data_uri})\n\n'

        except Exception as e:
            print(f'_handle_figure error: {str(e)}')
            return f'*[Error processing figure: {str(e)}]*\n\n'

    def _handle_table(self, text: str) -> str:
        """
        Convert table content to markdown format
        """
        try:
            markdown_content = []
            markdown_table = extract_table_from_html(text)
            table_content = markdown_table.replace('<tdcolspan=',
                                                   '<td colspan=')
            table_content = table_content.replace('<tdrowspan=',
                                                  '<td rowspan=')
            table_content = table_content.replace('"colspan=', '" colspan=')
            table_content = re.sub(r'<\|sn\|>', '', table_content)
            table_content = re.sub(r'<\|unk\|>', '', table_content)
            table_content = re.sub(r'\uffff', '', table_content)
            table_content = re.sub(r'_{4,}', '___', table_content)
            table_content = re.sub(r'\.{4,}', '...', table_content)

            table_content = re.sub(r'</td\s+colspan="[^"]*"\s*>',
                                   '</td>',
                                   table_content,
                                   flags=re.IGNORECASE)
            table_content = re.sub(r'</td\s+rowspan="[^"]*"\s*>',
                                   '</td>',
                                   table_content,
                                   flags=re.IGNORECASE)
            table_content = re.sub(r'</th\s+rowspan="[^"]*"\s*>',
                                   '</th>',
                                   table_content,
                                   flags=re.IGNORECASE)
            table_content = re.sub(r'</th\s+colspan="[^"]*"\s*>',
                                   '</th>',
                                   table_content,
                                   flags=re.IGNORECASE)
            table_content = table_content.replace('\(', '$').replace('\)', '$')
            table_content = table_content.replace('\[',
                                                  '$$').replace('\]', '$$')
            # markdown_table = re.sub(r'>\s*\n+\s*',
            #                         '>',
            #                         table_content,
            #                         flags=re.DOTALL)
            markdown_content.append(table_content + '\n')
            return '\n'.join(markdown_content) + '\n\n'

        except Exception as e:
            print(f'_handle_table error: {str(e)}')
            return f'*[Error processing table: {str(e)}]*\n\n'

    def _handle_formula(self, text: str) -> str:
        """
        Handle formula-specific content
        """
        try:
            text = text.replace(r'\upmu', r'\mu')
            result = re.sub(r'\\] \(\d+\)\n\n', r'\\]', text)
            result = re.sub(r'<\|sn\|>', '', result)
            result = re.sub(r'<\|unk\|>', '', result)
            result = re.sub(r'\uffff', '', result)
            result = re.sub(r'_{4,}', '___', result)
            result = result.replace('\]\n*\[', '\\\\')
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
            text = text.strip('$').rstrip('\ ').replace(r'\upmu', r'\mu')
            for key, value in self.replace_dict.items():
                text = text.replace(key, value)
            processed_text = '$$' + text + '$$'
            processed_text = processed_text.replace('\n', '\\\\\n')
            processed_text = fix_latex_brackets(processed_text)

            # 替换为 \big( 或 \Big[ 等
            # processed_text = re.sub(self.bigpattern, r"\\\1\2", processed_text)
            return f'{processed_text}\n\n'

        except Exception as e:
            print(f'_handle_formula error: {str(e)}')
            return f'*[Error processing formula: {str(e)}]*\n\n'

    def convert(self, recognition_results: List[Dict[str, Any]]) -> str:
        """
        Convert recognition results to markdown format
        """
        try:
            markdown_content = []

            # {'text', 'header', 'number', 'figure_title', 'content', 'chart', 'footer',
            #  'vision_footnote', 'aside_text', 'inline_formula', 'algorithm', 'table',
            #  'image', 'display_formula', 'footer_image', 'vertical_text',
            #  'header_image', 'paragraph_title', 'reference_content', 'doc_title', 'footnote', 'seal', 'abstract'}
            #  block_label_set_ignore = set(['chart', 'image', 'footer_image', 'header_image', 'seal'])

            for section_count, result in enumerate(recognition_results):
                try:
                    label = result.get('label', '')
                    text = result.get('text_unirec', '').strip()

                    # Skip empty text
                    if not text:
                        continue
                    if label in [
                            'header', 'header_image', 'footer_image', 'footer',
                            'aside_text', 'inline_formula', 'number'
                    ]:
                        continue
                    if label == 'number' and (section_count == 0
                                              or section_count
                                              == len(recognition_results) - 1):
                        continue

                    text = truncate_repeated_tail(text)
                    if label == 'doc_title':
                        label = 'sec_0'
                    elif label == 'paragraph_title':
                        label = 'sec_1'
                    # Handle different content types
                    if label in {
                            'sec_0', 'sec_1', 'sec_2', 'sec_3', 'sec_4',
                            'sec_5'
                    }:
                        markdown_content.append(
                            self._handle_heading(text, label))
                    elif label in ['image', 'chart', 'seal']:
                        markdown_content.append(
                            self._handle_figure(text, section_count))
                    elif label == 'table':
                        markdown_content.append(self._handle_table(text))
                    elif label in ['display_formula']:
                        markdown_content.append(self._handle_formula(text))
                    elif label == 'list':
                        markdown_content.append(self._handle_list_item(text))
                    elif label == 'code':
                        markdown_content.append(f'```bash\n{text}\n```\n\n')
                    else:
                        # Handle regular text (paragraphs, etc.)
                        processed_text = self._handle_text(text)
                        markdown_content.append(f'{processed_text}\n\n')

                except Exception as e:
                    print(f'Error processing item {section_count}: {str(e)}')
                    # Add a placeholder for the failed item
                    markdown_content.append(
                        f'*[Error processing content]*\n\n')

            # Join all content and apply post-processing
            result = ''.join(markdown_content)
            return result

        except Exception as e:
            print(f'convert error: {str(e)}')
            return f'Error generating markdown content: {str(e)}'


if __name__ == '__main__':

    markdown_converter = MarkdownConverter()
    img_path = f'./OmniDocBench/images'
    save_res_path = './rec_results'
    img_path_list = os.listdir(img_path)
    md_save_path = f'{save_res_path}/markdown_results'
    os.makedirs(md_save_path, exist_ok=True)
    for img_name in img_path_list:
        file_path = os.path.join(save_res_path, img_name[:-4] + '_res.json')
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        block_list = json_data['boxes']
        markdown_content = markdown_converter.convert(block_list)
        with open(os.path.join(md_save_path, img_name[:-4] + '.md'), 'w') as f:
            f.write(markdown_content)

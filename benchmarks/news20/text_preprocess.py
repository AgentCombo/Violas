import re
import os
from tqdm import tqdm


def preprocess_newsgroup_text(text: str) -> str:
    """
    20Newsgroups （）
    """
    if not text:
        return ""

    lines = text.split('\n')

    # Step 1: Find the beginning of the text (skip the email header)
    body_start = 0
    header_fields = [
        'Xref:', 'Path:', 'From:', 'Newsgroups:', 'Subject:',
        'Message-ID:', 'Date:', 'Expires:', 'Followup-To:',
        'Distribution:', 'Organization:', 'Approved:', 'Supersedes:',
        'Keywords:', 'Summary:', 'NNTP-Posting-Host:', 'Reply-To:',
        'Sender:', 'References:', 'In-Reply-To:', 'Content-Type:',
        'Content-Transfer-Encoding:', 'MIME-Version:', 'Lines:',
        'Archive-name:'
    ]

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # After finding the first blank line, the text begins
        if line_stripped == '' and i > 0:
            # Check if there are previous email headers
            prev_line = lines[i-1].strip() if i > 0 else ''
            if any(prev_line.startswith(field) for field in header_fields) or prev_line.startswith('NNTP'):
                body_start = i + 1
                break
        # If you encounter non-email header content, it may already be the body text.
        if line_stripped and not any(line_stripped.startswith(field) for field in header_fields):
            if i > 5:  # Assume there are at least 5 lines of email headers
                body_start = i
                break

    # Step 2: Extract the text and process it by paragraphs
    body_lines = lines[body_start:]

    # Split into paragraphs by blank lines
    paragraphs = []
    current_paragraph = []

    for line in body_lines:
        line_stripped = line.strip()

        if line_stripped == '':
            # A blank line indicates the end of a paragraph
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = []
        else:
            # Clean row contents
            cleaned_line = line_stripped
            # Remove quotation marks
            cleaned_line = re.sub(r'^[>|\s]+', '', cleaned_line)
            # remove number
            cleaned_line = re.sub(r'^\d+\.\s*', '', cleaned_line)
            # Preserve meaningful punctuation
            cleaned_line = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\@\#\$\%\&\*\+\=\<\>]', '', cleaned_line)

            if len(cleaned_line.strip()) > 3:
                current_paragraph.append(cleaned_line.strip())

    # Don't forget the last paragraph
    if current_paragraph:
        paragraphs.append(current_paragraph)

    # Step 3: Use spaces to connect each paragraph and separate paragraphs with double line breaks.
    cleaned_paragraphs = []
    for para in paragraphs:
        para_text = ' '.join(para)
        para_text = re.sub(r'\s+', ' ', para_text).strip()
        if len(para_text) > 10:  # Filter paragraphs that are too short
            cleaned_paragraphs.append(para_text)

    cleaned_text = '\n\n'.join(cleaned_paragraphs)

    return cleaned_text


def collect_text_data_from_folders(root_path: str):
    """
    : root_path: max_files_per_folder: : all_data: {folder_name: [(file_name, processed_text), ...]}
    """
    all_data = {}
    folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

    # folders = folders[:3] #Small amount of testing
    print(f"Folders: {folders}")

    for folder in tqdm(folders, desc="Encoding batches"):
        folder_path = os.path.join(root_path, folder)

        files = [f for f in os.listdir(folder_path)]
        files.sort()

        folder_data = []
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    raw_text = f.read()
                text = preprocess_newsgroup_text(raw_text)
                if text.strip():
                    folder_data.append((file_name, text))
            except Exception as e:
                continue

        if folder_data:
            all_data[folder] = folder_data

    print(f"\nCollected in total {len(all_data)} folder data")
    return all_data

import html
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime

import nltk
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

# On macOS the system Python often can't verify SSL certs out of the box,
# breaking both nltk downloads and arxiv requests. Point the default HTTPS
# context at certifi's bundle so the tool works without extra setup.
try:
    import certifi
    ssl._create_default_https_context = (
        lambda *a, **k: ssl.create_default_context(cafile=certifi.where())
    )
except ImportError:
    pass

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# arXiv's official, public, no-registration API. Returns Atom XML.
# Docs + terms of use: https://info.arxiv.org/help/api/index.html
API_URL = 'http://export.arxiv.org/api/query'
ATOM_NS = '{http://www.w3.org/2005/Atom}'
PAGE_SIZE = 100        # entries to request per API call
MAX_PAGES = 10         # safety cap so we never loop forever
REQUEST_DELAY = 3.0    # seconds between calls (arXiv asks for >= 3s)


def is_float(token):
    return re.match("^\d+?\.\d+?$", token) is not None


def is_citation_year(tokens, i):
    if len(tokens[i]) != 4:
        return False
    if re.match(r'[12][0-9]{3}', tokens[i]) is None:
        return False
    if i == 0 or i == len(tokens) - 1:
        return False
    if (tokens[i - 1] == ',' or tokens[i - 1] == '(') and tokens[i + 1] == ')':
        return True
    return False


def is_list_numer(tokens, i, value):
    if value < 1 or value > 4:
        return False
    if i == len(tokens) - 1:
        return False

    if (i == 0 or tokens[i - 1] in set(['(', '.', ':'])) and tokens[i + 1] == ')':
        return True
    return False


def has_number(sent):
    tokens = word_tokenize(sent)
    for i, token in enumerate(tokens):
        if token.endswith('\\'):
            token = token[:-2]
        if token.endswith('x'):  # sometimes people write numbers as 1.7x
            token = token[:-1]
        if token.startswith('x'):  # sometimes people write numbers as x1.7
            token = token[1:]
        if token.startswith('$') and token.endswith('$'):
            token = token[1:-1]
        if is_float(token):
            return True
        try:
            value = int(token)
        except:
            continue
        if (not is_citation_year(tokens, i)) and (not is_list_numer(tokens, i, value)):
            return True

    return False


def contains_sota(sent):
    return 'state-of-the-art' in sent or 'state of the art' in sent or 'SOTA' in sent


def extract_line(abstract, keyword, limit):
    lines = []
    numbered_lines = []
    kw_mentioned = False
    abstract = abstract.replace("et. al", "et al.")
    sentences = abstract.split('. ')
    kw_sentences = []
    for i, sent in enumerate(sentences):
        if keyword in sent.lower():
            kw_mentioned = True
            if has_number(sent):
                numbered_lines.append(sent)
            elif contains_sota(sent):
                numbered_lines.append(sent)
            else:
                kw_sentences.append(sent)
                lines.append(sent)
            continue

        if kw_mentioned and has_number(sent):
            if not numbered_lines:
                numbered_lines.append(kw_sentences[-1])
            numbered_lines.append(sent)
        if kw_mentioned and contains_sota(sent):
            lines.append(sent)

    if len(numbered_lines) > 0:
        return '. '.join(numbered_lines), True
    return '. '.join(lines[-2:]), False


def get_report(paper, keyword):
    if keyword in paper['abstract'].lower():
        title = html.unescape(paper['title'])
        headline = '{} ({} - {})\n'.format(title, paper['authors'][0], paper['date'])
        abstract = html.unescape(paper['abstract'])
        extract, has_number = extract_line(abstract, keyword, 280 - len(headline))
        if extract:
            report = headline + extract + '\nLink: {}'.format(paper['main_page'])
            return report, has_number
    return '', False


def format_date(published):
    """Convert an Atom timestamp (2026-06-18T17:59:05Z) to '18 June, 2026'."""
    try:
        dt = datetime.strptime(published[:10], '%Y-%m-%d')
        return dt.strftime('%-d %B, %Y')
    except (ValueError, TypeError):
        return published


def parse_entry(entry):
    """Turn one Atom <entry> element into a paper dict."""
    def text(tag):
        node = entry.find(ATOM_NS + tag)
        return node.text.strip() if node is not None and node.text else ''

    paper = {}
    # Collapse the whitespace arXiv puts inside <title>/<summary>.
    paper['title'] = ' '.join(text('title').split())
    paper['abstract'] = ' '.join(text('summary').split())
    paper['date'] = format_date(text('published'))

    authors = []
    for author in entry.findall(ATOM_NS + 'author'):
        name = author.find(ATOM_NS + 'name')
        if name is not None and name.text:
            authors.append(name.text.strip())
    paper['authors'] = authors

    # <id> is the canonical abstract page; force https.
    paper['main_page'] = text('id').replace('http://', 'https://')
    paper['pdf'] = ''
    for link in entry.findall(ATOM_NS + 'link'):
        if link.get('title') == 'pdf':
            paper['pdf'] = link.get('href', '').replace('http://', 'https://')
    return paper


def build_query(keyword):
    """
    Build the arXiv API search_query.

    If the keyword is plain English, restrict to the computer-science archive
    to avoid ambiguous hits from other fields (e.g. 'transformer' in physics).
    Unknown words (model/dataset jargon) are searched across all of arXiv.
    """
    keyword = keyword.lower()
    words = keyword.split()
    cs_only = keyword in set(['gan', 'bpc'])
    if not cs_only:
        spell = SpellChecker()
        cs_only = not spell.unknown(words)

    search = 'all:{}'.format(keyword)
    if cs_only:
        search += ' AND cat:cs.*'
    return search


def fetch_page(search_query, start):
    params = urllib.parse.urlencode({
        'search_query': search_query,
        'start': start,
        'max_results': PAGE_SIZE,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending',
    })
    req = urllib.request.Request(API_URL + '?' + params)
    response = urllib.request.urlopen(req)
    return response.read()


def get_papers(keyword, num_results=5):
    keyword = keyword.lower()
    search_query = build_query(keyword)

    shown = 0
    unshown = []          # papers that mention the keyword but report no numbers
    any_results = False

    for page in range(MAX_PAGES):
        if shown >= num_results:
            break

        try:
            raw = fetch_page(search_query, page * PAGE_SIZE)
        except urllib.error.HTTPError as e:
            print('Error {}: problem accessing the arXiv API'.format(e.code))
            return
        except urllib.error.URLError as e:
            print('Error: could not reach the arXiv API ({})'.format(e.reason))
            return

        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            print('Error: could not parse the arXiv API response')
            return

        entries = root.findall(ATOM_NS + 'entry')
        if not entries:
            break  # no more papers to page through

        for entry in entries:
            if shown >= num_results:
                break
            any_results = True
            paper = parse_entry(entry)
            if not paper['authors']:
                continue
            report, paper_has_number = get_report(paper, keyword)
            if paper_has_number:
                print(report)
                print('====================================================')
                shown += 1
            elif report:
                unshown.append(report)

        if len(entries) < PAGE_SIZE:
            break  # last page reached

        if shown < num_results and page < MAX_PAGES - 1:
            time.sleep(REQUEST_DELAY)  # be polite to arXiv

    # Fall back to keyword-mentioning papers without numeric results.
    if shown < num_results and unshown:
        for report in unshown[:num_results - shown]:
            print(report)
            print('====================================================')
            shown += 1

    if shown == 0:
        if any_results:
            print('Sorry, we found papers but none with a usable abstract '
                  'summary for the word {}'.format(keyword))
        else:
            print('Sorry, we were unable to find any abstract with the word '
                  '{}'.format(keyword))


def main():
    if 'nt' in os.name:
        try:
            import win_unicode_console
            win_unicode_console.enable()
        except ImportError:
            warnings.warn('On Windows, encoding errors may arise when displaying the data.\n'
                          'If such errors occur, please install `win_unicode_consolde` via \n'
                          'the command `pip install win-unicode-console`.')

    if len(sys.argv) < 2:
        raise ValueError('You must specify a keyword')

    try:
        num_results = int(sys.argv[-1])
        assert num_results > 0, 'You must choose to show a positive number of results'
        keyword = ' '.join(sys.argv[1:-1])

    except ValueError:
        keyword = ' '.join(sys.argv[1:])
        num_results = 5

    get_papers(keyword, num_results)


if __name__ == '__main__':
    main()

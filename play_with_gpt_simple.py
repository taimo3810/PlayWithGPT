"""
created_at: 2021-09-01
author: taimo3810

"""

import datetime
import io
import json
import os
import pprint
import re
import time
import urllib.request

import fitz
import openai
import requests
import dotenv
dotenv.load_dotenv()

# set openai api key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_KEY")


def translate_by_gpt(text, current_language, target_language):
    translated_text = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        organization=openai.organization,
        messages=[
            {"role": "user", "content": f"Translate the following {current_language} text into {target_language}: {text}"},
        ]
    ).choices[0].message.content
    return translated_text
def translate_japanese_into_english(text):
    # if user_input is japanese, translate it into english
    if re.search(r'[ぁ-んァ-ン]', text):
        text = translate_by_gpt(text, "Japanese", "English")
    return text

def parse_arxiv_result(search_result):

    return

def read_pdf(pdf_link):
    from_local = os.path.exists(pdf_link)
    # Send a GET request to the URL
    if from_local:
        with open(pdf_link, "rb") as f:
            pdf = io.BytesIO(f.read())
    else:
        response = requests.get(pdf_link)
        assert response.status_code == 200, f"Failed to get PDF: {response.status_code}"
        pdf = io.BytesIO(response.content)

    # decode binary pdf to text
    with fitz.open(stream=pdf) as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    # strip [new line] from text
    text = re.sub(r'\n', ' ', text)

    # replace consecutive spaces with single space
    text = re.sub(r' +', ' ', text)

    return text


def get_date_from_string(date_string: str):
    # 2021-08-31T13:00:00Z is an example of date_string
    date_string = date_string.split("T")
    date_string[1] = date_string[1].split("Z")[0]
    date_string = " ".join(date_string)

    date = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    # remove timezone info
    date = date.replace(tzinfo=None)

    return f"{date.year}/{date.month}/{date.day}"

def parse_search_results_into_dict(search_result: str):
    # strip [new line] from search_result
    search_result = re.sub(r'\n', ' ', search_result)

    # parse entry
    entries = search_result.split('<entry>')[1:]
    parsed_entries = []
    if len(entries) == 0:
        return parsed_entries
    for e in entries:
        # print(e)
        dict = {
            "authors": [], "title": "", "summary": "", "published": "",
            "updated": "", "id": "", "pdf_link": "", "category": [],
            "comments": []
        }
        # parse authors
        for author in re.findall(r'<author>(.*?)</author>', e):
            name = re.findall(r'<name>(.*?)</name>', author)
            dict["authors"].extend(name)

        # parse title
        title = re.search(r'<title>(.*?)</title>', e).group(1)
        dict["title"] = title

        # parse summary
        summary = re.search(r'<summary>(.*?)</summary>', e).group(1)
        dict["abstract"] = summary

        # parse published
        published = re.search(r'<published>(.*?)</published>', e).group(1)
        dict["published"] = get_date_from_string(published)

        # parse updated
        updated = re.search(r'<updated>(.*?)</updated>', e).group(1)
        dict["updated"] = get_date_from_string(updated)

        # parse id
        id = re.search(r'<id>(.*?)</id>', e).group(1)
        dict["id"] = id

        # parse pdf_link
        pdf_link = re.search(r'<link title="pdf" href="(.*?)" rel="related" type="application/pdf"/>', e).group(1)
        dict["pdf_link"] = pdf_link

        # parse category
        category = re.findall(r'<category term="(.*?)"', e)
        dict["category"] = category

        # parse page
        parsed_result = re.findall(r'<arxiv:comment[^>]*>(.*?)</arxiv:comment>', e)
        dict["comments"] = parsed_result

        parsed_entries.append(dict)
    return parsed_entries
def extract_keywords(user_input, keyword_limit=3, gpt_model="gpt-3.5-turbo-16k"):

    completion = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[{"role": "user", "content": user_input}],
        functions=[
            {
                "name": "extract_keywords",
                "description": f"Get a list of English keywords for searching for papers."
                               f"A keyword is a representative word that describes the topic of a paper."
                               f"Words surrounded by double quotation marks are treated as a single keyword."
                               f"Correct the keywords if you find any mistakes or typos. If keywords include Japanese keywords, please translate them into english.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "A keyword string"
                            },
                            "description": f"List of keyword strings to be used for searching for papers"
                        }
                    },
                    "required": ["keywords"]
                }
            }
        ],
        function_call={"name": "extract_keywords"},
    )
    reply_content = completion.choices[0].message
    funcs = reply_content.to_dict()['function_call']['arguments']
    funcs = json.loads(funcs)


    if len(funcs['keywords']) <= 2:
        return funcs['keywords']
    # if keywords amount is more than 2,
    # remove first two keywords because they are too unique to search papers
    slide_size = min(2, len(funcs['keywords']))
    keywords = funcs['keywords'][slide_size:keyword_limit+slide_size]
    return keywords



def download_pdf_from_url(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

def download_paper_pdfs(papers):
    for paper in papers:
        save_file_title = paper['title'].replace(":", "").replace(" ", "_")
        download_pdf_from_url(
            paper['pdf_link'],
            f"pdfs/{save_file_title}.pdf"
        )
        time.sleep(1)
def parse_keywords_into_search_query(keywords: list):
    for i, keyword in enumerate(keywords):
        keywords[i] = keyword.replace(" ", "+")
    return "+AND+".join(keywords)
def search_papers(user_input, input_pdf=None, search_papers_amount = 10):

    # extract keywords from user input
    keywords = []
    if user_input:
        # if user input is japanese, translate it into english
        # requesting by japanese keyword cause error
        user_input = translate_japanese_into_english(user_input)
        keywords = extract_keywords(user_input)

    # if user input is empty and pdf is uploaded, extract keywords from pdf
    if input_pdf:
        pdf_text = read_pdf(input_pdf)

        # if the pdf is written in japanese, use smaller chunk size to prevent token limit error
        chunk_size_for_keywords = 25000
        if re.search(r'[ぁ-んァ-ン]', pdf_text[:1000]):
            chunk_size_for_keywords //= 2

        # extract keywords from pdf text after checking pdf text to prevent token limit error
        keywords.extend(extract_keywords(pdf_text[:chunk_size_for_keywords], keyword_limit=3, gpt_model="gpt-3.5-turbo-16k"))

    # if keywords are extracted, search papers. otherwise, return empty list
    papers = []
    if len(keywords) > 0:

        # search papers from arxiv using  keywords extracted by GPT API
        search_query = parse_keywords_into_search_query(keywords)
        url = f'http://export.arxiv.org/api/query?search_query={search_query}' \
              f'&start=0&max_results={search_papers_amount}' \
              f'&sortBy=relevance' \
              f'&sortOrder=descending'

        data = urllib.request.urlopen(url)
        data = data.read().decode('utf-8')

        # parse xml data into dict
        papers = parse_search_results_into_dict(data)

    return papers

def chunk_text(text, chunk_size=6000):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


def summarize_target_info_in_paper(pdf_file, target_information, output_language):
    summary = ""
    if pdf_file is None and target_information is None or target_information == "":
        Warning("No target information is given.")
        return summary

    # read pdf from local
    paper_text = read_pdf(pdf_file)

    # if the pdf is written in japanese, use smaller chunk size to prevent token limit error
    chunk_size = 30000
    if re.search(r'[ぁ-んァ-ン]', paper_text[:1000]):
        chunk_size //= 2

    # chunk text to prevent token limit error
    pdf_text_chunks = chunk_text(paper_text, chunk_size=30000)

    # summarize target information in paper
    if len(pdf_text_chunks) > 0:
        summary_format_prompt = "#Format\n  #One sentence summary\n  #Detail\n  #Key Points\n\n"
        command_prompt ="# Procedure of Your Job\n" \
                        f"1. find information related to '{target_information} in the paper (#Paper). Otherwise, Please return 'No information is found.' and stop the task." \
                        f"2. summarize all the portions of the academic paper (#Paper) that focus only on '{target_information}' in English." \
                        f"3. format the summary as follows:\n" \

        for text_chunk in pdf_text_chunks:
            # combine parts of the prompt into one prompt
            prompt_paper_chunk = f"#Paper\n {text_chunk}"
            prompt = f"{prompt_paper_chunk}{command_prompt}{summary_format_prompt}"

            # ask GPT to summarize target information in paper
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                organization=openai.organization,
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )

            # only when the response is not "No information is found.", summary variable is updated
            if response.choices[0].message.content != "No information is found.":
                summary = response.choices[0].message.content
            break

        # translate summary into output language
        if summary != "" and output_language != "English":
            summary = translate_by_gpt(summary, "English", output_language)
    return summary


def translate_by_gpt(text, current_language, target_language):
    translated_text = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        organization=openai.organization,
        messages=[
            {"role": "user", "content": f"Translate the following {current_language} text into {target_language}: {text}"},
        ]
    ).choices[0].message.content
    return translated_text



if __name__ == "__main__":
    result = search_papers(
        user_input="私は大規模言語モデルのLLaMAに関する論文が読みたいです。",
        input_pdf=None,
        search_papers_amount=1
    )
    # save as json
    with open("parsed_entries.json", "w") as f:
        json.dump(result, f, indent=4)

    # download pdfs fetched by search_papers function
    download_paper_pdfs(result)

    # summarize target information in paper
    summary_result = summarize_target_info_in_paper(
        pdf_file="pdfs/Code_Llama_Open_Foundation_Models_for_Code.pdf", # path to pdf file
        target_information="詳しく論文で提案された手法を説明してもらえますか？", # target information to summarize
        output_language="Japanese" # output language of the summarization
    )
    print(summary_result)

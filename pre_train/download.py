import urllib.request

# 下载gpt_download.py
url = ("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch05/"
        "01_main-chapter-code/gpt_download.py")
filename = url.split("/")[-1]
urllib.request.urlretrieve(url, filename)
# -*- coding:utf-8 -*-
import re
import requests
import csv

'''
最多50页
'''


def crawl(start_url):
    base_url = 'http://www.zgshige.com'

    req_headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
    }

    restart_url = start_url
    print("主页面： ", restart_url)

    res = requests.get(restart_url, headers=req_headers)
    if res.status_code == requests.codes.ok:
        html = res.text

        # 获取所有诗人的链接 <li><a href="/c/2016-05-29/1295595.shtml" class="p-t-xs p-b-xs block text-center" target="_blank">çç«åµ</a></li>
        parttern_href = re.compile(
            r'<li><a href="(.*?)" class="p-t-xs p-b-xs block text-center" target="_blank">.*?</a>',flags=re.DOTALL)
        hrefs = re.findall(parttern_href, html)

        # 获取每一首诗的内容,并保存到本地
        # with open('poem.csv', mode='a', encoding='utf-8') as f:
        with open('writer.csv', 'a', encoding='gb18030') as f:
            write = csv.writer(f, lineterminator='\n')
            for href in hrefs:
                href = base_url + href
                print("子页面： ", href)
                res = requests.get(href, headers=req_headers)
                if res.status_code == requests.codes.ok:
                    html = res.content
                    html = str(html, 'utf-8')  # html_doc=html.decode("utf-8","ignore")

                    # 作者
                    parttern_author = re.compile(r'<div class="sr_n_v_box">.*?<p><a href="#">(.*?)</a ></p ></div>',re.DOTALL)  # ??????
                    author = re.findall(parttern_author, html)
                    print("author", author)
                  #  print(author[0])

                    # 简介
                    parttern_introduction = re.compile(r'<div class="ps-sm">.*?<p>(.*?)</p >',re.DOTALL)  # ????????
                    introduction = re.findall(parttern_introduction, html)
                    print(introduction)
                  #  print(introduction[0])





                    # content[0].repace("<br>", "")
                    # print(type(content[0]))
                    # content[0] = content[0].replace("&nbsp;", " ")




if __name__ == '__main__':
    start_url = 'http://www.zgshige.com/tjsr/'    #
    # 每一页
    with open('writer.csv', 'a', encoding='gb18030') as f:
        row = ['author','introduction']#
        write = csv.writer(f, lineterminator='\n')
        write.writerow(row)
    for i in range(2, 100):
        start_url1 = start_url + 'index_' + str(i) + '.shtml'
        crawl(start_url1)
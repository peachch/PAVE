import requests  # For sending HTTP requests
import pandas as pd  # 用于数据整理和导出
from bs4 import BeautifulSoup, Tag  # 用于解析HTML页面
import re  # 用于正则提取时间
import time  # 用于请求延迟
import config
logger = config.logger

BASE_URL = 'https://en.wikipedia.org'  # 维基百科主域名
YEAR_PAGE = BASE_URL + '/wiki/2025'  # 2024年页面
MAX_RESULTS = 500000  # 采集条数上限

results = []  # 存储所有采集结果

# 提取月份页面链接
def get_month_links():
    resp = requests.get(YEAR_PAGE, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, 'html.parser')
    month_links = []
    # 查找所有2024年月份链接（如January_2024、February_2024...December_2024）
    for a in soup.select('a[href^="/wiki/"]'):
        if not isinstance(a, Tag):
            continue
        text = a.get_text(strip=True)
        # 只采集2024年各月
        if re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December)$', text):
            href = a['href'] if 'href' in a.attrs else None
            # 只保留链接中包含'_2024'的页面
            if href and isinstance(href, str) and href.startswith('/wiki/') and '_2025' in href:
                month_links.append(BASE_URL + href)
    # 去重
    return list(dict.fromkeys(month_links))

# 解析每个月份页面，采集事件
def parse_month_page(month_url, year):
    global results
    resp = requests.get(month_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, 'html.parser')
    # 获取当前月份和年份
    m = re.search(rf'/([A-Za-z]+)_{year}', month_url)
    if m:
        month_str = m.group(1)
        year = year
        month = str(['January','February','March','April','May','June','July','August','September','October','November','December'].index(month_str)+1)
    else:
        return
    # 优化：遍历所有li，查找日期li和事件li
    # for li in soup.find_all('li'):
    for li in soup.find_all('li'):
        # if 'marker' in li.get_text():
        time.sleep(0.1)  # 避免请求过快被封禁
        # lis_with_marker.append(li)
        # 只处理Tag类型li，防止类型错误
        if not isinstance(li, Tag):
            continue
        # 检查li下第一个a标签是否为日期
        a = li.find('a')
        if not (a and isinstance(a, Tag)):
            continue
        date_text = a.get_text(strip=True)
        m_date = re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})$', date_text)
        if m_date:
            # 采集事件li
            fact = None
            resource = None  # 默认用月份页面URL
            # 这是日期li，记录当前日期
            day = m_date.group(2)
            current_month = m_date.group(1)
            current_day = day
            # 检查是否有子ul作为事件列表
            event_ul = li.find('ul') if isinstance(li, Tag) else None
            if event_ul and isinstance(event_ul, Tag):
                for event_li in event_ul.find_all('li', recursive=False):
                    fact = ""
                    resource = "" 
                    time.sleep(0.1) 
                    if not isinstance(event_li, Tag):
                        continue
                    for a in event_li.find_all('a'):
                        if not isinstance(a, Tag):
                            continue
                        href = a.attrs.get('href')
                        if href and isinstance(href, str) and href.startswith('/wiki/') and year in href:
                            # fact = a.get_text(strip=True)
                            resource = BASE_URL + href  # 详情页URL
                            break
                    if not fact:
                        # fact = event_li.get_text(strip=True).split('.')[0]
                        fact = event_li.get_text()
                    evidence = event_li.get_text(strip=True)
                    # 查找分类（向上找最近的h2-h5）
                    cat = 'Event'
                    prev = li
                    while prev:
                        prev = prev.find_previous(['h2','h3','h4','h5'])
                        if prev and isinstance(prev, Tag):
                            cat_candidate = prev.get_text(strip=True)
                            if cat_candidate and not re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December)', cat_candidate):
                                cat = cat_candidate
                                break
                    results.append({
                        'Facts': fact,
                        'Evidence': evidence,
                        'Time-year': year,
                        'Time-month': current_month,
                        'Time-day': current_day,
                        'Spotlight': 'NA',
                        'Category': cat,
                        'Resources': resource
                    })
                    logger.info("collected: %s %s-%s-%s %s %s", fact, year, current_month, current_day, cat, resource)
                    if len(results) >= MAX_RESULTS:
                        return
            else:
                # 没有子ul，可能是单独的日期li，直接记录
                fact = date_text
                # resource = month_url
                evidence = date_text
                # 查找分类（向上找最近的h2-h5）
                cat = 'Date'
                prev = li
                # while prev:
                #     prev = prev.find_previous(['h2','h3','h4','h5'])
                #     if prev and isinstance(prev, Tag):
                #         cat_candidate = prev.get_text(strip=True)
                #         if cat_candidate and not re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December)', cat_candidate):
                #             cat = cat_candidate
                #             break
                # 如果li.find('ul')失败，直接读取get_text
                fact = li.get_text()
                evidence = li.get_text()
                resource = ""  # 默认用月份页面URL
                for a in li.find_all('a'):
                    if not isinstance(a, Tag):
                        continue
                    href = a.attrs.get('href')
                    if href and isinstance(href, str) and href.startswith('/wiki/') and year in href:
                        # fact = a.get_text(strip=True)
                        resource = BASE_URL + href  # 详情页URL
                        break
                results.append({
                    'Facts': fact,
                    'Evidence': evidence,
                    'Time-year': year,
                    'Time-month': current_month,
                    'Time-day': current_day,
                    'Spotlight': 'NA',
                    'Category': cat,
                    'Resources': resource
                })
                logger.info("collected: %s %s-%s-%s %s %s", fact, year, current_month, current_day, cat, resource)
                if len(results) >= MAX_RESULTS:
                    return

def death_parse_month_page(month_url, year, month):
    global results
    resp = requests.get(month_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, 'html.parser')
    # time.sleep(0.1)  # 避免请求过快被封禁
    for i in range(1, 31):
        time.sleep(0.1)
        h3 = soup.find('h3', id=str(i))
        if h3:
            # 2. 找到父div
            parent_div = h3.find_parent('div')
            if parent_div:
                # 3. 从div后找ul
                next_node = parent_div
                ul = None
                while next_node:
                    next_node = next_node.next_sibling
                    if isinstance(next_node, Tag) and next_node.name == 'ul':
                        ul = next_node
                        break
                # 4. 遍历li
                if ul:
                    for li in ul.find_all('li', recursive=False):
                        time.sleep(0.1)
                        logger.debug(li.get_text(strip=True))
                        fact = li.get_text(strip=True)
                        evidence = None
                        resource = None  
                        for a in li.find_all('a'):
                            if not isinstance(a, Tag):
                                continue
                            href = a.attrs.get('href')
                            if href and isinstance(href, str) and href.startswith('/wiki/'):
                                # fact = a.get_text(strip=True)
                                resource = BASE_URL + href  # 详情页URL
                                break
                        results.append({
                            'Facts': fact,
                            'Evidence': evidence,
                            'Time-year': year,
                            'Time-month': month,
                            'Time-day': h3.get_text(),
                            'Spotlight': 'NA',
                            'Category': "human death",
                            'Resources': resource
                        })
        

def current_parse_month_page(month_url, year, month, count):
    global results
    resp = requests.get(month_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, 'html.parser')
    # time.sleep(0.1)  # 避免请求过快被封禁
    target_div = soup.find_all('div', class_='current-events-content description')
    if target_div:
        day = 1
        for div in target_div:
            # 2. 只在这个div里查找<p><b>xxx</b></p>
            for p in div.find_all('p'):
                time.sleep(0.1)
                b_tag = p.find('b')
                if b_tag:
                    category = b_tag.get_text(strip=True)
                    fact_items = []
                    # parent_div = p.next_sibling('ul')
                    # li = p.find_next_sibling(lambda tag: isinstance(tag, Tag) and tag.name == 'li')
                    ul = p.find_next_sibling(lambda tag: isinstance(tag, Tag) and tag.name == 'ul')
                    if ul:
                    #     li_texts = [li.get_text(strip=True) for li in ul.find_all('li')]
                    # # 3. 从该div后找ul
                    # uls = parent_div.find_all('li')
                    # for ul in uls:
                        # ul = li.find_next_sibling(lambda tag: isinstance(tag, Tag) and tag.name == 'ul')
                        # a = li.find_all("a")
                        # time.sleep(0.1)
                        for li in ul.find_all('li', recursive=False):  # 只取当前ul的直接li
                            li_text = li.get_text()
                            full_url = ""
                            # all_li_facts.append(li_text)
                            # fact_items.append(li_text)
                    
                            # 保存结果
                            # if fact_items:
                            for a_tag in li.find_all('a', href=True):
                                href = a_tag['href']
                                a_class = a_tag.get('class', [])
                                if year in href and "external" not in a_class:
                                    # 处理相对路径（如./xxx/xx 或 /xx/xx）
                                    full_url = f"{BASE_URL}{href}"
                                    count += 1
                                    
                            results.append({
                                'Facts': li_text,
                                'Evidence': None,
                                'Time-year': year,
                                'Time-month': month,
                                'Time-day': day,
                                'Spotlight': 'NA',
                                'Category': category,
                                'Evidence_resource': full_url
                            })

                            logger.info("collected: %s %s-%s-%s %s", li_text, year, month, day, category)
            day += 1
    logger.info("collected count: %d", count)

if __name__ == '__main__':
    
    # month_links = get_month_links() 
    # import ipdb
    # print(f'----{len(month_links)} pages。')
    # month_links = ["https://en.wikipedia.org/wiki/Portal:Current_events/June_2025"]
    # month_links = ["https://en.wikipedia.org/wiki/2025_Polish_Presidency_of_the_Council_of_the_European_Union"]
    # month_links = ["https://en.wikipedia.org/wiki/January_2025", "https://en.wikipedia.org/wiki/February_2025", "https://en.wikipedia.org/wiki/March_2025", "https://en.wikipedia.org/wiki/April_2025", "https://en.wikipedia.org/wiki/May_2025", "https://en.wikipedia.org/wiki/June_2025", "https://en.wikipedia.org/wiki/July_2025", "https://en.wikipedia.org/wiki/August_2025", "https://en.wikipedia.org/wiki/September_2025", "https://en.wikipedia.org/wiki/October_2025", "https://en.wikipedia.org/wiki/November_2025", "https://en.wikipedia.org/wiki/December_2025"]
    
    year = "2024"
    cate = "current_event"  # "month_event" or "current_event" or "death_event"
    
    if cate == "month_event":
        logger.info('collect %s...%s....', year, cate)
        month_links = [f"https://en.wikipedia.org/wiki/January_{year}"]
        
        # 处理月份页面
        for month_url in month_links:
            logger.info('Month: %s', month_url)
            # import ipdb
            # ipdb.set_trace()s 
            parse_month_page(month_url, year)
            if len(results) >= MAX_RESULTS:
                break
        df = pd.DataFrame(results)
        # df.to_csv(f'wjbk_events_{year}_{len(df)}.csv', index=False, encoding='utf-8-sig')
        df.to_json(f'wjbk_events_{year}_{len(df)}.json', orient='records', force_ascii=False, indent=2)
        logger.info('Crawling completed. Saved as wjbk.csv and wjbk_demo_%s_%d', year, len(df))
    
    if cate == "death_event":
        logger.info('collect %s...%s....', year, cate)
        # 处理死亡事件
        if year == "2025":
            months = ["January", "February", "March", "April", "May", "June", "July"]
        else:
            months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        
        
        for month in months:
            logger.info('Month: %s', month)
            death_url = f"https://en.wikipedia.org/wiki/Deaths_in_{month}_{year}"
            # import ipdb
            # ipdb.set_trace()s
            death_parse_month_page(death_url, year, month)
            if len(results) >= MAX_RESULTS:
                break

        df = pd.DataFrame(results)
        # df.to_csv(f'wjbk_death_{year}_{month}.csv', index=False, encoding='utf-8-sig')
        df.to_json(f'wjbk_death_{year}_{len(df)}.json', orient='records', force_ascii=False, indent=2)
        logger.info('Crawling completed. Saved as wjbk.csv and death_wjbk_demo_%s_%s', year, month)


    if cate == "current_event":
        logger.info('collect %s...%s....', year, cate)
        # 事件
        count = 0
        if year == "2025":
            months = ["January", "February", "March", "April", "May", "June", "July", "Angust"]
        else:
            months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        # months = ["January", "February", "March", "April", "May", "June", "July"]
        # months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        for month in months:
            death_month_url = f"https://en.wikipedia.org/wiki/Portal:Current_events/{month}_{year}"
            logger.info('Month: %s', death_month_url)
            # import ipdb
            # ipdb.set_trace()s
            current_parse_month_page(death_month_url, year, month, count)
            if len(results) >= MAX_RESULTS:
                break

        df = pd.DataFrame(results)
        logger.info('total records: %d', len(df))
        df.to_json(f'evidence_url_wjbk_{cate}_{year}_{len(df)}.json', orient='records', force_ascii=False, indent=2)
        # df = df.replace('\n', '\\n', regex=True)
        # df.to_csv('output.csv', index=False)
        # df.to_csv(f'{cate}_wjbk_{year}.csv', index=False, encoding='utf-8-sig')
        logger.info('Crawling completed. Saved as wjbk.csv and %s_wjbk_demo_%s', cate, year)

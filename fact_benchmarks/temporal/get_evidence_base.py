import requests
from bs4 import BeautifulSoup, Tag
import json
import time
import config
logger = config.logger

def event_get_text_until_second_h2(url):
    # time.sleep(0.001)
    human_texts = ""
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        div = soup.find('div', class_='mw-body-content')
        if div:
            for p in div.find_all_next('p'):
                parent = p.find_parent('tbody')
                if parent:
                    continue
                parent = p.find_parent('td')
                if parent:
                    continue
                parent = p.find_parent('tr')
                if parent:
                    continue
                a_tags = p.find_all('a')
                    # if a_tags:
                if a_tags:
                    # 找到第一个包含a的p标签
                    a_texts = p.get_text()
                    # print(a_texts)
                    human_texts = a_texts
                    break

        else:
            logger.warning("没有找到目标div for %s", url)
    except Exception as e:
        logger.exception("Error fetching %s: %s", url, e)
        human_texts = ""
    return human_texts

# Function to get all text from the first h2 to the second h2 (not including the second h2)
def death_get_text_until_second_h2(url):
    # time.sleep(0.001)
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")
            div = soup.find('div', class_='shortdescription nomobile noexcerpt noprint searchaux')
            # div = soup.find('div', class_='xxx')
            human_texts = ""
            if div:
                # 2. 找到紧挨着的下一个p标签
                p = div.find_next_sibling(lambda tag: isinstance(tag, Tag) and tag.name == 'p')
                if p:
                    a_tags = p.find_all('a')
                    # if a_tags:
                    if a_tags:
                    # 3. 获取p下所有a标签的文本
                        human_texts = p.get_text()# 输出: ['链接1', '链接2']
                else:
                    # human_texts = ""
                    logger.warning("no p tag found after the div for %s", url)
    except Exception as e:
        logger.exception("Error fetching %s: %s", url, e)
        human_texts = ""
    text = []
    if human_texts != "":
        return human_texts
    else:
        try:
            div = soup.find('div', class_='mw-body-content')
            if div:
                for p in div.find_all_next('p'):
                    parent = p.find_parent('tbody')
                    if parent:
                        continue
                    parent = p.find_parent('td')
                    if parent:
                        continue
                    parent = p.find_parent('tr')
                    if parent:
                        continue
                    a_tags = p.find_all('a')
                        # if a_tags:
                    if a_tags:
                        # 找到第包含a的p标签
                        parent_div = p.find_previous_sibling('div')
                        if parent_div and 'mw-heading2' in parent_div.get('class', []) and 'mw-heading' in parent_div.get('class', []):
                            human_texts = ' '.join(text)
                            break
                        else:
                            a_texts = p.get_text()
                            human_texts = a_texts
                            text.append(human_texts)
                        # human_texts = ' '.join(text)  # 只取前两个
                                # break
                        # break
            else:
                logger.warning("no div for %s", url)

            try:
                div2 = soup.find_all('div', class_='mw-heading mw-heading2')
                if div2:
                    for div in div2:
                        h2 = div.get_text()
                        if h2 == 'Death[edit]':
                        # if h2 == 'Death':
                            # 找到第一个包含h2的p标签
                            p = div.find_next('p')
                            if p:
                                a_tags = p.find_all('a')
                                if a_tags:
                                    death_text = p.get_text()
                human_texts = human_texts + " " + death_text
            except:
                logger.warning("No death info %s", url)
                human_texts = human_texts

        except Exception as e:
            logger.exception("Error fetching %s: %s", url, e)
            human_texts = ""
    return human_texts

import config
import os

cate = "death" 
# cate = "current_event"  # Set the category to "death"
year = "2025"

# Read wjbk_death_demo_2025_5160.json and process Resources fields
if cate == "death": 
    # Load the JSON data from the file
    if year == "2025":
        path = os.path.join(config.DEFAULT_BASE_DIR, 'fact_bechmarks', 'new_data_crawl', 'wjbk_death_demo_2025_5160.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
# Read wjbk_events_2024_214.json or wjbk_events_2025_155.json and process Resources fields
if cate == "event":
    if year == "2024":
        path = os.path.join(config.DEFAULT_BASE_DIR, 'fact_bechmarks', 'new_data_crawl', 'wjbk_events_2024_214.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    if year == "2025":      
        path = os.path.join(config.DEFAULT_BASE_DIR, 'fact_bechmarks', 'new_data_crawl', 'wjbk_events_2025_155.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
if cate == "current_event":
    if year == "2024":
        path = os.path.join(config.DEFAULT_BASE_DIR, 'evidence_url_wjbk_current_event_2024_4411.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    if year == "2025":      
        path = os.path.join(config.DEFAULT_BASE_DIR, 'evidence_url_wjbk_current_event_2025_2588.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
count = 0
results = []
for item in data:
    # time.sleep(0.1)
    url = item.get('Resources', '')
    if cate == "death":
        evidence = death_get_text_until_second_h2(url)
        item['evidence'] = evidence
        time.sleep(0.1)
        logger.info("Processed URL: %s", url)
        with open('evidence_log.txt', 'a', encoding='utf-8') as logf:
            logf.write(f"Processed URL: {url}, Evidence: {evidence}\n")
    if cate == "event":
        evidence = event_get_text_until_second_h2(url)
        item['evidence'] = evidence
        time.sleep(0.1)
        logger.info("Processed URL: %s", url)
        with open('evidence_log.txt', 'a', encoding='utf-8') as logf:
            logf.write(f"Processed URL: {url}, Evidence: {evidence}\n")
    if cate == "current_event":
        url = item.get('Evidence_resource', '')
        if url == "":
            continue
        evidence = event_get_text_until_second_h2(url)
        item['evidence'] = evidence
        time.sleep(0.1)
        logger.info("Processed URL: %s", url)
        count += 1
        results.append({
            'Facts': item.get('Facts', ''),
            'Evidence': evidence,
            'Time-year': year,
            'Time-month': item.get('Time-month', ''),
            'Time-day': item.get('Time-day', ''),
            'Spotlight': 'NA',
            'Category': item.get('Category', ''),
            'Resources': url
                        })
        
        with open('current_event_evidence_log.txt', 'a', encoding='utf-8') as logf:
            logf.write(f"Processed URL: {url}, Evidence: {evidence}\n")

logger.info("Total processed current_event items: %d", count)

if cate == "death":
    # Save the updated data back to the JSON file
        outp = os.path.join(config.DEFAULT_BASE_DIR, 'fact_bechmarks', 'new_data_crawl', 'wjbk_death_demo_2025_5160_evidence.json')
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info('wjbk_death_demo_2025_5160_evidence.json updated with evidence field.')
if cate == "event":
    # Save the updated data back to the JSON file
        if year == "2024":
            outp = os.path.join(config.DEFAULT_BASE_DIR, 'fact_bechmarks', 'new_data_crawl', 'wjbk_events_2024_214_evidence.json')
            with open(outp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        if year == "2025":  
            outp = os.path.join(config.DEFAULT_BASE_DIR, 'fact_bechmarks', 'new_data_crawl', 'wjbk_events_2025_155_evidence.json')
            with open(outp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

if cate == "current_event":
    # Save the updated data back to the JSON file
        if year == "2024":
            outp = os.path.join(config.DEFAULT_BASE_DIR, 'fact_bechmarks', 'new_data_crawl', 'wjbk_events_2024_4411_evidence.json')
            with open(outp, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        if year == "2025":  
            outp = os.path.join(config.DEFAULT_BASE_DIR, 'fact_bechmarks', 'new_data_crawl', 'evidence_url_wjbk_current_event_2025_2588_evidence.json')
            with open(outp, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

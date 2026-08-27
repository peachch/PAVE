import json, os, random, re, time
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY','')
OPENAI_BASE_URL=os.environ.get('OPENAI_BASE_URL','https://api.openai.com/v1')
_THIS_DIR=os.path.dirname(os.path.abspath(__file__))
BASE_DIR=os.environ.get('BASE_DIR',_THIS_DIR)
DATA_ROOT=os.environ.get('PCD_DATA_ROOT',BASE_DIR)
OUTPUT_ROOT=os.environ.get('PCD_OUTPUT_ROOT',BASE_DIR)
_MOCK=False
_client=None

def set_mock(flag):
    global _MOCK; _MOCK=bool(flag)
def ensure_dir(path):
    d=os.path.dirname(path)
    if d: os.makedirs(d,exist_ok=True)
def read_jsonl(path):
    out=[]
    with open(path,encoding='utf-8') as f:
        head=f.read(1); f.seek(0)
        if head=='[': return json.load(f)
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def read_json(path):
    with open(path,encoding='utf-8') as f: return json.load(f)

def write_json(path,obj):
    ensure_dir(path)
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
def parse_extra_body(raw):
    if not raw: return None
    return json.loads(raw)
def add_llm_args(parser):
    parser.add_argument('--max-tokens',type=int,default=None)
    parser.add_argument('--extra-body',type=str,default=None)
    parser.add_argument('--mock-llm',action='store_true')
def llm_request(prompt, model_name, temperature, max_tokens=None, extra_body=None):
    if _MOCK: return random.choice(['support','refute'])
    global _client
    if _client is None:
        from openai import OpenAI
        _client=OpenAI(api_key=OPENAI_API_KEY,base_url=OPENAI_BASE_URL)
    kw={'model':model_name,'temperature':temperature,'messages':[{'role':'user','content':prompt}]}
    if max_tokens is not None: kw['max_tokens']=max_tokens
    if extra_body: kw['extra_body']=extra_body
    r=_client.chat.completions.create(**kw)
    return (r.choices[0].message.content or '').strip()

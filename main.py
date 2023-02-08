import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.utilities import RequestsWrapper
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import ConversationChain, OpenAI, PromptTemplate, VectorDBQA, VectorDBQAWithSourcesChain, SerpAPIWrapper, LLMChain, LLMCheckerChain, LLMMathChain, SQLDatabase, SQLDatabaseChain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, ConversationalAgent
from langchain.cache import InMemoryCache
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.chains import LLMRequestsChain, LLMChain
from gpt_index import GPTListIndex, SimpleWebPageReader
import re

openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

template = """Between >>> and <<< are the raw search result text from google.
Extract the answer to the question '{query}' or say "not found" if the information is not contained.
Use the format
Extracted:<answer or "not found">
>>> {requests_result} <<<
Extracted:"""

PROMPT = PromptTemplate(
  input_variables=["query", "requests_result"],
  template=template,
)

requests_chain = LLMRequestsChain(llm_chain=LLMChain(
  llm=OpenAI(temperature=0, openai_api_key=openai_api_key), prompt=PROMPT))


def create_prompt(tools, prefix, suffix, input_variables):
  prompt = ZeroShotAgent.create_prompt(tools=tools,
                                       prefix=prefix,
                                       suffix=suffix,
                                       input_variables=input_variables)
  return prompt


def extract_links(text):
  url_extract_pattern = "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]+)"
  links = re.findall(url_extract_pattern, text)
  text_without_links = re.sub(url_extract_pattern, '', text)
  return text_without_links, links


def search_links_simple_reader(query, links):
  documents = SimpleWebPageReader(html_to_text=True).load_data(links)
  index = GPTListIndex(documents)
  response = index.query(query, verbose=True)
  return response


reader_tool_desc = "useful for when you need to get specific content from a site. Use this tool instead of Requests tool, because Requests uses too many model tokens."


def reader_func(q):
  text, links = extract_links(q)
  print("text:", text, links)
  return search_links_simple_reader(text, links)


def define_tools():
  search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
  requests = RequestsWrapper()
  # google = "https://www.google.com/search?q=" + query.replace(" ", "+")
  # inputs = {
  #     "query": query,
  #     "url": google
  # }
  req_chain = requests_chain

  tools = [
    # Tool(
    #   name="Google Query",
    #   func=lambda q: req_chain(
    #     {
    #       "query": q,
    #       "url": "https://www.google.com/search?q=" + q.replace(" ", "+")
    #     }),
    #   description=
    #   "useful for when you need to get specific content about a topic, from Google. Input should be a specific url, and the output will be all the text on that page."
    # ),
    # Tool(
    #   name="Requests",
    #   func=lambda q: str(requests.run(q)),
    #   description=
    #   "useful for when you need to get specific content from a site. Input should be a specific url, and the output will be all the text on that page.",
    # ),
    Tool(name="Search",
         func=search.run,
         description=
         "useful for when you need to answer questions about current events"),
    # Tool(name="Link and Site Crawler",
    #      func=lambda q: reader_func(q),
    #      description=reader_tool_desc)
  ]
  return tools


def create_llm_chain(llm, prompt, verbose=True):
  return LLMChain(llm=llm, prompt=prompt, verbose=verbose)


def create_agent(llm_chain, tools):
  return ZeroShotAgent(llm_chain=llm_chain, tools=tools)


def create_agent_executor(agent, tools, verbose):
  return AgentExecutor.from_agent_and_tools(agent=agent,
                                            tools=tools,
                                            verbose=verbose)


prefix = """Answer each question individually by using all of our tools to get the answer for each individual question. First, ask the base language model if it knows anything, then use our tools to search the internet. use wikipedia and google as well. Search the internet and looks at different links until you get the answer. Give answers with at least 3-5 sentences of substance.
You have access to the following tools:"""
suffix = """
Questions: {input}
{agent_scratchpad}"""


def create_search_agent(tools):
  # pre = template + " " + prefix
  input = ["input", "agent_scratchpad"]
  prompt = create_prompt(tools=tools,
                         prefix=prefix,
                         suffix=suffix,
                         input_variables=input)
  memory = ConversationBufferMemory(memory_key="chat_history")
  llm = OpenAI(temperature=0.8,
               top_p=1,
               frequency_penalty=0,
               presence_penalty=0,
               max_tokens=2000,
               openai_api_key=openai_api_key)
  llm_chain = create_llm_chain(llm=llm, prompt=prompt, verbose=True)

  # agent = initialize_agent(
  #     tools, llm_chain, agent="zero-shot-react-description", verbose=True, memory=memory)
  agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, memory=memory)
  return agent


def run_agent_executor(agent, tools, query):
  print('run_agent_executor')
  agent_executor = create_agent_executor(agent, tools, True)
  result = agent_executor.run(input=query, verbose=True)
  return result


# tools = define_tools()
# agent = create_search_agent(tools)
# company = "Google"
# q="Query"
# run_agent_executor(agent, tools, query=q, company=company)

app = Flask(__name__)
CORS(app)


@app.route('/ylang', methods=['POST'])
def receive_input():
  query = request.get_json().get('query')
  print('query', query)
  tools = define_tools()
  agent = create_search_agent(tools)
  result = run_agent_executor(agent, tools, query=query)
  return jsonify(result)


@app.route('/')
def index():
  return 'Hello from Flask!'


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=81)

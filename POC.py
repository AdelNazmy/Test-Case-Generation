#!/usr/bin/env python
# coding: utf-8

# Use Cases to Discuss with ESS:
# 
# 1.  **Test Case Generation:** Generate test cases based on BRD/solution documents.
# 2.  **Visual Test Case Generation:** Automate the generation of test cases for visual testing.
# 3.  **AI-Driven Test Data Generation:** Generate synthetic test data for diverse testing scenarios.

# ---

# ## 1. **Test Case Generation:** Generate test cases based on BRD/solution documents.

# ### Saleor e-commerce (hosted locally)

# ![image-2.png](attachment:image-2.png)

# In[3]:


from IPython.display import display, Markdown


# ### ***BRD Document Contents***

# In[4]:


with open('Test-Case-Generation/source/saleor_brd.txt', 'r') as f:
    brd_text = f.read()
    display(Markdown(brd_text))


# ### ***Generate Test Cases from BRD Document***

# In[5]:


get_ipython().system('ollama list')


# ### Build the prompt

# In[6]:


from llama_index.core.prompts import RichPromptTemplate
import time

template_str = """You are an expert test engineer who excels in creating test cases for web products, create test cases from the following Business Requirements Document (BRD):
---------------------
{{ context_str }}
---------------------
"""
qa_template = RichPromptTemplate(template_str)

# you can create text prompt (for completion API)
prompt = qa_template.format(context_str=brd_text)

# or easily convert to message prompts (for chat API)
messages = qa_template.format_messages(context_str=brd_text)


# ### 1st run set model temperature from 0.1 to retrive deterministic results (less creativity)

# In[7]:


from llama_index.llms.ollama import Ollama


llm= Ollama(model="gemma3:12b", temperature=0.1,context_window=16384,request_timeout=500)
gen=llm.stream_chat(messages=messages)

stream_output = display(Markdown(""), display_id=True)  # Empty initial display
full_response = ""

for response in gen:
    full_response += response.delta
    stream_output.update(Markdown(full_response))  # Update the same output
    time.sleep(0.01)  # Small delay for smoother rendering


# ### 2nd run modified model temperature from 0.1 to 0.7 to increase model creativity

# In[8]:


llm= Ollama(model="gemma3:12b", temperature=0.7,context_window=16384,request_timeout=500)

gen=llm.stream_chat(messages=messages)

stream_output = display(Markdown(""), display_id=True)  # Empty initial display
full_response = ""

for response in gen:
    full_response += response.delta
    stream_output.update(Markdown(full_response))  # Update the same output
    time.sleep(0.01)  # Small delay for smoother rendering


# # Using OpenAI GPT-4.1-mini instead 

# In[4]:


from llama_index.llms.openai import OpenAI

llm= OpenAI(model="gpt-4.1-mini", temperature=0.1, max_tokens=4096, request_timeout=500)

gen=llm.stream_chat(messages=messages)

stream_output = display(Markdown(""), display_id=True)  # Empty initial display
full_response = ""

for response in gen:
    full_response += response.delta
    stream_output.update(Markdown(full_response))  # Update the same output
    time.sleep(0.01)  # Small delay for smoother rendering


# ### 2nd run modified model temperature from 0.1 to 0.9 to increase model creativity

# In[5]:


llm= OpenAI(model="gpt-4.1-mini", temperature=0.9, max_tokens=4096, request_timeout=500)

gen=llm.stream_chat(messages=messages)

stream_output = display(Markdown(""), display_id=True)  # Empty initial display
full_response = ""

for response in gen:
    full_response += response.delta
    stream_output.update(Markdown(full_response))  # Update the same output
    time.sleep(0.01)  # Small delay for smoother rendering


# ---

# # 2. **Visual Test Case Generation:** Automate the generation of test cases for visual testing

# ### Using GPT-4.1-mini vision capability 

# ### Using selenium to get screenshots

# In[13]:


from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def take_screenshot(url, output_path="screenshot.png", window_size=(1920, 1080)):
    """
    Takes a screenshot of the given website URL and saves it to output_path.

    Args:
        url (str): The website URL to capture.
        output_path (str): Path to save the screenshot image.
        window_size (tuple): Browser window size (width, height).
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        driver.save_screenshot(output_path)
    finally:
        driver.quit()


# In[14]:


url="http://localhost:9000/dashboard/products/?asc=false&sort=date"

take_screenshot(url, output_path="Test-Case-Generation/screenshot.png", window_size=(1920, 1080))


# In[16]:


from IPython.display import Image
Image("Test-Case-Generation/screenshot.png")


# In[13]:


from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4.1-mini", temperature=0.1, max_tokens=8192, request_timeout=500)

messages = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path="Test-Case-Generation/screenshot.png"),
            TextBlock(text="Create test cases from the following screenshot."),
        ],
    )
]

resp = llm.chat(messages)


# In[14]:


display(Markdown(resp.message.blocks[0].text))


# #### Using pre-fetched screenshots

# In[17]:


Image("Test-Case-Generation/Screenshot 2025-06-27 101226.png")


# In[18]:


messages = [
    ChatMessage(
    role="system",
    blocks=[TextBlock(text="You are an expert test engineer who excels in creating test cases for web products")]
    ),
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path="Test-Case-Generation/Screenshot 2025-06-27 101226.png"),
            TextBlock(text="Create test cases from the following screenshot."),
        ],
    )
]

resp = llm.chat(messages)


# In[19]:


display(Markdown(resp.message.blocks[0].text))


# ## 3. **AI-Driven Test Data Generation:** Generate synthetic test data for diverse testing scenarios.

# In[41]:


from llama_index.core.memory import ChatMemoryBuffer

# Create a chat memory buffer to store conversation history
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

# Example: add the current messages to memory
for msg in resp:
    chat_memory.put(msg)


# In[42]:


# Use the chat history stored in chat_memory to generate synthetic test data with the LLM

# Prepare a new prompt using the chat history as context
history_text = "\n\n".join(
    block.text
    for entry in chat_memory.chat_store.store['chat_history']
    if isinstance(entry, tuple) and hasattr(entry[1], "blocks")
    for block in entry[1].blocks
    if hasattr(block, "text")
)

data_prompt = [
    ChatMessage(
        role="system",
        blocks=[TextBlock(text="You are an expert test data generator for QA. Based on the following test cases, generate a table of synthetic test data suitable for automated testing. Include edge cases and realistic values.")]
    ),
    ChatMessage(
        role="user",
        blocks=[TextBlock(text=history_text)]
    ),
]

data_resp = llm.chat(data_prompt)
display(Markdown(data_resp.message.blocks[0].text))


# In[45]:


messages 


# ### Using Local Model to generate test cases via vision 

# In[46]:


from llama_index.llms.ollama import Ollama

llm= Ollama(model="gemma3:12b", temperature=0.1,context_window=16384,request_timeout=500)
gen=llm.stream_chat(messages=messages)


# In[47]:


stream_output = display(Markdown(""), display_id=True)  # Empty initial display
full_response = ""

for response in gen:
    full_response += response.delta
    stream_output.update(Markdown(full_response))  # Update the same output
    time.sleep(0.01)  # Small delay for smoother rendering


# ### Using Local Model to generate test data for generated test cases

# In[48]:


#Markdown(data_prompt[1].blocks[0].text)
data_prompt


# In[49]:


data_resp = llm.chat(data_prompt)


# In[51]:


display(Markdown(data_resp.message.blocks[0].text))


# # Scratch Book
# 

# In[ ]:


get_ipython().system('huggingface-cli login')


# In[5]:


from llama_index.multi_modal_llms.huggingface import HuggingFaceMultiModal
from llama_index.core.schema import ImageDocument

# Initialize the model
model = HuggingFaceMultiModal.from_model_name("google/gemma-3n-E2B-it")

# Prepare your image and prompt
image_document = ImageDocument(image_path="Test-Case-Generation/Screenshot 2025-06-27 101226.png")
prompt = "Describe this image in detail."

# Generate a response
response = model.complete(prompt, image_documents=[image_document])

print(response.text)


# In[ ]:im





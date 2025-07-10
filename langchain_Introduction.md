### langchain is opensource framwork for apllucation dev powred by LLMs.
# why
# Support all LLMs 
# intergrity  with major tools
# all genai uses

## How sementic seach works?
### convert para in vector and convert query on vector and find distace b/w those vectors
### need doc loader for pdf loadring, then need text spitter for pdf text spilling and vector embeeer 
### most challging is orsechration 

# Why usefule
# have chains(for complecx workflows) o/p of one become i/p for other
# model angostic
# compelete ecosystem
## memort and state handling


# what we can built`
## conversation chatbot
## ai knowledge assistance
## ai agents
## workflow automation
## summization / research 

# Langchain components 

## Models
### core interface to intervact with models
### size of llm is large, hence api for models
### implemantion of diff api is not same, hence coding is diff . chnaging api is hard --> langchain give standarsation
### provide interface
### langchain has language models and embedding models ( for semetic serach )

## promts
### input provided to LLMs
### dynamic and resuseble promts ( Expamples of this )
### Role based promts
### Few short promting ( Exmaple of this )

## chains
### to built pipelined in langchain
### one stage o/p is next stange i/p
### can make complex chain( parallel chain, condition chain )

## Indexes
### it connects your appliction fo external knowldge
### has doc loader , text spillter , vector stroage , retrivers 

## memory
### LLm api calls are stateless
### conversation buffer memory ( store all memory of a chat)
### conversation buffer memory window buffer 
### summerise bassed memort (peroidically send summery of chat)

## Agents
### have NLU and NLG 
### have reasoning capabilites and have tools ( using chain of thoughts)


